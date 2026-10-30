import os
import sys
from pathlib import Path

try:
    from .file_names import AVL_MAX_NAME, short_name
except ImportError:  # run as a script: python src/simulation_support/avl_simulation.py (self-test)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from file_names import AVL_MAX_NAME, short_name

IS_WINDOWS = sys.platform == "win32"

# Windows drives AVL through winpty, Linux through pexpect
if IS_WINDOWS:
    import multiprocessing.pool
    from winpty import PtyProcess
else:
    import pexpect

class AVL_Run_Case:
    """
    A class to automate AVL simulations for a batch of geometry files.
    Works on Windows (winpty) and Linux (pexpect); the platform is detected automatically.

    This class handles the creation of a temporary directory,
    copies necessary files, runs AVL for each geometry, copies the
    results to a specified output folder, and cleans up all temporary files.
    """
    def __init__(self, geo_files: list, case_file: str, avl_exe_path: str, output_dir: str, debug: bool = False):
        """
        Initializes the AVLBatchProcessor.

        Args:
            geo_files (list): A list of absolute string paths to the geometry (.avl) files.
            case_file (str): An absolute string path to the case file.
            avl_exe_path (str): An absolute string path to the AVL executable (avl.exe on Windows).
            output_dir (str): An absolute string path to the folder where output force files will be saved.
            debug (bool, optional): If True, enables detailed console messages. Defaults to False.
        """
        if not os.path.exists(avl_exe_path):
            raise FileNotFoundError(f"AVL executable not found at: {avl_exe_path}")

        self.geo_files = [Path(p) for p in geo_files]
        self.case_file = case_file
        self.avl_exe = os.path.abspath(avl_exe_path)  # absolute: AVL is started inside the output folder
        self.output_dir = Path(output_dir)
        self.debug = debug

        self.output_dir.mkdir(exist_ok=True)
        # Fail now, not after the batch, if AVL could not be given a file name in full
        for geo_file in self.geo_files:
            self._avl_file_names(geo_file)

    def _force_file_path(self, geo_file):
        return self.output_dir / f"force_{Path(geo_file).stem.replace('_geo', '')}.txt"

    def _avl_file_names(self, geo_file):
        """
        (geometry, case, force file) names as given to AVL, which runs inside the output folder. AVL silently cuts
        the force file name at 80 characters (see simulation_support.file_names), so every name is kept as short
        as possible. Raises ValueError if a name still does not fit.
        """
        return tuple(short_name(path, self.output_dir, AVL_MAX_NAME, program="AVL")
                     for path in (geo_file, self.case_file, self._force_file_path(geo_file)))

    def _spawn(self):
        if IS_WINDOWS:
            return PtyProcess.spawn([str(self.avl_exe)], cwd=str(self.output_dir))
        return pexpect.spawn(str(self.avl_exe), encoding='utf-8', cwd=str(self.output_dir))

    def _send(self, proc, command: str, expect_prompt: str = "AVL   c>", timeout_sec: int = 30):
        """Sends a command to the AVL process and waits for the expected prompt."""
        if self.debug:
            print(f"[DEBUG] Sending command: {command}")
        if IS_WINDOWS:
            received, closed = self._winpty_exchange(proc, command, expect_prompt, timeout_sec)
        else:
            received, closed = self._pexpect_exchange(proc, command, expect_prompt, timeout_sec)

        if received:
            if self.debug:
                print(f"[DEBUG] Prompt '{expect_prompt}' received.")
            return True
        if closed:
            # If the command was to quit, an EOF is a successful exit
            if command == "QUIT":
                return True
            if self.debug:
                print("[DEBUG] EOFError encountered (Process closed unexpectedly).")
            return False
        if self.debug:
            print(f"[DEBUG] Timeout waiting for prompt '{expect_prompt}' after sending '{command}'.")
        return False

    def _pexpect_exchange(self, proc, command, expect_prompt, timeout_sec):
        """Returns (prompt received, process closed)."""
        try:
            proc.sendline(command)
            proc.expect_exact(expect_prompt, timeout=timeout_sec)
            return True, False
        except pexpect.TIMEOUT:
            return False, False
        except pexpect.EOF:
            return False, True

    def _winpty_exchange(self, proc, command, expect_prompt, timeout_sec):
        """Returns (prompt received, process closed). winpty reads block, hence the thread timeout."""
        def exchange():
            output = ""
            try:
                proc.write(command + "\r\n")
                while True:
                    chunk = proc.read(64)
                    if not chunk:
                        return False, True
                    output += chunk
                    if expect_prompt in output:
                        return True, False
            except EOFError:
                return False, True

        pool = multiprocessing.pool.ThreadPool(processes=1)
        try:
            return pool.apply_async(exchange).get(timeout_sec)
        except multiprocessing.TimeoutError:
            return False, False
        finally:
            pool.terminate()

    def _execute_avl_commands(self, proc, geo_file):
        """
        Executes the sequence of AVL commands.
        Returns True on success, False on timeout/failure.
        """
        geo_name, case_name, force_name = self._avl_file_names(geo_file)
        if not self._send(proc, f"LOAD {geo_name}"): return False
        if not self._send(proc, f"CASE {case_name}"): return False
        if not self._send(proc, "OPER", expect_prompt='.OPER (case 1/1)   c>'): return False
        if not self._send(proc, "X", expect_prompt='.OPER (case 1/1)   c>'): return False
        if not self._send(proc, "FT", expect_prompt='Enter filename, or <return> for screen output   s>'): return False
        if not self._send(proc, force_name, expect_prompt='.OPER (case 1/1)   c>'): return False
        if not self._send(proc, ""): return False # Advance to the AVL prompt
        self._send(proc, "QUIT")
        return True

    def run(self):
        """
        Orchestrates the entire batch simulation process.
        Runs AVL for each geometry file
        """
        # A try...finally block ensures cleanup happens even if errors occur
        try:
            retry = 3

            # --- Loop through each geometry file ---
            for geo_file in self.geo_files:
                success = False
                for attempt in range(1, retry + 1):
                    proc = None
                    try:
                        if self.debug:
                            print(f"\n--- Processing {geo_file.name} (Attempt {attempt}/{retry}) ---")

                        proc = self._spawn()

                        # --- AVL Command Sequence (names relative to the output folder) ---
                        if self._execute_avl_commands(proc, geo_file):
                            success = True
                            if self.debug:
                                print(f"--- Successfully processed {geo_file.name} ---")
                            break
                    except Exception as e:
                        # Catch unexpected errors during an attempt
                        if self.debug:
                            print(f"An unexpected error occurred on attempt {attempt}: {e}")
                    finally:
                        # Cleanup: ensure the process is closed after each attempt
                        if proc and proc.isalive():
                            proc.close(force=True)

                    if not success and self.debug:
                        print(f"Attempt {attempt} failed. Retrying...")

                # After all retries, if still not successful, raise an error
                if not success:
                    # This exception will be caught and logged in the avl runner.
                    raise RuntimeError(f"Failed to process {geo_file.name} after {retry} attempts.")

        except Exception as e:
            if self.debug:
                print(f"An error ocurred: {e}")


# =============================================================================================== #
# === Self-test: python src/simulation_support/avl_simulation.py [--avl PATH] [--keep] === #
# =============================================================================================== #

GEOMETRY_TEMPLATE = """self test wing
0.0
0 0 0
1.0 1.0 4.0
0 0 0
SURFACE
wing
8 1.0 8 1.0
YDUPLICATE
0.0
SECTION
0 0 0 1.0 0
AFIL
{airfoil}
SECTION
0 2 0 1.0 0
AFIL
{airfoil}
"""


def self_test(avl_exe, case_file, keep=False):
    """
    Runs AVL on a plain rectangular wing inside folders whose full paths are far longer than AVL's 80-character
    force file limit, checks that the force file is written in full, and that a file name that cannot fit is
    refused before AVL starts. Returns True if every check passed.
    """
    import shutil
    import tempfile
    from polar_creation import _write_naca0012

    root = Path(tempfile.gettempdir()) / "avl_self_test"
    work = root / "a_deliberately_long_folder_name_well_past_the_avl_force_file_limit" / "and_one_more_level"
    shutil.rmtree(root, ignore_errors=True)
    geo_dir, force_dir, support_dir = work / "geometry", work / "forces", work / "support"
    for d in (geo_dir, force_dir, support_dir):
        d.mkdir(parents=True)
    airfoil = work / "naca0012_self_test.dat"
    _write_naca0012(airfoil)
    geo = geo_dir / "wing_self_test_geo.avl"
    geo.write_text(GEOMETRY_TEMPLATE.format(airfoil=airfoil))
    case = support_dir / Path(case_file).name
    shutil.copy(case_file, case)
    force = force_dir / "force_wing_self_test.txt"

    print(f"AVL runner self-test ({'Windows' if IS_WINDOWS else 'Linux'})")
    print(f"  AVL: {avl_exe}")
    print(f"  Force file path length: {len(str(force))} characters (AVL limit {AVL_MAX_NAME})\n")

    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    AVL_Run_Case([str(geo)], str(case), avl_exe, str(force_dir)).run()
    text = force.read_text() if force.exists() else ""
    check("force file written", "CLtot" in text and "CDtot" in text,
          "CLtot and CDtot present" if text else "no force file was written")
    stray = [p.name for p in force_dir.iterdir() if p.name != force.name]
    check("no truncated or leftover files", not stray, "none" if not stray else ", ".join(stray))

    long_geo = geo_dir / ("wing_" + "x" * 80 + "_geo.avl")
    long_geo.write_text(geo.read_text())
    try:
        AVL_Run_Case([str(long_geo)], str(case), avl_exe, str(force_dir))
        check("over-long file name refused", False, "no error was raised")
    except ValueError:
        check("over-long file name refused", True, "stopped with a clear error before AVL started")

    failed = results.count(False)
    print(f"\n{'ALL ' + str(len(results)) + ' CHECKS PASSED' if not failed else f'{failed} OF {len(results)} CHECKS FAILED'}")
    if keep:
        print(f"Test files kept in: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return not failed


if __name__ == "__main__":
    import argparse
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Self-test of the AVL runner with long folder paths.")
    parser.add_argument("--avl", default=str(project_root / "bin" / ("avl.exe" if IS_WINDOWS else "avl")),
                        help="AVL executable (default: bin/avl of the project).")
    parser.add_argument("--case", default=str(project_root / "data" / "support" / "Hossain_sqwing_case.run"),
                        help="AVL case file (default: data/support/Hossain_sqwing_case.run).")
    parser.add_argument("--keep", action="store_true", help="Keep the test folder for inspection.")
    args = parser.parse_args()
    sys.exit(0 if self_test(args.avl, args.case, args.keep) else 1)
