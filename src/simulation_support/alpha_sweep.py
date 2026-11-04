"""
Alpha-sweep strategies for XFOIL polars, used by Xfoil_Polar_Creator.run_simulation on every platform.

WHY TWO SWEEPS
    'sequential' is the original sweep, used to generate every dataset up to dataset 17. It produces a
    non-physical low-drag branch (see the note below), so 'zero_out' replaced it as the default. 'sequential'
    remains only so legacy tools and datasets keep their original behavior.

NOTE ON 'sequential' (NOT RECOMMENDED, LEGACY ONLY)
    One XFOIL process walks the whole alpha sequence from the lowest angle upwards, so every angle starts
    from the boundary layer solution of the previous one. Starting in deep stall (-20 deg) leaves many
    airfoils on a spurious low-drag branch from about -8 deg to +8 deg: CDmin around 0.002, below the
    laminar flat-plate friction limit (2 x 1.328 / sqrt(Re), about 0.0053 at Re = 2.5e5). The neighbour-
    consistency diagnosis of dataset 17 (data/datasets/dataset_17/rectification, 2026-10-28) showed that
    every strongly flagged airfoil, including the dataset best (E ~ 57.9), reproduces E ~ 53-58 with this
    sweep and E ~ 20-24 with 'zero_out', in agreement with its grid neighbours. The result is deterministic,
    so repeating the sweep does not help. This mode is kept only to reproduce legacy datasets.

'zero_out' (default)
    Two XFOIL processes both start at 0 deg: one walks up to the highest angle and the other down to the
    lowest, so neither branch inherits a deep-stall solution. If a branch stops converging before the end
    of its range, the missing tail is filled by a cold pass that starts at the extreme angle and walks
    inwards; deep-stall angles agree between both sweeps, and rows from the zero_out branches always take
    priority. Branch files are written to a sibling '<polar folder>_branches' directory (see
    branch_directory) and merged into the requested polar file, sorted by alpha.

'cold' (diagnostic only)
    Every angle is solved by its own fresh XFOIL process, so no angle inherits a boundary layer solution from
    another: the protocol-independent reference. About 80 process starts per polar, so it is too slow for
    dataset generation; src/utilities/sweep_protocol_check.py uses it to check that 'zero_out' polars do not
    depend on their sweep history (a laminar separation bubble can make XFOIL's solution bistable, and a
    warm start then decides which branch it returns).
"""
from pathlib import Path

SWEEP_MODES = ("zero_out", "sequential", "cold")
DEFAULT_SWEEP_MODE = "zero_out"
# Revision of each mode's output. Bump it when a mode's polars change, so stored rectification runs of an
# older revision are not reused (zero_out 2 = tail fill added; zero_out 3 = XFOIL file names kept short: on Linux
# revision 2 wrote its branch files under names longer than XFOIL reads, so it produced no polars at all).
SWEEP_REVISIONS = {"sequential": 1, "zero_out": 3, "cold": 1}
COMPLETION_FLAG = "XFOIL process completed successfully"


def run_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter, sweep_mode=None):
    """
    Runs the requested sweep with runner.alpha_pass and appends the completion flag on success.
    sweep_mode None selects DEFAULT_SWEEP_MODE. Returns True on success.
    """
    mode = DEFAULT_SWEEP_MODE if sweep_mode is None else sweep_mode
    if mode == "zero_out":
        return run_zero_out_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter)
    if mode == "sequential":
        return run_sequential_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter)
    if mode == "cold":
        return run_cold_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter)
    raise ValueError(f"Unknown sweep_mode '{sweep_mode}'. Options: {SWEEP_MODES}")


# --- Sequential (legacy) --- #

def run_sequential_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter):
    """One pass over alfa_sequence in the given order. See the module note before using it."""
    if not runner.alpha_pass(airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter):
        return False
    runner.completion_flag(str(polar_file_path))
    return True


# --- Zero-out --- #

def zero_out_branches(alfa_sequence):
    """Splits an alpha sequence into an ascending branch and a descending branch, both starting at 0 deg."""
    angles = sorted(set(float(a) for a in alfa_sequence))
    positive = [a for a in angles if a >= 0]
    negative = sorted((a for a in angles if a < 0), reverse=True)
    if negative:
        negative = [0.0] + negative
    return [branch for branch in (positive, negative) if branch]


def tail_branches(alfa_sequence, converged):
    """
    Cold passes for the angles beyond the furthest converged angle of each zero_out branch: from the lowest
    angle upwards and from the highest angle downwards. Gaps inside a converged range are not refilled.
    """
    angles = sorted(set(float(a) for a in alfa_sequence))
    converged = set(converged)
    low_edge = min((a for a in converged if a <= 0), default=0.0)
    high_edge = max((a for a in converged if a >= 0), default=0.0)
    lower_tail = [a for a in angles if a < low_edge]
    upper_tail = sorted((a for a in angles if a > high_edge), reverse=True)
    return [tail for tail in (lower_tail, upper_tail) if tail]


def branch_directory(polar_file_path):
    """Folder that holds the temporary branch polars of polar_file_path (outside the polar folder itself)."""
    polar_dir = Path(polar_file_path).parent
    return polar_dir.with_name(f"{polar_dir.name}_branches")


def _split_polar(path):
    """Returns (header lines up to the dashed separator, {alpha: data line}) of an XFOIL polar file."""
    lines = Path(path).read_text().splitlines()
    separator = next((i for i, line in enumerate(lines) if line.strip().startswith("------")), None)
    if separator is None:
        return lines, {}
    rows = {}
    for line in lines[separator + 1:]:
        if COMPLETION_FLAG in line or not line.strip():
            continue
        try:
            alpha = float(line.split()[0])
        except (ValueError, IndexError):
            continue
        rows.setdefault(alpha, line)
    return lines[:separator + 1], rows


def merge_polar_files(branch_paths, output_path):
    """
    Merges branch polar files into one polar sorted by alpha. The header comes from the first branch that
    has one; for an angle present in several branches, the earliest branch's row is kept.
    """
    header, merged = None, {}
    for path in branch_paths:
        if not Path(path).exists():
            continue
        branch_header, rows = _split_polar(path)
        if header is None and rows:
            header = branch_header
        for alpha, line in rows.items():
            merged.setdefault(alpha, line)
    if header is None:
        return False
    body = [merged[a] for a in sorted(merged)]
    Path(output_path).write_text("\n".join(header + body) + "\n")
    return True


def run_zero_out_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter):
    """
    Runs both zero_out branches and any tail passes through runner.alpha_pass, merges them into
    polar_file_path and appends the completion flag. Returns True if any pass produced data.
    """
    polar_file_path = Path(polar_file_path)
    work_dir = branch_directory(polar_file_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    branch_paths = []

    def run_pass(tag, sequence):
        path = work_dir / f"{polar_file_path.stem}_{tag}{polar_file_path.suffix}"
        if path.exists():
            path.unlink()
        runner.alpha_pass(airfoil_file_path, str(path), sequence, visc, mach, iter)
        branch_paths.append(path)
        return _split_polar(path)[1].keys() if path.exists() else []

    try:
        converged = set()
        for i, branch in enumerate(zero_out_branches(alfa_sequence)):
            converged |= set(run_pass(f"b{i}", branch))
        for i, tail in enumerate(tail_branches(alfa_sequence, converged)):
            run_pass(f"t{i}", tail)
        merged = merge_polar_files(branch_paths, polar_file_path)
    finally:
        for path in branch_paths:
            if path.exists():
                path.unlink()
    if merged:
        runner.completion_flag(str(polar_file_path))
    return merged


# --- Cold (diagnostic) --- #

def run_cold_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter):
    """
    One fresh XFOIL process per angle (no warm start anywhere), merged into polar_file_path with the
    completion flag. Returns True if any angle converged.
    """
    polar_file_path = Path(polar_file_path)
    work_dir = branch_directory(polar_file_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    angle_paths = []
    try:
        for i, alfa in enumerate(sorted(set(float(a) for a in alfa_sequence))):
            path = work_dir / f"{polar_file_path.stem}_c{i}{polar_file_path.suffix}"
            if path.exists():
                path.unlink()
            runner.alpha_pass(airfoil_file_path, str(path), [alfa], visc, mach, iter)
            angle_paths.append(path)
        merged = merge_polar_files(angle_paths, polar_file_path)
    finally:
        for path in angle_paths:
            if path.exists():
                path.unlink()
    if merged:
        runner.completion_flag(str(polar_file_path))
    return merged
