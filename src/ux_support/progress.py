"""
Runs a long job in a background thread while the main thread shows its progress (Progress_GUI window, or a
console bar when no display is available). Tkinter must own the main thread, so the work goes to the thread.

The job reports progress by putting one item per finished unit into `progress_queue` (a queue.Queue for
threads, a multiprocessing Manager().Queue() for process pools) and may write what each worker is doing into
`status_list`. The window closes when the job ends, even if it reported fewer units than `total` or raised;
an exception from the job is re-raised on the main thread once the window has closed.
"""
import os
import queue
import sys
import threading
from typing import Callable, Optional

from tqdm import tqdm


def display_available() -> bool:
    """
    True when a graphical display can be used. On Linux this requires DISPLAY or WAYLAND_DISPLAY;
    Windows and macOS sessions are assumed to have one.
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def run_with_progress(job: Callable[[], None], total: int, progress_queue, title: str, header: str,
                      status_list=None, n_workers: int = 1, details: bool = True, detailed_only: bool = False,
                      use_gui: Optional[bool] = None, console_desc: str = "Progress", unit: str = "file") -> None:
    """
    Runs `job()` in a background thread and shows its progress until it ends.

    total          : units the job is expected to report (one progress_queue item each).
    status_list    : per-worker status texts shown in the detail view (default: one 'Working...' slot).
    details        : False hides the "More details..." button (single-thread stages have nothing to detail).
    detailed_only  : the per-worker view is the whole window (used by the rectifier).
    use_gui        : None picks the window when a display is available, else the console bar.
    """
    if status_list is None:
        status_list = ["Working..."] * n_workers
    state = {"error": None}

    def target():
        try:
            job()
        except BaseException as e:  # re-raised below, on the main thread
            state["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    gui = None
    if total > 0 and (display_available() if use_gui is None else use_gui):
        try:
            from .GUI import Progress_GUI
            gui = Progress_GUI(total_files=total, progress_queue=progress_queue, worker_status_list=status_list,
                               num_workers=n_workers, window_title=title, header_text=header,
                               detailed_only=detailed_only, show_details=details,
                               is_done=lambda: not thread.is_alive())
        except Exception as e:
            print(f"[INFO] Progress window unavailable ({e}); using the console progress bar.")
            gui = None
    thread.start()
    if gui is not None:
        gui.run()
    else:
        with tqdm(total=total, desc=console_desc, unit=unit) as bar:
            while True:
                try:
                    progress_queue.get(timeout=0.5)
                    if bar.n < total:
                        bar.update(1)
                except queue.Empty:
                    if not thread.is_alive():
                        break
    thread.join()
    if state["error"] is not None:
        raise state["error"]
