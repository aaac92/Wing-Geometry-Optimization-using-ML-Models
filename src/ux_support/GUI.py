import tkinter as tk
from tkinter import ttk
import time
import queue as queue_mod
from typing import Callable, Optional

class Progress_GUI:
    """A Tkinter GUI to show multiprocessing progress."""

    def __init__(
        self, total_files, progress_queue, worker_status_list, num_workers,
        window_title, header_text, initial_completed=0,
        detailed_only: bool = False,
        show_acceleration: bool = False,
        acceleration_state: Optional[dict] = None,
    ):
        """
        detailed_only, show_acceleration, and acceleration_state are all
        opt-in (default off/None) so existing callers are unaffected.

        detailed_only : bool
            False (default): original simple-view-first behavior --
            progress bar/ETA with a "More details..." button that opens
            the per-worker view in a separate Toplevel with a "Less
            details" way back.
            True: the per-worker detail view IS the main window from the
            start -- no simple view, no button, no way back (there's
            nothing to go back to).
        show_acceleration : bool
            If True, adds a label to the detail view (in either mode)
            showing live acceleration state from `acceleration_state`,
            positioned between the ETA and the per-worker status boxes.
        acceleration_state : dict, optional
            Shared mutable {'dropped': int, 'value': float}, updated by
            the caller's worker thread and polled here every 500ms --
            same cross-thread pattern already used for
            `worker_status_list`, no locking. Required (and read fresh
            each poll) when `show_acceleration` is True; a missing key
            or None state falls back to a 0/0.0 display rather than
            raising.
        """
        self.total_files = total_files
        self.progress_queue = progress_queue
        self.worker_status_list = worker_status_list
        self.num_workers = num_workers
        self.window_title = window_title
        self.header_text = header_text

        self.detailed_only = detailed_only
        self.show_acceleration = show_acceleration
        self.acceleration_state = acceleration_state

        # initial_completed lets a resumed run start the bar/ETA at the
        # right offset instead of counting back up from 0. The ETA rate
        # below is still computed only from progress made THIS session
        # (files_completed - initial_completed over elapsed time since
        # this window opened), since start_time has no knowledge of how
        # long the earlier, already-completed portion actually took.
        self.initial_completed = initial_completed
        self.files_completed = initial_completed
        self.start_time = time.time()
        self.is_finished = False

        self.root = tk.Tk()
        self.root.title(self.window_title)

        self.detail_window = None
        self._detail_root = None
        self.accel_label = None

        if self.detailed_only:
            # Per-worker detail view IS the main window -- no simple view,
            # no button, nothing to switch back to.
            self._detail_root = self.root
            self._build_detail_widgets(parent=self.root, on_back=None)
        else:
            # --- Main (Simple) View Widgets --- (unchanged legacy path)
            self.main_frame = tk.Frame(self.root, padx=10, pady=10)
            self.main_frame.pack(fill="both", expand=True)

            self.header_label = tk.Label(self.main_frame, text=self.header_text, justify=tk.LEFT)
            self.header_label.pack(pady=(0, 10))

            self.progress_label = tk.Label(self.main_frame, text="Starting...")
            self.progress_label.pack()

            self.progress_bar = ttk.Progressbar(self.main_frame, length=400, mode="determinate")
            self.progress_bar.pack(pady=5)

            self.eta_label = tk.Label(self.main_frame, text="")
            self.eta_label.pack()

            self.more_button = tk.Button(self.main_frame, text="More details...", command=self.show_detailed_view)
            self.more_button.pack(pady=10)

    def format_eta(self, seconds, end=False):
        """Formats seconds into a human-readable string: H-M-S."""
        if seconds < 0: return ""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        eta_str = f"{hours}h {minutes}m {seconds}s"
        return eta_str + " remaining" if not end else "Total runtime: " + eta_str

    def _format_acceleration(self) -> str:
        """Formats acceleration_state as 'Acceleration: -{dropped} pts/validated ({value})'."""
        state = self.acceleration_state or {}
        dropped = state.get('dropped', 0)
        value = state.get('value', 0.0)
        return f"Acceleration: -{dropped} pts/validated ({value:.2f})"

    def _build_detail_widgets(self, parent, on_back: Optional[Callable[[], None]]) -> None:
        """
        Builds the detail-view widgets (overall progress, optional
        acceleration readout, per-worker status boxes) into `parent`.
        Shared between the legacy Toplevel path (create_detailed_window)
        and the detailed_only default-window path -- `on_back` is the
        "Less details" callback in the legacy case, None (no button) in
        the detailed_only case.
        """
        top_frame = tk.Frame(parent, padx=10, pady=5)
        top_frame.pack(fill="x")

        if self.detailed_only and self.header_text:
            # Only shown here when there's no separate simple view to
            # show it -- the legacy Toplevel never displayed header_text.
            tk.Label(top_frame, text=self.header_text, justify=tk.LEFT).pack(pady=(0, 10))

        self.detail_total_label = tk.Label(top_frame, text="Overall Progress: 0/0 (0.0%)")
        self.detail_total_label.pack()
        self.detail_total_bar = ttk.Progressbar(top_frame, length=300, mode="determinate")
        self.detail_total_bar.pack()
        self.detail_eta_label = tk.Label(top_frame, text="")
        self.detail_eta_label.pack()

        if self.show_acceleration:
            self.accel_label = tk.Label(top_frame, text=self._format_acceleration())
            self.accel_label.pack()

        if on_back is not None:
            tk.Button(top_frame, text="Less details", command=on_back).pack(pady=5)

        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        self.worker_labels = []
        for i in range(self.num_workers):
            frame = tk.Frame(scrollable_frame, bd=2, relief="groove", padx=5, pady=5)
            frame.pack(fill="x", expand=True, pady=3)

            title = tk.Label(frame, text=f"Thread {i + 1}", font=("Helvetica", 10, "bold"))
            title.pack(anchor="w")

            status_label = tk.Label(frame, text="Initializing...", anchor="w", justify=tk.LEFT)
            status_label.pack(fill="x", expand=True)
            self.worker_labels.append(status_label)

    def check_for_completion(self):
        """Checks if all files have been processed and closes the GUI if so."""
        if self.files_completed >= self.total_files and not self.is_finished:
            self.is_finished = True
            print(f"Calculation complete. {self.format_eta(time.time() - self.start_time, end=True)}")
            self.root.destroy() # Automatically close the window
            return True
        return False

    def update_simple_view(self):
        """Updates the main progress bar and labels."""
        if self.is_finished:
            return

        try:
            while True:
                self.progress_queue.get_nowait()
                self.files_completed += 1
        except queue_mod.Empty:
            pass

        if self.check_for_completion():
            return

        percent = (self.files_completed / self.total_files) * 100 if self.total_files > 0 else 0
        self.progress_bar["value"] = percent
        self.progress_label.config(text=f"Files processed: {self.files_completed}/{self.total_files} ({percent:.1f}%)")

        elapsed = time.time() - self.start_time
        completed_this_session = self.files_completed - self.initial_completed
        eta_seconds = (
            (elapsed / completed_this_session) * (self.total_files - self.files_completed)
            if completed_this_session > 0 else 0
        )
        self.eta_label.config(text=self.format_eta(eta_seconds))
        
        self.root.after(500, self.update_simple_view)


    def show_detailed_view(self):
        """Hides the main window and shows the detailed thread view."""
        self.root.withdraw()
        self.create_detailed_window()
        self.update_detailed_view()

    def create_detailed_window(self):
        """Builds the detailed view Toplevel window (legacy simple+detail mode)."""
        self.detail_window = tk.Toplevel(self.root)
        self.detail_window.title(self.window_title + " - Details")
        self._detail_root = self.detail_window

        def switch_back():
            self.detail_window.destroy()
            self.detail_window = None
            self._detail_root = None
            if not self.is_finished:
                self.root.deiconify()

        self.detail_window.protocol("WM_DELETE_WINDOW", switch_back)
        self._build_detail_widgets(parent=self.detail_window, on_back=switch_back)

    def update_detailed_view(self):
        """
        Updates the labels in the detail view -- the Toplevel in legacy
        mode, or the main window itself when detailed_only=True.
        """
        if self.is_finished or self._detail_root is None or not self._detail_root.winfo_exists():
            return

        try:
            while True:
                self.progress_queue.get_nowait()
                self.files_completed += 1
        except queue_mod.Empty:
            pass

        if self.check_for_completion():
            return

        percent = (self.files_completed / self.total_files) * 100 if self.total_files > 0 else 0
        self.detail_total_bar["value"] = percent
        self.detail_total_label.config(text=f"Overall Progress: {self.files_completed}/{self.total_files} ({percent:.1f}%)")

        for i, label in enumerate(self.worker_labels):
            status_text = self.worker_status_list[i]
            label.config(text=f"Processing: {status_text}")
        
        elapsed = time.time() - self.start_time
        completed_this_session = self.files_completed - self.initial_completed
        eta_seconds = (
            (elapsed / completed_this_session) * (self.total_files - self.files_completed)
            if completed_this_session > 0 else 0
        )
        self.detail_eta_label.config(text=self.format_eta(eta_seconds))

        if self.show_acceleration and self.accel_label is not None:
            self.accel_label.config(text=self._format_acceleration())

        self._detail_root.after(500, self.update_detailed_view)

    def run(self):
        """Starts the GUI update loop and the main event loop."""
        if self.detailed_only:
            self.update_detailed_view()
        else:
            self.update_simple_view()
        self.root.mainloop()