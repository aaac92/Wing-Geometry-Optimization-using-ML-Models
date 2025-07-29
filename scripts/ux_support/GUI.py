import tkinter as tk
from tkinter import ttk
import time
import queue as queue_mod

class Progress_GUI:
    """A Tkinter GUI to show multiprocessing progress."""

    def __init__(self, total_files, progress_queue, worker_status_list, num_workers, window_title, header_text):
        self.total_files = total_files
        self.progress_queue = progress_queue
        self.worker_status_list = worker_status_list
        self.num_workers = num_workers
        self.window_title = window_title
        self.header_text = header_text

        self.files_completed = 0
        self.start_time = time.time()
        self.is_finished = False

        self.root = tk.Tk()
        self.root.title(self.window_title)

        # --- Main (Simple) View Widgets ---
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

        self.detail_window = None

    def format_eta(self, seconds, end=False):
        """Formats seconds into a human-readable string: H-M-S."""
        if seconds < 0: return ""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        eta_str = f"{hours}h {minutes}m {seconds}s"
        return eta_str + " remaining" if not end else "Total runtime: " + eta_str

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
        eta_seconds = (elapsed / self.files_completed) * (self.total_files - self.files_completed) if self.files_completed > 0 else 0
        self.eta_label.config(text=self.format_eta(eta_seconds))
        
        self.root.after(500, self.update_simple_view)


    def show_detailed_view(self):
        """Hides the main window and shows the detailed thread view."""
        self.root.withdraw()
        self.create_detailed_window()
        self.update_detailed_view()

    def create_detailed_window(self):
        """Builds the detailed view Toplevel window."""
        self.detail_window = tk.Toplevel(self.root)
        self.detail_window.title(self.window_title + " - Details")

        def switch_back():
            self.detail_window.destroy()
            self.detail_window = None
            if not self.is_finished:
                self.root.deiconify()

        self.detail_window.protocol("WM_DELETE_WINDOW", switch_back)

        top_frame = tk.Frame(self.detail_window, padx=10, pady=5)
        top_frame.pack(fill="x")
        self.detail_total_label = tk.Label(top_frame, text=f"Overall Progress: 0/0 (0.0%)")
        self.detail_total_label.pack()
        self.detail_total_bar = ttk.Progressbar(top_frame, length=300, mode="determinate")
        self.detail_total_bar.pack()
        self.detail_eta_label = tk.Label(top_frame, text="")
        self.detail_eta_label.pack()

        tk.Button(top_frame, text="Less details", command=switch_back).pack(pady=5)
        
        canvas = tk.Canvas(self.detail_window)
        scrollbar = ttk.Scrollbar(self.detail_window, orient="vertical", command=canvas.yview)
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

    def update_detailed_view(self):
        """Updates the labels in the detailed view window."""
        if self.is_finished or not self.detail_window or not self.detail_window.winfo_exists():
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
        eta_seconds = (elapsed / self.files_completed) * (self.total_files - self.files_completed) if self.files_completed > 0 else 0
        self.detail_eta_label.config(text=self.format_eta(eta_seconds))
        
        self.detail_window.after(500, self.update_detailed_view)

    def run(self):
        """Starts the GUI update loop and the main event loop."""
        self.update_simple_view()
        self.root.mainloop()