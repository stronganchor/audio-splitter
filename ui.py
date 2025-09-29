# ui.py
# Tkinter UI for interactive splitting: visualize waveform, auto-detect splits,
# shade Speech/Silence, tweak thresholds live, drag boundaries, play/export,
# and (optionally) transcribe+rename.

import os
import math
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from processor import AudioProcessor, ffplay_available

# ---- UI defaults ----
DEFAULT_USE_MORE_NOISE_REDUCTION = True
DEFAULT_TARGET_LUFS = -18.0
DEFAULT_MIN_SILENCE = 0.40
DEFAULT_THRESHOLD_DB = -35
DEFAULT_HYST_DB = 8
DEFAULT_MIN_SEGMENT = 0.25
DEFAULT_MIN_GAP = 0.20
DEFAULT_VIEW_WIDTH = 20.0

BOUNDARY_SELECT_TOLERANCE_PX = 6
ALT_SEGMENT_SHADE_ALPHA = 0.10
SELECTED_SEGMENT_ALPHA = 0.25

HEAD_BUFFER = 0.02  # seconds to keep before start on export
TAIL_BUFFER = 0.15  # seconds to keep after end on export


def time_to_str(t: float) -> str:
    m = int(max(0.0, t) // 60)
    s = max(0.0, t) - 60 * m
    return f"{m:02d}:{s:06.3f}"


class SplitterUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Interactive Audio Splitter")
        self.geometry("1180x720")

        # Engine
        self.processor = AudioProcessor()

        # State
        self.boundaries: List[float] = []               # [t0, t1, ...]
        self.drag_idx: Optional[int] = None             # boundary being dragged
        self.selected_seg_idx: Optional[int] = None     # selected interval index
        self.envelope_x: Optional[np.ndarray] = None
        self.envelope_y: Optional[np.ndarray] = None
        self.play_proc: Optional[subprocess.Popen] = None
        self.current_segments: List[Tuple[float, float]] = []
        self.first_interval_is_segment: bool = True     # parity helper

        # Controls state
        self.var_view_width = tk.DoubleVar(value=DEFAULT_VIEW_WIDTH)
        self.var_view_start = tk.DoubleVar(value=0.0)

        self.var_more_noise = tk.BooleanVar(value=DEFAULT_USE_MORE_NOISE_REDUCTION)
        self.var_lufs = tk.DoubleVar(value=DEFAULT_TARGET_LUFS)
        self.var_detector = tk.StringVar(value="Energy")            # Energy | FFmpeg
        self.var_min_sil = tk.DoubleVar(value=DEFAULT_MIN_SILENCE)
        self.var_thresh = tk.IntVar(value=DEFAULT_THRESHOLD_DB)
        self.var_hyst = tk.IntVar(value=DEFAULT_HYST_DB)
        self.var_min_gap = tk.DoubleVar(value=DEFAULT_MIN_GAP)
        self.var_min_seg = tk.DoubleVar(value=DEFAULT_MIN_SEGMENT)
        self.var_format = tk.StringVar(value="mp3")
        self.var_bitrate = tk.StringVar(value="192k")
        self.var_lang = tk.StringVar(value="en")
        self.var_segments_mode = tk.StringVar(value="Speech")       # Speech | Silence

        # Build UI
        self._build_controls()
        self._build_plot()

        # Events
        self.bind("<<ViewChanged>>", lambda e: self._apply_view_limits())
        self.canvas.mpl_connect("scroll_event", self._on_scroll_event)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.canvas.mpl_connect("button_release_event", self.on_plot_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_plot_motion)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._set_status("Load an audio/video file to begin.")

    # ------------------- Layout -------------------

    def _build_controls(self) -> None:
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(frm, text="Load File…", command=self.on_load_file).grid(row=0, column=0, padx=4)

        ttk.Label(frm, text="Noise:").grid(row=0, column=1, sticky="e")
        ttk.Checkbutton(frm, text="More (−30 dB)", variable=self.var_more_noise).grid(row=0, column=2, padx=2)

        ttk.Label(frm, text="Target LUFS:").grid(row=0, column=3, sticky="e")
        ttk.Entry(frm, width=6, textvariable=self.var_lufs).grid(row=0, column=4, padx=2)

        ttk.Label(frm, text="Segments:").grid(row=0, column=5, sticky="e")
        cmb_mode = ttk.Combobox(
            frm, width=8, textvariable=self.var_segments_mode,
            values=["Speech", "Silence"], state="readonly"
        )
        cmb_mode.grid(row=0, column=6, padx=2)
        cmb_mode.bind("<<ComboboxSelected>>", lambda e: self.on_detection_param_change())

        ttk.Button(frm, text="Process/Visualize", command=self.on_process).grid(row=0, column=7, padx=6)

        # Detector row
        ttk.Label(frm, text="Detector:").grid(row=1, column=0, sticky="e")
        cmb_det = ttk.Combobox(frm, width=10, textvariable=self.var_detector,
                               values=["Energy", "FFmpeg"], state="readonly")
        cmb_det.grid(row=1, column=1, padx=2)
        cmb_det.bind("<<ComboboxSelected>>", lambda e: self.on_detection_param_change())

        ttk.Label(frm, text="Min Silence (s):").grid(row=1, column=2, sticky="e")
        ttk.Scale(frm, from_=0.05, to=1.50, variable=self.var_min_sil,
                  command=self.on_detection_param_change).grid(row=1, column=3, sticky="we", padx=4)

        ttk.Label(frm, text="Threshold (dBFS):").grid(row=1, column=4, sticky="e")
        ttk.Scale(frm, from_=-80, to=-10, variable=self.var_thresh,
                  command=self.on_detection_param_change).grid(row=1, column=5, sticky="we", padx=4)

        ttk.Label(frm, text="Hysteresis (dB):").grid(row=1, column=6, sticky="e")
        ttk.Scale(frm, from_=2, to=18, variable=self.var_hyst,
                  command=self.on_detection_param_change).grid(row=1, column=7, sticky="we", padx=4)

        ttk.Label(frm, text="Min Gap (s):").grid(row=1, column=8, sticky="e")
        ttk.Scale(frm, from_=0.05, to=1.00, variable=self.var_min_gap,
                  command=self.on_detection_param_change).grid(row=1, column=9, sticky="we", padx=4)

        ttk.Label(frm, text="Min Segment (s):").grid(row=1, column=10, sticky="e")
        ttk.Scale(frm, from_=0.10, to=2.00, variable=self.var_min_seg,
                  command=lambda e: self._refresh_plot(True)).grid(row=1, column=11, sticky="we", padx=4)

        ttk.Button(frm, text="Auto-Detect Splits", command=lambda: self.auto_detect(True)).grid(row=1, column=12, padx=4)

        # Export + playback + transcription
        ttk.Label(frm, text="Export as:").grid(row=2, column=0, sticky="e")
        ttk.Combobox(frm, width=6, textvariable=self.var_format,
                     values=["mp3", "wav"], state="readonly").grid(row=2, column=1, padx=2)

        ttk.Label(frm, text="Bitrate (mp3):").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm, width=7, textvariable=self.var_bitrate).grid(row=2, column=3, padx=2)

        ttk.Button(frm, text="Export Segments…", command=self.export_segments).grid(row=2, column=4, padx=4)
        ttk.Button(frm, text="Play Selected", command=self.play_selected_segment).grid(row=2, column=5, padx=4)
        ttk.Button(frm, text="Stop", command=self.stop_playback).grid(row=2, column=6, padx=4)

        ttk.Label(frm, text="Lang:").grid(row=2, column=7, sticky="e")
        ttk.Entry(frm, width=5, textvariable=self.var_lang).grid(row=2, column=8, padx=2)
        ttk.Button(frm, text="Transcribe & Rename", command=self.transcribe_and_rename).grid(row=2, column=9, padx=4)

        self.status = ttk.Label(frm, text="", anchor="w")
        self.status.grid(row=3, column=0, columnspan=13, sticky="we", pady=(6, 0))

        ttk.Label(
            self, anchor="w",
            text=("Tips: Click inside segment to select • Double-click to play • "
                  "SHIFT+Click add boundary • CTRL+Click a line to delete • "
                  "Drag a red line to adjust • Mouse wheel = horizontal scroll • "
                  "Zoom via View controls")
        ).pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        # View controls
        view = ttk.Frame(self)
        view.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(view, text="View (s):").grid(row=0, column=0, sticky="e")
        self.sld_view_w = ttk.Scale(
            view, from_=5, to=90, variable=self.var_view_width,
            command=lambda e: self._on_view_changed()
        )
        self.sld_view_w.grid(row=0, column=1, sticky="we", padx=6)

        ttk.Label(view, text="Start (s):").grid(row=0, column=2, sticky="e")
        self.sld_view_start = ttk.Scale(
            view, from_=0, to=1, variable=self.var_view_start,
            command=lambda e: self._on_view_changed()
        )
        self.sld_view_start.grid(row=0, column=3, sticky="we", padx=6)
        view.columnconfigure(1, weight=1)
        view.columnconfigure(3, weight=1)

    def _build_plot(self) -> None:
        self.fig = Figure(figsize=(9.6, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Waveform (processed)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _set_status(self, msg: str) -> None:
        self.status.config(text=msg)
        self.update_idletasks()

    # ------------------- Actions -------------------

    def on_load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio/video file",
            filetypes=[
                ("Media", "*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg;*.opus;*.mp4;*.mkv;*.mov;*.avi"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        self.processor.input_path = path
        self._set_status(f"Selected: {path}")

    def on_process(self) -> None:
        if not self.processor.input_path:
            messagebox.showinfo("No file", "Please choose a file first.")
            return

        self._set_status("Processing with noise reduction + loudness normalization…")
        self.processor.load_and_process(
            self.processor.input_path,
            use_more_noise=self.var_more_noise.get(),
            target_lufs=float(self.var_lufs.get()),
        )

        # Build a light-weight envelope for plotting (decimated)
        x = self.processor.samples
        sr = self.processor.sample_rate
        if x is not None and sr is not None:
            max_points = 8000
            n = len(x)
            if n <= max_points:
                self.envelope_x = np.linspace(0, self.processor.duration, n)
                self.envelope_y = x
            else:
                bin_size = int(math.ceil(n / max_points))
                trimmed = x[: (n // bin_size) * bin_size]
                y = trimmed.reshape(-1, bin_size)
                y = np.max(np.abs(y), axis=1) * np.sign(np.sum(y, axis=1))
                self.envelope_x = np.linspace(0, self.processor.duration, len(y))
                self.envelope_y = y

        self.boundaries = [0.0, self.processor.duration]
        self.selected_seg_idx = None
        self.current_segments = []
        self.first_interval_is_segment = True

        self.var_view_start.set(0.0)
        self._update_view_slider_range()
        self._refresh_plot(False)
        self._apply_view_limits()

        self._set_status("Processed. Use Auto-Detect or add boundaries manually.")

    def on_detection_param_change(self, _evt=None) -> None:
        if self.processor.processed_wav:
            self.auto_detect(True)

    # ------------------- Auto detect & Plot -------------------

    def auto_detect(self, preserve_view: bool = True) -> None:
        if not self.processor.processed_wav:
            return

        cur_xlim = self.ax.get_xlim()

        thr = int(self.var_thresh.get())
        min_sil = float(self.var_min_sil.get())
        hyst = int(self.var_hyst.get())
        min_gap = float(self.var_min_gap.get())
        min_seg = float(self.var_min_seg.get())
        mode = self.var_detector.get()
        seg_mode = self.var_segments_mode.get()  # "Speech" | "Silence"

        # 1) Detect silences
        if mode == "FFmpeg":
            silences = self.processor.detect_silences_ffmpeg(threshold_db=thr, min_silence=min_sil)
        else:
            silences = self.processor.detect_silences_energy(
                threshold_db=thr, min_silence=min_sil, hysteresis_db=hyst
            )

        # 2) Convert to segments based on chosen mode
        if seg_mode == "Speech":
            segs = self.processor.invert_to_speech_segments(silences, min_seg=min_seg, min_gap=min_gap)
        else:
            segs = [(s, e) for s, e in silences if (e - s) >= min_gap] or [(0.0, self.processor.duration)]

        # 3) Snap boundaries to local minima
        segs = self.processor.snap_times_to_minima(segs)

        # 4) Convert segments -> boundary list and figure out parity
        self.current_segments = segs[:]
        bounds = [segs[0][0]]
        for a, b in segs:
            if not bounds or abs(a - bounds[-1]) > 1e-3:
                bounds.append(a)
            bounds.append(b)
        bounds = [max(0.0, min(self.processor.duration, t)) for t in bounds]

        first_is_segment = True
        if bounds and bounds[0] > 0.0:
            first_is_segment = False
            bounds = [0.0] + bounds
        if not bounds or bounds[-1] < self.processor.duration:
            bounds.append(self.processor.duration)

        self.boundaries = bounds
        self.first_interval_is_segment = first_is_segment
        self.selected_seg_idx = None

        self._refresh_plot(True)
        if preserve_view:
            self.ax.set_xlim(*cur_xlim)
            self.canvas.draw_idle()

        self._set_status(f"Found {len(self.current_segments)} segment(s).")

    def _interval_is_segment(self, i: int) -> bool:
        return (i % 2 == 0) if self.first_interval_is_segment else (i % 2 == 1)

    def _refresh_plot(self, preserve_xlim: bool = True) -> None:
        cur_xlim = self.ax.get_xlim() if preserve_xlim else None

        self.ax.clear()
        self.ax.set_title("Waveform (processed)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.2)

        if self.envelope_x is not None and self.envelope_y is not None:
            self.ax.plot(self.envelope_x, self.envelope_y, linewidth=0.7)

        if self.boundaries and len(self.boundaries) >= 2:
            # Shade only the chosen segments; gaps remain white
            for i in range(len(self.boundaries) - 1):
                a = self.boundaries[i]
                b = self.boundaries[i + 1]
                if self._interval_is_segment(i):
                    alpha = SELECTED_SEGMENT_ALPHA if (self.selected_seg_idx == i) else ALT_SEGMENT_SHADE_ALPHA
                    self.ax.axvspan(a, b, alpha=alpha)
            # Draw boundaries
            for b in self.boundaries:
                self.ax.axvline(b, color="r", linewidth=1.2, alpha=0.85)

        self.ax.set_ylim(-1.05, 1.05)
        if cur_xlim is not None:
            self.ax.set_xlim(*cur_xlim)
        elif self.processor.duration:
            self.ax.set_xlim(0, self.processor.duration)

        self.canvas.draw_idle()

    # ------------------- Mouse / Selection -------------------

    def _nearest_boundary_idx(self, xdata: float, px_tol: int = BOUNDARY_SELECT_TOLERANCE_PX) -> Optional[int]:
        if not self.boundaries:
            return None
        trans = self.ax.transData.transform
        px_per_sec = abs(trans((xdata + 1.0, 0.0))[0] - trans((xdata, 0.0))[0])
        tol_sec = px_tol / px_per_sec if px_per_sec > 0 else 0.02
        diffs = [abs(b - xdata) for b in self.boundaries]
        idx = int(np.argmin(diffs))
        return idx if diffs[idx] <= tol_sec else None

    def _segment_index_at_time(self, t: float) -> Optional[int]:
        if not self.boundaries or len(self.boundaries) < 2:
            return None
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= t <= self.boundaries[i + 1]:
                return i if self._interval_is_segment(i) else None
        return None

    def on_plot_click(self, event) -> None:
        if event.inaxes != self.ax or not self.processor.processed_wav:
            return

        x = float(event.xdata)

        # Add boundary
        if event.key == "shift":
            self.boundaries.append(max(0.0, min(self.processor.duration, x)))
            self.boundaries = sorted(set(self.boundaries))
            self.selected_seg_idx = self._segment_index_at_time(x)
            self._refresh_plot(True)
            self._set_status(f"Added boundary at {time_to_str(x)}")
            return

        # Delete boundary
        if event.key == "control":
            idx = self._nearest_boundary_idx(x)
            if idx is not None and 0 < idx < len(self.boundaries) - 1:
                val = self.boundaries.pop(idx)
                self.selected_seg_idx = self._segment_index_at_time(x)
                self._refresh_plot(True)
                self._set_status(f"Deleted boundary at {time_to_str(val)}")
            return

        # Drag boundary
        idx = self._nearest_boundary_idx(x)
        if idx is not None and 0 < idx < len(self.boundaries) - 1:
            self.drag_idx = idx
            self._set_status(f"Dragging boundary #{idx} @ {time_to_str(self.boundaries[idx])}")
            return

        # Select segment
        seg = self._segment_index_at_time(x)
        if seg is not None:
            self.selected_seg_idx = seg
            self._refresh_plot(True)
            a, b = self.boundaries[seg], self.boundaries[seg + 1]
            self._set_status(f"Selected segment #{seg + 1}: {time_to_str(a)} – {time_to_str(b)}")

    def on_plot_motion(self, event) -> None:
        if self.drag_idx is None or event.inaxes != self.ax:
            return
        x = float(event.xdata)
        lo = self.boundaries[self.drag_idx - 1] + 0.01
        hi = self.boundaries[self.drag_idx + 1] - 0.01
        x = max(lo, min(hi, x))
        self.boundaries[self.drag_idx] = x
        self.selected_seg_idx = self._segment_index_at_time(x)
        self._refresh_plot(True)

    def on_plot_release(self, event) -> None:
        if self.drag_idx is not None:
            self._set_status(f"Boundary #{self.drag_idx} set to {time_to_str(self.boundaries[self.drag_idx])}")
        self.drag_idx = None

    # ------------------- Playback & Export -------------------

    def play_selected_segment(self) -> None:
        if self.selected_seg_idx is None or not self.processor.processed_wav:
            messagebox.showinfo("No selection", "Click a segment to select it.")
            return
        if not ffplay_available():
            messagebox.showinfo("Missing ffplay", "Install ffplay (part of ffmpeg) to enable playback.")
            return

        a = self.boundaries[self.selected_seg_idx]
        b = self.boundaries[self.selected_seg_idx + 1]
        dur = max(0.05, b - a)

        self.stop_playback()
        self.play_proc = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "error",
             "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", self.processor.processed_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._set_status(f"Playing segment ({time_to_str(a)} – {time_to_str(b)})")

    def stop_playback(self) -> None:
        if self.play_proc and self.play_proc.poll() is None:
            try:
                self.play_proc.terminate()
            except Exception:
                pass
        self.play_proc = None

    def export_segments(self) -> None:
        if not self.processor.processed_wav:
            messagebox.showinfo("No audio", "Process a file first.")
            return
        if len(self.boundaries) < 2:
            messagebox.showinfo("No segments", "No boundaries defined.")
            return
    
        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir:
            return
    
        fmt = self.var_format.get().lower()
        bitrate = self.var_bitrate.get()
        src = self.processor.processed_wav
    
        count = 0
        for i in range(len(self.boundaries) - 1):
            if not self._interval_is_segment(i):
                continue  # export only true segments
    
            a = max(0.0, self.boundaries[i] - HEAD_BUFFER)
            b = min(self.processor.duration, self.boundaries[i + 1] + TAIL_BUFFER)
            if b <= a + 0.01:
                continue
    
            # duration for accurate trimming and fade placement
            dur = max(0.05, b - a)
    
            # very short micro-fades to suppress clicks at cut points
            # fade-in at 0s, fade-out ending at dur
            fade_d = 0.006  # 6 ms
            fade_out_start = max(0.0, dur - fade_d)
            fade_filter = f"afade=t=in:st=0:d={fade_d:.3f},afade=t=out:st={fade_out_start:.3f}:d={fade_d:.3f}"
    
            ext = "wav" if fmt == "wav" else "mp3"
            out_path = os.path.join(out_dir, f"segment_{i + 1:03d}.{ext}")
    
            # Use input-then-seek for sample-accurate cuts; add -accurate_seek
            cmd = ["ffmpeg", "-y",
                   "-accurate_seek",
                   "-i", src,
                   "-ss", f"{a:.3f}",
                   "-to", f"{b:.3f}",
                   "-af", fade_filter,
                   "-avoid_negative_ts", "make_zero"]
    
            if fmt == "wav":
                cmd += ["-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le", out_path]
            else:
                cmd += ["-ar", "48000", "-ac", "1", "-b:a", bitrate, "-c:a", "libmp3lame", out_path]
    
            run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if run.returncode == 0:
                count += 1
    
        self._set_status(f"Exported {count} segment(s) to: {out_dir}")
        messagebox.showinfo("Export complete", f"Exported {count} segment(s) to:\n{out_dir}")
    
    # ------------------- Transcription & Rename -------------------

    def transcribe_and_rename(self) -> None:
        # Soft-fail if not configured
        if os.getenv("ASSEMBLYAI_API_KEY") is None:
            messagebox.showinfo(
                "Transcription not configured",
                "Set ASSEMBLYAI_API_KEY in your environment to enable automatic renaming.")
            return
        if not self.processor.processed_wav or len(self.boundaries) < 2:
            messagebox.showinfo("No segments", "Process and split the audio first.")
            return

        lang = self.var_lang.get().strip() or "en"
        tr = self.processor.transcribe(language_code=lang)
        if tr is None or not tr.words:
            messagebox.showinfo("Transcription", "No words returned. Keeping default names.")
            return

        seg_dir = filedialog.askdirectory(title="Choose the folder with exported segments")
        if not seg_dir:
            return

        words = self.processor.transcript.words  # type: ignore

        def sanitize(name: str) -> str:
            import re
            name = re.sub(r'[<>:"/\\|?*\n\r\t]+', " ", name).strip()
            name = re.sub(r"\s+", " ", name)
            return name or "segment"

        for i in range(len(self.boundaries) - 1):
            a = self.boundaries[i]
            b = self.boundaries[i + 1]
            seg_words = [w.text for w in words if (w.end >= a - 0.10 and w.start <= b + 0.10)]
            base = sanitize(" ".join(seg_words[:8])) if seg_words else f"segment_{i + 1:03d}"

            cand_mp3 = os.path.join(seg_dir, f"segment_{i + 1:03d}.mp3")
            cand_wav = os.path.join(seg_dir, f"segment_{i + 1:03d}.wav")
            old = cand_mp3 if os.path.exists(cand_mp3) else cand_wav if os.path.exists(cand_wav) else None
            if not old:
                continue

            ext = os.path.splitext(old)[1].lower()
            new = os.path.join(seg_dir, f"{base}{ext}")

            k = 1
            final = new
            while os.path.exists(final):
                final = os.path.join(seg_dir, f"{base}_{k}{ext}")
                k += 1

            try:
                os.replace(old, final)
            except Exception:
                pass

        messagebox.showinfo("Done", f"Renamed segments in:\n{seg_dir}")

    # ------------------- View / Zoom -------------------

    def _update_view_slider_range(self) -> None:
        dur = self.processor.duration or 0.0
        width = float(self.var_view_width.get())
        max_start = max(0.0, dur - width)
        self.sld_view_start.configure(from_=0.0, to=max_start)

    def _apply_view_limits(self) -> None:
        dur = self.processor.duration or 0.0
        width = float(self.var_view_width.get())
        start = float(self.var_view_start.get())
        start = max(0.0, min(start, max(0.0, dur - width)))
        end = start + width
        self.ax.set_xlim(start, min(end, max(dur, width)))
        self.canvas.draw_idle()

    def _on_view_changed(self) -> None:
        self._update_view_slider_range()
        self.event_generate("<<ViewChanged>>", when="tail")

    def _on_scroll_event(self, event) -> None:
        if not self.processor.duration:
            return
        width = float(self.var_view_width.get())
        delta = -np.sign(event.step) * (0.10 * width)  # 10% per wheel notch
        new_start = float(self.var_view_start.get()) + delta
        self.var_view_start.set(max(0.0, min(new_start, max(0.0, self.processor.duration - width))))
        self._apply_view_limits()

    # ------------------- Shutdown -------------------

    def on_close(self) -> None:
        try:
            self.stop_playback()
        finally:
            self.destroy()


# Small launcher convenience (so you can run `python ui.py` directly)
if __name__ == "__main__":
    app = SplitterUI()
    app.mainloop()
