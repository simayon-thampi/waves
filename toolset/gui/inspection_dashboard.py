import copy
import math
from typing import Optional, List, Dict, Tuple
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import time
import datetime
import os
import csv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from toolset.processing.channel_response import ChannelResponse, QUALITY_UNAVAILABLE, Role
from toolset.cs_utils.cs_subevent import SubeventResults
from toolset.cs_utils.cs_step import CSStepMode2, ToneQualityIndicator
from toolset.preprocess.config import PreprocessorConfig, UnwrapStrategy
from toolset.preprocess.pipeline import Preprocessor
from toolset.preprocess.temporal_smoother import TemporalSmoother
from toolset.preprocess.scene_classifier import SceneClassifier, SCENE_NAMES, SCENE_LOS, SCENE_NLOS, SCENE_MULTIPATH
from toolset.pipeline.replay_buffer import ReplayBuffer
from toolset.estimators.base import Estimator, EstimatorResult
from toolset.estimators.phase_slope import PhaseSlopeEstimator
from toolset.estimators.ifft import IFFTEstimator
from toolset.estimators.music import MUSICEstimator
from toolset.estimators.weighted_ls import WeightedLSEstimator
from toolset.constants import SPEED_OF_LIGHT, BLE_CS_STEP_1MHZ
import toolset.processing.cs_music

# Apply dark background style for high-end dark theme look
plt.style.use('dark_background')


class PipelineInspectorDashboard(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent.root)
        self.parent = parent
        self.title("RF CS Pipeline Stage Inspector & Debugger")
        self.geometry("1800" + "x980")
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

        # Thread safe queues & control
        self.replay_buffer = ReplayBuffer(200)
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        
        # State variables
        self.live_mode = tk.BooleanVar(value=True)
        self._last_raw_data: Optional[Tuple[ChannelResponse, Optional[SubeventResults]]] = None
        self._latest_res = None
        self._debounce_after_id = None
        
        # Coherent Smoother & Scene Classifier
        self.live_smoother = TemporalSmoother(window_size=8)
        self.scene_classifier = SceneClassifier()
        
        # Ground Truth & Accumulators
        self._ground_truth_m = 1.0
        self._cal_offset_a = 0.0   # learned bias correction for estimator A
        self._cal_offset_b = 0.0   # learned bias correction for estimator B
        self.session_log = []
        self.reset_stats_flag = False
        
        # Accumulators Lock & State in Thread
        self.accum_lock = threading.Lock()
        self.reset_accumulators()
        
        self._create_widgets()
        self.worker_thread.start()
        self._start_result_polling()

    def reset_accumulators(self):
        """Resets all incremental running statistics registers. Thread-safe."""
        with self.accum_lock:
            self.n_samples_a = 0
            self.sum_err_a = 0.0
            self.sum_sq_err_a = 0.0
            self.sum_est_a = 0.0
            self.sum_sq_est_a = 0.0
            
            self.n_samples_b = 0
            self.sum_err_b = 0.0
            self.sum_sq_err_b = 0.0
            self.sum_est_b = 0.0
            self.sum_sq_est_b = 0.0
            
            self.export_records = []

    def _create_widgets(self):
        # Configure layout grids
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=4)  # Pane 1
        self.grid_columnconfigure(1, weight=1)  # Pane 2
        self.grid_columnconfigure(2, weight=3)  # Pane 3

        # Frame styles
        style = ttk.Style(self)
        style.configure('Muted.TLabel', foreground="#00adb5")

        # -------------------------------------------------------------
        # PANE 1: Pipeline Stage Inspector (Left Column)
        # -------------------------------------------------------------
        pane1_frame = ttk.LabelFrame(self, text=" PANE 1: Pipeline Stage Inspector ", padding=10)
        pane1_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        pane1_frame.grid_rowconfigure(0, weight=0)
        pane1_frame.grid_rowconfigure(1, weight=1)
        pane1_frame.grid_rowconfigure(2, weight=0)
        pane1_frame.grid_columnconfigure(0, weight=1)

        # Toolbar Frame
        toolbar_frame = ttk.Frame(pane1_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        self.paused_var = tk.BooleanVar(value=False)
        self.btn_pause = tk.Button(
            toolbar_frame, 
            text="⏸ Pause Stream", 
            command=self._toggle_pause,
            bg="#2e2e2e",
            fg="white",
            relief="flat",
            bd=0,
            padx=10,
            pady=3,
            activebackground="#ff9f1c",
            activeforeground="black"
        )
        self.btn_pause.pack(side="left", padx=5)

        # Embedded Matplotlib Figure
        self.fig, self.axes = plt.subplots(6, 1, sharex=True, figsize=(7.0, 9.0), dpi=95)
        self.fig.subplots_adjust(hspace=0.22, top=0.97, bottom=0.05, left=0.08, right=0.96)
        
        self.canvas1 = FigureCanvasTkAgg(self.fig, master=pane1_frame)
        self.canvas1.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        self.tooltip_ann = None
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

        # Bottom row of checkboxes for quick stage bypass
        bypass_frame = ttk.Frame(pane1_frame)
        bypass_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        
        self.stage_vars = {
            "cfo": tk.BooleanVar(value=True),
            "rejection": tk.BooleanVar(value=True),
            "unwrap": tk.BooleanVar(value=True),
            "weighting": tk.BooleanVar(value=True),
            "mrc": tk.BooleanVar(value=True),
        }
        
        for idx, (stage, var) in enumerate(self.stage_vars.items()):
            cb = ttk.Checkbutton(
                bypass_frame, 
                text=f"Enable Stage {idx+1}: {stage.upper()}", 
                variable=var, 
                command=self._trigger_update
            )
            cb.grid(row=0, column=idx, padx=10, sticky="w")

        # -------------------------------------------------------------
        # PANE 2: Settings Panel (Middle Column)
        # -------------------------------------------------------------
        pane2_frame = ttk.LabelFrame(self, text=" PANE 2: Pipeline Settings ", padding=15)
        pane2_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        pane2_frame.grid_columnconfigure(0, weight=1)

        # Replay / Live Mode Toggle
        mode_frame = ttk.LabelFrame(pane2_frame, text="Data Mode Source", padding=5)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        ttk.Radiobutton(mode_frame, text="Live DSP Stream", variable=self.live_mode, value=True, command=self._trigger_update).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(mode_frame, text="Offline Replay", variable=self.live_mode, value=False, command=self._trigger_update).grid(row=0, column=1, padx=10)

        # File Load & CSV Export Subframe
        btn_subframe = ttk.Frame(pane2_frame)
        btn_subframe.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        self.btn_load = ttk.Button(btn_subframe, text="Load Replay (.npz)", command=self._load_replay_file)
        self.btn_load.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_export = ttk.Button(btn_subframe, text="Export Session CSV", command=self._export_session_csv)
        self.btn_export.pack(side="left", fill="x", expand=True)

        # Denoising & Scene Classification settings
        denoise_lf = ttk.LabelFrame(pane2_frame, text=" Spatio-Temporal Denoising & Scene Classification ", padding=10)
        denoise_lf.grid(row=2, column=0, sticky="ew", pady=5)
        denoise_lf.grid_columnconfigure(1, weight=1)

        ttk.Label(denoise_lf, text="Smoothing Window (N):").grid(row=0, column=0, sticky="w")
        self.slider_smoothing = ttk.Scale(denoise_lf, from_=1, to=10, value=8, command=lambda v: self._trigger_update())
        self.slider_smoothing.grid(row=0, column=1, sticky="ew", padx=5)

        self.lbl_scene_badge = tk.Label(
            denoise_lf,
            text="SCENE: DETECTING...",
            bg="#2e2e2e",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=5,
            relief="flat"
        )
        self.lbl_scene_badge.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # CFO correction settings
        cfo_lf = ttk.LabelFrame(pane2_frame, text="Stage 1: CFO Correction", padding=10)
        cfo_lf.grid(row=3, column=0, sticky="ew", pady=5)
        self.cfo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfo_lf, text="Enable CFO Compensator", variable=self.cfo_var, command=self._trigger_update).pack(anchor="w")

        # Unwrap Method dropdown
        unwrap_lf = ttk.LabelFrame(pane2_frame, text="Stage 3: Phase Unwrap Method", padding=10)
        unwrap_lf.grid(row=4, column=0, sticky="ew", pady=5)
        self.unwrap_var = tk.StringVar(value="weighted")
        unwrap_cb = ttk.Combobox(unwrap_lf, textvariable=self.unwrap_var, values=["numpy", "itoh", "weighted"], state="readonly")
        unwrap_cb.pack(fill="x")
        unwrap_cb.bind("<<ComboboxSelected>>", lambda e: self._trigger_update())

        # Bad tone rejection threshold slider
        rej_lf = ttk.LabelFrame(pane2_frame, text="Stage 2: Bad Tone Rejection", padding=10)
        rej_lf.grid(row=5, column=0, sticky="ew", pady=5)
        
        ttk.Label(rej_lf, text="Amp Dip Limit (dB):").grid(row=0, column=0, sticky="w")
        self.slider_amp = ttk.Scale(rej_lf, from_=0, to=30, value=10, command=lambda v: self._trigger_update())
        self.slider_amp.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(rej_lf, text="Phase Jump Limit (rad):").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.slider_disc = ttk.Scale(rej_lf, from_=0.1 * math.pi, to=math.pi, value=1.57, command=lambda v: self._trigger_update())
        self.slider_disc.grid(row=1, column=1, sticky="ew", padx=5, pady=(10, 0))

        # Exponent Slider
        weight_lf = ttk.LabelFrame(pane2_frame, text="Stage 4: Amp Weight Exponent", padding=10)
        weight_lf.grid(row=6, column=0, sticky="ew", pady=5)
        self.slider_exp = ttk.Scale(weight_lf, from_=0, to=3, value=3, command=lambda v: self._trigger_update())
        self.slider_exp.pack(fill="x")

        # Subarray size spinner (MUSIC)
        music_lf = ttk.LabelFrame(pane2_frame, text="MUSIC Model Settings", padding=10)
        music_lf.grid(row=7, column=0, sticky="ew", pady=5)
        ttk.Label(music_lf, text="Subarray Len:").grid(row=0, column=0, sticky="w")
        self.music_spinner = ttk.Spinbox(music_lf, from_=4, to=36, width=10, command=self._trigger_update)
        self.music_spinner.set(16)
        self.music_spinner.grid(row=0, column=1, padx=5, sticky="w")

        # RANSAC Settings LabelFrame
        ransac_lf = ttk.LabelFrame(pane2_frame, text="RANSAC & IFFT Settings", padding=10)
        ransac_lf.grid(row=8, column=0, sticky="ew", pady=5)
        ransac_lf.grid_columnconfigure(1, weight=1)
        
        # 1. RANSAC Iterations
        ttk.Label(ransac_lf, text="Iterations:").grid(row=0, column=0, sticky="w", pady=2)
        self.ransac_iter_spin = ttk.Spinbox(ransac_lf, from_=10, to=500, width=8, command=self._trigger_update)
        self.ransac_iter_spin.set(100)
        self.ransac_iter_spin.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        # 2. Inlier Threshold (rad)
        ttk.Label(ransac_lf, text="Inlier Thresh (rad):").grid(row=1, column=0, sticky="w", pady=2)
        self.slider_ransac_thresh = ttk.Scale(ransac_lf, from_=0.05, to=1.0, value=0.2, command=lambda v: self._trigger_update())
        self.slider_ransac_thresh.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        # 3. Min Sample Size
        ttk.Label(ransac_lf, text="Min Samples:").grid(row=2, column=0, sticky="w", pady=2)
        self.ransac_sample_spin = ttk.Spinbox(ransac_lf, from_=2, to=30, width=8, command=self._trigger_update)
        self.ransac_sample_spin.set(6)
        self.ransac_sample_spin.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
        # 4. IFFT Direct Path Max (ns)
        ttk.Label(ransac_lf, text="Direct Path Max (ns):").grid(row=3, column=0, sticky="w", pady=2)
        self.slider_ifft_max = ttk.Scale(ransac_lf, from_=5.0, to=50.0, value=8.0, command=lambda v: self._trigger_update())
        self.slider_ifft_max.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        
        # 5. IFFT Multipath Ratio
        ttk.Label(ransac_lf, text="Multipath Ratio:").grid(row=4, column=0, sticky="w", pady=2)
        self.slider_ifft_ratio = ttk.Scale(ransac_lf, from_=0.1, to=1.0, value=0.7, command=lambda v: self._trigger_update())
        self.slider_ifft_ratio.grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        # Estimators dropdowns
        est_lf = ttk.LabelFrame(pane2_frame, text="Pane 3: Estimators Comparison", padding=10)
        est_lf.grid(row=9, column=0, sticky="ew", pady=10)
        
        ttk.Label(est_lf, text="Estimator A (Solid):").grid(row=0, column=0, sticky="w")
        self.est_a_var = tk.StringVar(value="Weighted LS")
        est_a_cb = ttk.Combobox(est_lf, textvariable=self.est_a_var, values=["Phase Slope", "Weighted LS", "IFFT", "MUSIC", "Auto (Scene Adaptive)", "None"], state="readonly")
        est_a_cb.grid(row=0, column=1, padx=5, sticky="ew", pady=5)
        est_a_cb.bind("<<ComboboxSelected>>", lambda e: self._trigger_update())

        ttk.Label(est_lf, text="Estimator B (Dashed):").grid(row=1, column=0, sticky="w")
        self.est_b_var = tk.StringVar(value="Phase Slope")
        est_b_cb = ttk.Combobox(est_lf, textvariable=self.est_b_var, values=["Phase Slope", "Weighted LS", "IFFT", "MUSIC", "Auto (Scene Adaptive)", "None"], state="readonly")
        est_b_cb.grid(row=1, column=1, padx=5, sticky="ew", pady=5)
        est_b_cb.bind("<<ComboboxSelected>>", lambda e: self._trigger_update())

        # Ground Truth Distance Inputs Subframe
        ttk.Label(est_lf, text="True Distance (m):").grid(row=2, column=0, sticky="w")
        
        gt_subframe = ttk.Frame(est_lf)
        gt_subframe.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        self.true_dist_var = tk.StringVar(value="1.0")
        self.true_dist_spinner = ttk.Spinbox(gt_subframe, from_=0.0, to=100.0, increment=0.1, width=7, textvariable=self.true_dist_var)
        self.true_dist_var.trace_add("write", self._on_true_distance_spin_change)
        self.true_dist_spinner.pack(side="left", padx=(0, 5))
        
        self.btn_reset = tk.Button(
            gt_subframe,
            text="Set Ground Truth & Reset",
            bg="#2e2e2e",
            fg="white",
            activebackground="#00adb5",
            activeforeground="black",
            relief="flat",
            bd=0,
            padx=8,
            pady=2,
            font=("Arial", 9, "bold"),
            command=self._on_set_ground_truth
        )
        self.btn_reset.pack(side="left")

        # Self-Tuning Optimization Engine LabelFrame
        opt_lf = ttk.LabelFrame(pane2_frame, text=" ⚙️ Real-Time Self-Tuning Optimizer ", padding=10)
        opt_lf.grid(row=10, column=0, sticky="ew", pady=10)
        opt_lf.grid_columnconfigure(0, weight=1)
        
        self.btn_autotune = ttk.Button(opt_lf, text="⚡ Optimize Parameters (Auto-Tune)", command=self._run_autotune_optimization)
        self.btn_autotune.pack(fill="x", pady=2)
        
        self.lbl_opt_status = tk.Label(
            opt_lf,
            text="Optimizer Status: Idle",
            bg="#2e2e2e",
            fg="#cccccc",
            font=("Arial", 9, "italic"),
            anchor="w",
            padx=5,
            pady=3
        )
        self.lbl_opt_status.pack(fill="x", pady=(5, 0))

        # Calibration Offset Correction subframe
        cal_lf = ttk.LabelFrame(pane2_frame, text=" 🎯 Bias Calibration Correction ", padding=8)
        cal_lf.grid(row=11, column=0, sticky="ew", pady=(0, 10))
        cal_lf.grid_columnconfigure(0, weight=1)

        self.btn_lock_cal = tk.Button(
            cal_lf,
            text="🔒 Lock Bias Correction (from current stats)",
            bg="#1a2e2e",
            fg="#00d9e8",
            activebackground="#00adb5",
            activeforeground="black",
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            font=("Arial", 9, "bold"),
            command=self._lock_calibration_offset
        )
        self.btn_lock_cal.pack(fill="x", pady=2)

        self.btn_clear_cal = tk.Button(
            cal_lf,
            text="✕ Clear Calibration",
            bg="#2e1a1a",
            fg="#ff7777",
            activebackground="#ff3333",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=8,
            pady=3,
            font=("Arial", 8),
            command=self._clear_calibration_offset
        )
        self.btn_clear_cal.pack(fill="x", pady=(2, 0))

        self.lbl_cal_status = tk.Label(
            cal_lf,
            text="Calibration: OFF (no offset applied)",
            bg="#2e2e2e",
            fg="#888888",
            font=("Arial", 9, "italic"),
            anchor="w",
            padx=5,
            pady=3
        )
        self.lbl_cal_status.pack(fill="x", pady=(4, 0))

        # -------------------------------------------------------------
        # PANE 3: Estimator Comparison Panel (Right Column)
        # -------------------------------------------------------------
        pane3_frame = ttk.LabelFrame(self, text=" PANE 3: Historical Estimator Comparison ", padding=10)
        pane3_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        pane3_frame.grid_rowconfigure(0, weight=3)
        pane3_frame.grid_rowconfigure(1, weight=1)
        pane3_frame.grid_columnconfigure(0, weight=1)

        # embedded comparison plots: ax_dist, ax_conf, ax_error
        self.fig_comp, (self.ax_dist, self.ax_conf, self.ax_error) = plt.subplots(3, 1, sharex=True, figsize=(6.0, 7.5), dpi=95)
        self.fig_comp.subplots_adjust(hspace=0.25, top=0.96, bottom=0.08, left=0.08, right=0.96)
        
        self.canvas3 = FigureCanvasTkAgg(self.fig_comp, master=pane3_frame)
        self.canvas3.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Running Incremental Statistics Panel
        stats_frame = ttk.LabelFrame(pane3_frame, text=" Ranging Statistics (Running Summary Since Reset) ", padding=10)
        stats_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        stats_frame.grid_columnconfigure((0, 1), weight=1)

        # Column Titles
        self.lbl_title_a = ttk.Label(stats_frame, text="ESTIMATOR A", font=("Courier", 11, "bold"), foreground="#00adb5")
        self.lbl_title_a.grid(row=0, column=0, sticky="w", padx=10, pady=(0, 5))
        self.lbl_title_b = ttk.Label(stats_frame, text="ESTIMATOR B", font=("Courier", 11, "bold"), foreground="#ff5722")
        self.lbl_title_b.grid(row=0, column=1, sticky="w", padx=10, pady=(0, 5))

        # RMSE (bold & colored)
        self.lbl_rmse_a = ttk.Label(stats_frame, text="RMSE:   N/A", font=("Courier", 10, "bold"))
        self.lbl_rmse_a.grid(row=1, column=0, sticky="w", padx=10)
        self.lbl_rmse_b = ttk.Label(stats_frame, text="RMSE:   N/A", font=("Courier", 10, "bold"))
        self.lbl_rmse_b.grid(row=1, column=1, sticky="w", padx=10)

        # Bias
        self.lbl_bias_a = ttk.Label(stats_frame, text="Bias:   N/A", font=("Courier", 10))
        self.lbl_bias_a.grid(row=2, column=0, sticky="w", padx=10)
        self.lbl_bias_b = ttk.Label(stats_frame, text="Bias:   N/A", font=("Courier", 10))
        self.lbl_bias_b.grid(row=2, column=1, sticky="w", padx=10)

        # Jitter
        self.lbl_jitter_a = ttk.Label(stats_frame, text="Jitter: N/A", font=("Courier", 10))
        self.lbl_jitter_a.grid(row=3, column=0, sticky="w", padx=10)
        self.lbl_jitter_b = ttk.Label(stats_frame, text="Jitter: N/A", font=("Courier", 10))
        self.lbl_jitter_b.grid(row=3, column=1, sticky="w", padx=10)

        # 95%ile
        self.lbl_pct95_a = ttk.Label(stats_frame, text="95%ile: N/A", font=("Courier", 10))
        self.lbl_pct95_a.grid(row=4, column=0, sticky="w", padx=10)
        self.lbl_pct95_b = ttk.Label(stats_frame, text="95%ile: N/A", font=("Courier", 10))
        self.lbl_pct95_b.grid(row=4, column=1, sticky="w", padx=10)

        # n
        self.lbl_n_a = ttk.Label(stats_frame, text="n =     0", font=("Courier", 10))
        self.lbl_n_a.grid(row=5, column=0, sticky="w", padx=10)
        self.lbl_n_b = ttk.Label(stats_frame, text="n =     0", font=("Courier", 10))
        self.lbl_n_b.grid(row=5, column=1, sticky="w", padx=10)

        # Shared Ground truth
        self.lbl_shared_gt = ttk.Label(stats_frame, text="Ground truth: 1.00 m", font=("Courier", 10, "italic"))
        self.lbl_shared_gt.grid(row=6, column=0, columnspan=2, pady=(8, 0), padx=10, sticky="w")

        # Scene Distribution
        self.lbl_scene_dist = ttk.Label(stats_frame, text="Scene Dist: N/A", font=("Courier", 10, "bold"))
        self.lbl_scene_dist.grid(row=7, column=0, columnspan=2, pady=(8, 0), padx=10, sticky="w")

        self.scene_canvas = tk.Canvas(stats_frame, height=15, bg="#1e1e1e", highlightthickness=0)
        self.scene_canvas.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 4), padx=10)

    def add_raw_channel_response(self, cr: ChannelResponse, subevent: Optional[SubeventResults] = None):
        """Called from parent viewer thread-safe context to push raw live sweeps."""
        if not self.live_mode.get():
            return
        if self.paused_var.get():
            return  # ignore new sweeps when paused
            
        self._last_raw_data = (cr, subevent)
        self.replay_buffer.append(cr, subevent)
        
        # Incremental live sweep addition
        self._submit_live_sweep_job(cr, subevent)

    def _toggle_pause(self):
        is_paused = not self.paused_var.get()
        self.paused_var.set(is_paused)
        if is_paused:
            self.btn_pause.config(text="▶ Resume Stream", bg="#ff9f1c", fg="black")
        else:
            self.btn_pause.config(text="⏸ Pause Stream", bg="#2e2e2e", fg="white")
            # Hide tooltip on resume
            if hasattr(self, "tooltip_ann") and self.tooltip_ann is not None:
                self.tooltip_ann.set_visible(False)
                self.fig.canvas.draw_idle()

    def _on_hover(self, event):
        # Tooltips only active when paused or not in live stream
        if not self.paused_var.get() and self.live_mode.get():
            if self.tooltip_ann is not None:
                self.tooltip_ann.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        if event.inaxes is None:
            if self.tooltip_ann is not None:
                self.tooltip_ann.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        # Find which ax we are hovering over
        ax_idx = None
        for idx, ax in enumerate(self.axes):
            if event.inaxes == ax:
                ax_idx = idx
                break

        if ax_idx is None:
            return

        # Fetch latest result data
        res = getattr(self, "_last_res_for_tooltips", None)
        if not res:
            return

        p1 = res.get("pane1_data", {})
        ch = p1.get("raw_channels", [])
        if len(ch) == 0:
            return

        # Find nearest channel
        x_mouse = event.xdata
        if x_mouse is None:
            return
        
        idx_nearest = np.argmin(np.abs(ch - x_mouse))
        chan = int(ch[idx_nearest])
        freq = 2402 + chan

        # Format tooltip text per row
        text = ""
        if ax_idx == 0:
            # Row 1: Raw Phase
            val = p1["raw_phase"][idx_nearest]
            q = p1["raw_quality_flags"][idx_nearest]
            q_str = "HIGH" if q == 0 else "MEDIUM" if q == 1 else "LOW" if q == 2 else "UNAVAILABLE"
            text = (
                f"Channel {chan} ({freq} MHz)\n"
                f"Raw phase: {val:.3f} rad\n"
                f"Quality: {q_str}\n"
                f"——\n"
                f"Core physical lesson:\n"
                f"The raw measured phase at this tone frequency. A smooth,\n"
                f"continuous linear slope indicates a clean line-of-sight path."
            )
        elif ax_idx == 1:
            # Row 2: CFO Phase
            raw = p1["raw_phase"][idx_nearest]
            corr = p1["phase_cfo"][idx_nearest]
            cfo_hz = p1.get("cfo_hz", 0.0)
            text = (
                f"Channel {chan} ({freq} MHz)\n"
                f"Raw phase: {raw:.3f} rad\n"
                f"CFO-corrected: {corr:.3f} rad\n"
                f"Applied CFO: {cfo_hz:.1f} Hz\n"
                f"——\n"
                f"Core physical lesson:\n"
                f"Carrier Frequency Offset rotates the phase linearly with time.\n"
                f"This stage removes that drift to isolate static path phase."
            )
        elif ax_idx == 2:
            # Row 3: Unwrapped Phase + Fit
            val = p1["phase_unwrapped"][idx_nearest]
            fit = p1["fit_line"][idx_nearest]
            res_val = val - fit
            text = (
                f"Channel {chan} ({freq} MHz)\n"
                f"Unwrapped phase: {val:.3f} rad\n"
                f"Linear fit value: {fit:.3f} rad\n"
                f"Residual: {res_val:+.3f} rad\n"
                f"——\n"
                f"Core physical lesson:\n"
                f"The unwrapped phase slope is directly proportional to distance.\n"
                f"Larger residuals show phase distortion from multipath reflections."
            )
        elif ax_idx == 3:
            # Row 4: Tone Mask
            rejected_mask = p1["rejected_mask"]
            status = "REJECTED" if rejected_mask[idx_nearest] else "KEPT"
            
            # Rejection reason logic from Preprocessor
            reasons = p1.get("rejection_reasons", [])
            reason = reasons[idx_nearest] if idx_nearest < len(reasons) else "—"
            text = (
                f"Channel {chan} ({freq} MHz)\n"
                f"Status: {status}\n"
                f"Reason: {reason}\n"
                f"——\n"
                f"Core physical lesson:\n"
                f"Excludes unreliable tones (deep amplitude dips, extreme phase\n"
                f"jumps, or hardware flags) to protect regression accuracy."
            )
        elif ax_idx == 4:
            # Row 5: Amplitude
            val = p1["raw_amplitude"][idx_nearest]
            med = np.median(p1["raw_amplitude"])
            dip = med - val
            
            # Find the two deepest dips, compute spacing in Hz, reflector_dist = c / (2 * spacing_hz)
            amp_data = p1["raw_amplitude"]
            sorted_indices = np.argsort(amp_data)
            dips_ch = []
            for i in sorted_indices:
                ch_candidate = ch[i]
                if not any(abs(ch_candidate - existing) < 5 for existing in dips_ch):
                    dips_ch.append(ch_candidate)
                if len(dips_ch) >= 2:
                    break
            
            if len(dips_ch) >= 2:
                spacing_mhz = abs(dips_ch[0] - dips_ch[1])
                spacing_hz = spacing_mhz * 1e6
                reflector_dist = SPEED_OF_LIGHT / (2.0 * spacing_hz)
                reflector_str = f"{reflector_dist:.2f} m"
            else:
                reflector_str = "N/A"

            text = (
                f"Channel {chan} ({freq} MHz)\n"
                f"Amplitude: {val:.1f} dBm\n"
                f"Median: {med:.1f} dBm\n"
                f"Dip depth: {dip:.1f} dB\n"
                f"——\n"
                f"Core physical lesson:\n"
                f"Deep dips occur where multi-path signals destructively interfere.\n"
                f"Spacing between nulls predicts extra reflector path of ~{reflector_str}."
            )
        elif ax_idx == 5:
            # Row 6: Weights
            val = p1["weights"][idx_nearest]
            total_w = np.sum(p1["weights"])
            pct = (val / total_w) * 100 if total_w > 0 else 0.0
            amp_val = p1["raw_amplitude"][idx_nearest]
            text = (
                f"Channel {chan} ({freq} MHz)\n"
                f"Weight: {val:.4f} ({pct:.1f}%)\n"
                f"Amplitude: {amp_val:.1f} dBm\n"
                f"——\n"
                f"Core physical lesson:\n"
                f"Applies MRC (Maximal Ratio Combining) scaling, de-weighting channels\n"
                f"near deep amplitude dips where signal SNR is severely degraded."
            )

        if not text:
            return

        # Show standard Matplotlib annotation tooltip
        ax = self.axes[ax_idx]
        if self.tooltip_ann is not None:
            try:
                self.tooltip_ann.remove()
            except Exception:
                pass
        
        # Draw dynamic tooltip annotation in the hovered subplot
        self.tooltip_ann = ax.annotate(
            text,
            xy=(chan, event.ydata if event.ydata is not None else 0.0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#1e1e1e", ec="#444444", lw=1.0, alpha=0.95),
            color="white",
            fontproperties={"family": "monospace", "size": 7.5},
            zorder=100
        )
        self.tooltip_ann.set_visible(True)
        self.fig.canvas.draw_idle()

    def _trigger_update(self):
        """Debounce settings modifications to prevent thread overflow."""
        if self._debounce_after_id is not None:
            self.after_cancel(self._debounce_after_id)
        self._debounce_after_id = self.after(50, lambda: self._submit_pipeline_job(job_type="settings_change"))

    def _submit_live_sweep_job(self, cr: ChannelResponse, subevent: Optional[SubeventResults]):
        """Directly queues a single incoming subevent for incremental calculations."""
        self._submit_pipeline_job(job_type="live_sweep", active_sweep=(cr, subevent))

    def _submit_pipeline_job(self, job_type="settings_change", active_sweep=None):
        """Assembles settings and places a recalculation job on the background worker thread."""
        self._debounce_after_id = None
        
        sweep = active_sweep if active_sweep is not None else self._last_raw_data
        if sweep is None:
            return

        # Fetch UI configs
        cfo_enabled = self.cfo_var.get() and self.stage_vars["cfo"].get()
        
        unwrap_sel = self.unwrap_var.get()
        if not unwrap_sel:
            unwrap_sel = "weighted"
            self.unwrap_var.set("weighted")

        amp_threshold = float(self.slider_amp.get())
        disc_threshold = float(self.slider_disc.get())
        exp = float(self.slider_exp.get())
        
        try:
            music_sub = int(self.music_spinner.get())
        except ValueError:
            music_sub = 16

        # If ground_truth_m is None or 0, treat true_dist as 0.0 (no truth)
        true_dist = self._ground_truth_m if (self._ground_truth_m is not None and self._ground_truth_m != 0.0) else 0.0

        est_a_name = self.est_a_var.get()
        if not est_a_name:
            est_a_name = "IFFT"
            self.est_a_var.set("IFFT")

        est_b_name = self.est_b_var.get()
        if not est_b_name:
            est_b_name = "Phase Slope"
            self.est_b_var.set("Phase Slope")

        smoothing_n = int(self.slider_smoothing.get())

        try:
            ransac_n_iter = int(self.ransac_iter_spin.get())
        except ValueError:
            ransac_n_iter = 100

        ransac_thresh = float(self.slider_ransac_thresh.get())

        try:
            ransac_min_sample = int(self.ransac_sample_spin.get())
        except ValueError:
            ransac_min_sample = 6

        ifft_max_ns = float(self.slider_ifft_max.get())
        ifft_ratio = float(self.slider_ifft_ratio.get())

        config = PreprocessorConfig(
            enable_cfo=cfo_enabled,
            enable_rejection=self.stage_vars["rejection"].get(),
            amplitude_dip_threshold_db=amp_threshold,
            phase_discontinuity_threshold=disc_threshold,
            unwrap_strategy=UnwrapStrategy(unwrap_sel),
            enable_weighting=self.stage_vars["weighting"].get(),
            enable_mrc=self.stage_vars["mrc"].get(),
            smoothing_window_size=smoothing_n,
            ransac_n_iterations=ransac_n_iter,
            ransac_inlier_threshold_rad=ransac_thresh,
            ransac_min_sample_size=ransac_min_sample,
            ifft_direct_path_max_ns=ifft_max_ns,
            ifft_multipath_ratio_threshold=ifft_ratio,
        )

        job = {
            "job_type": job_type,
            "config": config,
            "bypass_stages": {
                "cfo": not self.stage_vars["cfo"].get(),
                "rejection": not self.stage_vars["rejection"].get(),
                "unwrap": not self.stage_vars["unwrap"].get(),
                "weighting": not self.stage_vars["weighting"].get(),
                "mrc": not self.stage_vars["mrc"].get()
            },
            "exponent": exp,
            "music_sub": music_sub,
            "est_a_name": est_a_name,
            "est_b_name": est_b_name,
            "true_dist": true_dist,
            "active_sweep": sweep,
            # For live sweeps cap history to last 25 to prevent O(n²) per-frame work growth
            "history": self.replay_buffer.get_all()[-25:] if job_type == "live_sweep" else self.replay_buffer.get_all()
        }

        # Congestion Control: if we are queuing a new live_sweep, drop older pending live_sweeps to prevent backlog
        if job_type == "live_sweep":
            temp_jobs = []
            while not self.job_queue.empty():
                try:
                    pending_job = self.job_queue.get_nowait()
                    # Preserve settings change or auto-tune jobs
                    if pending_job.get("job_type") != "live_sweep":
                        temp_jobs.append(pending_job)
                except queue.Empty:
                    break
            # Put preserved jobs back in queue
            for p_job in temp_jobs:
                self.job_queue.put(p_job)

        self.job_queue.put(job)

    def _worker_loop(self):
        """Daemon processing loop running off the main GUI thread."""
        while self.running:
            try:
                job = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                job_type = job.get("job_type", "settings_change")
                config = job["config"]
                bypass = job["bypass_stages"]
                music_sub = job["music_sub"]
                exp = job["exponent"]
                est_a_name = job["est_a_name"]
                est_b_name = job["est_b_name"]
                true_dist = job["true_dist"]
                active_sweep = job["active_sweep"]
                history = job["history"]

                # Override global MUSIC dimension
                toolset.processing.cs_music._SUBARRAY_LEN = music_sub

                preprocessor = Preprocessor(config)
                scene_classifier = SceneClassifier(config)

                # Atomic reset check
                if self.reset_stats_flag:
                    self.reset_accumulators()
                    self.reset_stats_flag = False

                if job_type == "settings_change":
                    # Settings changed or file loaded: rebuild accumulators from active history
                    self.reset_accumulators()
                    
                    timeseries_a = []
                    timeseries_b = []
                    confidence_a = []
                    confidence_b = []
                    history_scenes = []

                    replay_smoother = TemporalSmoother(config.smoothing_window_size)

                    for cr_hist, sub_hist in history:
                        try:
                            smoothed_cr = replay_smoother.process(cr_hist)
                            cleaned_cr = preprocessor.preprocess(smoothed_cr, sub_hist)
                            scene = scene_classifier.classify(cleaned_cr)
                            history_scenes.append(scene)
                            
                            act_est_a = self._get_dynamic_estimator(est_a_name, scene, exp)
                            act_est_b = self._get_dynamic_estimator(est_b_name, scene, exp)
                            
                            if act_est_a is not None:
                                res_a = act_est_a(cleaned_cr)
                                if scene == SCENE_NLOS and est_a_name == "Auto (Scene Adaptive)":
                                    res_a.confidence = 0.05
                                d_a = res_a.distance_m
                                c_a = res_a.confidence
                                name_a = act_est_a.name
                            else:
                                d_a = math.nan
                                c_a = 0.0
                                name_a = "None"
                                
                            if act_est_b is not None:
                                res_b = act_est_b(cleaned_cr)
                                if scene == SCENE_NLOS and est_b_name == "Auto (Scene Adaptive)":
                                    res_b.confidence = 0.05
                                d_b = res_b.distance_m
                                c_b = res_b.confidence
                                name_b = act_est_b.name
                            else:
                                d_b = math.nan
                                c_b = 0.0
                                name_b = "None"
                            
                            timeseries_a.append(d_a)
                            confidence_a.append(c_a)
                            timeseries_b.append(d_b)
                            confidence_b.append(c_b)
                            
                            self._accumulate_single(d_a, d_b, true_dist, name_a, name_b, c_a, c_b, cr_hist.timestamp, SCENE_NAMES[scene])
                        except Exception:
                            timeseries_a.append(math.nan)
                            timeseries_b.append(math.nan)
                            confidence_a.append(0.0)
                            confidence_b.append(0.0)
                else:
                    # Live sweep: incrementally process only the new active sweep
                    cr_raw, subevent = active_sweep
                    self.live_smoother.set_window_size(config.smoothing_window_size)
                    smoothed_cr = self.live_smoother.process(cr_raw)
                    try:
                        cleaned_cr = preprocessor.preprocess(smoothed_cr, subevent)
                        scene = scene_classifier.classify(cleaned_cr)
                        
                        act_est_a = self._get_dynamic_estimator(est_a_name, scene, exp)
                        act_est_b = self._get_dynamic_estimator(est_b_name, scene, exp)
                        
                        if act_est_a is not None:
                            res_a = act_est_a(cleaned_cr)
                            if scene == SCENE_NLOS and est_a_name == "Auto (Scene Adaptive)":
                                res_a.confidence = 0.05
                            d_a = res_a.distance_m
                            c_a = res_a.confidence
                            name_a = act_est_a.name
                        else:
                            d_a = math.nan
                            c_a = 0.0
                            name_a = "None"

                        if act_est_b is not None:
                            res_b = act_est_b(cleaned_cr)
                            if scene == SCENE_NLOS and est_b_name == "Auto (Scene Adaptive)":
                                res_b.confidence = 0.05
                            d_b = res_b.distance_m
                            c_b = res_b.confidence
                            name_b = act_est_b.name
                        else:
                            d_b = math.nan
                            c_b = 0.0
                            name_b = "None"
                        
                        self._accumulate_single(d_a, d_b, true_dist, name_a, name_b, c_a, c_b, cr_raw.timestamp, SCENE_NAMES[scene])
                    except Exception:
                        pass

                    # Build comparison timeseries from active history using a fresh replay smoother
                    timeseries_a = []
                    timeseries_b = []
                    confidence_a = []
                    confidence_b = []
                    history_scenes = []
                    replay_smoother = TemporalSmoother(config.smoothing_window_size)
                    for cr_hist, sub_hist in history:
                        try:
                            smoothed_hist = replay_smoother.process(cr_hist)
                            cleaned_hist = preprocessor.preprocess(smoothed_hist, sub_hist)
                            scene_hist = scene_classifier.classify(cleaned_hist)
                            history_scenes.append(scene_hist)
                            
                            act_est_a = self._get_dynamic_estimator(est_a_name, scene_hist, exp)
                            act_est_b = self._get_dynamic_estimator(est_b_name, scene_hist, exp)
                            
                            if act_est_a is not None:
                                res_a = act_est_a(cleaned_hist)
                                if scene_hist == SCENE_NLOS and est_a_name == "Auto (Scene Adaptive)":
                                    res_a.confidence = 0.05
                                d_a = res_a.distance_m
                                c_a = res_a.confidence
                            else:
                                d_a = math.nan
                                c_a = 0.0

                            if act_est_b is not None:
                                res_b = act_est_b(cleaned_hist)
                                if scene_hist == SCENE_NLOS and est_b_name == "Auto (Scene Adaptive)":
                                    res_b.confidence = 0.05
                                d_b = res_b.distance_m
                                c_b = res_b.confidence
                            else:
                                d_b = math.nan
                                c_b = 0.0
                                    
                            timeseries_a.append(d_a)
                            confidence_a.append(c_a)
                            timeseries_b.append(d_b)
                            confidence_b.append(c_b)
                        except Exception:
                            timeseries_a.append(math.nan)
                            timeseries_b.append(math.nan)
                            confidence_a.append(0.0)
                            confidence_b.append(0.0)

                # Stage-by-stage analysis for Pane 1 (operates on the coherently smoothed sweep)
                cr_raw, subevent = active_sweep
                # Recalculate or extract smoothed raw sweep
                try:
                    active_smoother = TemporalSmoother(config.smoothing_window_size)
                    # Find matching active raw smoothed sweep
                    smoothed_active = cr_raw
                    if job_type == "settings_change":
                        for cr_h, _ in history:
                            smoothed_active = active_smoother.process(cr_h)
                    else:
                        self.live_smoother.set_window_size(config.smoothing_window_size)
                        # We can use the active running smoother's state
                        # Pushing again would duplicate, so we can just use the already smoothed_cr
                        smoothed_active = smoothed_cr
                except Exception:
                    smoothed_active = cr_raw

                pane1_data = self._process_stages_detailed(smoothed_active, subevent, config, bypass)
                
                # Extract final active scene for current sweep
                try:
                    cleaned_active = preprocessor.preprocess(smoothed_active, subevent)
                    active_scene = scene_classifier.classify(cleaned_active)
                except Exception:
                    active_scene = SCENE_MULTIPATH

                # Formulate final statistics block
                stats_metrics = self._calculate_stats_metrics(true_dist)

                # Apply calibration offsets to Pane 3 timeseries for display
                cal_a = self._cal_offset_a
                cal_b = self._cal_offset_b
                timeseries_a = [(v - cal_a) if not math.isnan(v) else math.nan for v in timeseries_a]
                timeseries_b = [(v - cal_b) if not math.isnan(v) else math.nan for v in timeseries_b]

                # Push results to GUI queue
                result = {
                    "pane1_data": pane1_data,
                    "timeseries_a": timeseries_a,
                    "timeseries_b": timeseries_b,
                    "confidence_a": confidence_a,
                    "confidence_b": confidence_b,
                    "est_a_name": est_a_name if est_a_name != "Auto (Scene Adaptive)" else (f"Auto ({act_est_a.name})" if act_est_a else "Auto (None)"),
                    "est_b_name": est_b_name if est_b_name != "Auto (Scene Adaptive)" else (f"Auto ({act_est_b.name})" if act_est_b else "Auto (None)"),
                    "true_dist": true_dist,
                    "bypass_stages": bypass,
                    "stats": stats_metrics,
                    "active_scene": active_scene,
                    "history_scenes": history_scenes,
                }
                self.result_queue.put(result)
            except Exception as e:
                import traceback
                print(f"Error in background DSP worker: {e}")
                traceback.print_exc()

    def _accumulate_single(self, d_a: float, d_b: float, true_dist: float, name_a: str, name_b: str, conf_a: float, conf_b: float, ts: float, scene_name: str = "MULTIPATH"):
        """Thread-safe incremental metrics accumulation."""
        with self.accum_lock:
            # Apply learned calibration offsets before accumulating
            d_a_cal = (d_a - self._cal_offset_a) if not math.isnan(d_a) else math.nan
            d_b_cal = (d_b - self._cal_offset_b) if not math.isnan(d_b) else math.nan

            if true_dist is None or true_dist == 0.0:
                err_a = math.nan
                err_b = math.nan
            else:
                err_a = d_a_cal - true_dist if not math.isnan(d_a_cal) else math.nan
                err_b = d_b_cal - true_dist if not math.isnan(d_b_cal) else math.nan

            if not math.isnan(d_a_cal):
                self.n_samples_a += 1
                if not math.isnan(err_a):
                    self.sum_err_a += err_a
                    self.sum_sq_err_a += err_a ** 2
                self.sum_est_a += d_a_cal
                self.sum_sq_est_a += d_a_cal ** 2

            if not math.isnan(d_b_cal):
                self.n_samples_b += 1
                if not math.isnan(err_b):
                    self.sum_err_b += err_b
                    self.sum_sq_err_b += err_b ** 2
                self.sum_est_b += d_b_cal
                self.sum_sq_est_b += d_b_cal ** 2

            # Save in export records (calibrated values)
            self.export_records.append({
                "timestamp": ts,
                "subevent_index": max(self.n_samples_a, self.n_samples_b),
                "true_distance_m": true_dist if (true_dist is not None and true_dist != 0.0) else math.nan,
                "estimator_a_name": name_a,
                "estimate_a_m": d_a_cal,
                "error_a_m": err_a,
                "confidence_a": conf_a,
                "estimator_b_name": name_b,
                "estimate_b_m": d_b_cal,
                "error_b_m": err_b,
                "confidence_b": conf_b,
                "scene": scene_name
            })

    def _calculate_stats_metrics(self, true_dist: float) -> dict:
        """Formulate incremental values into standard metrics. Thread-safe."""
        with self.accum_lock:
            metrics = {
                "n_a": self.n_samples_a,
                "n_b": self.n_samples_b,
                "rmse_a": math.nan,
                "bias_a": math.nan,
                "jitter_a": math.nan,
                "pct95_a": math.nan,
                "rmse_b": math.nan,
                "bias_b": math.nan,
                "jitter_b": math.nan,
                "pct95_b": math.nan,
            }

            # Estimator A
            if self.n_samples_a > 0:
                if true_dist is not None and true_dist != 0.0:
                    metrics["bias_a"] = self.sum_err_a / self.n_samples_a
                    metrics["rmse_a"] = math.sqrt(self.sum_sq_err_a / self.n_samples_a)
                var_a = (self.sum_sq_est_a / self.n_samples_a) - (self.sum_est_a / self.n_samples_a) ** 2
                metrics["jitter_a"] = math.sqrt(max(0.0, var_a))

            # Estimator B
            if self.n_samples_b > 0:
                if true_dist is not None and true_dist != 0.0:
                    metrics["bias_b"] = self.sum_err_b / self.n_samples_b
                    metrics["rmse_b"] = math.sqrt(self.sum_sq_err_b / self.n_samples_b)
                var_b = (self.sum_sq_est_b / self.n_samples_b) - (self.sum_est_b / self.n_samples_b) ** 2
                metrics["jitter_b"] = math.sqrt(max(0.0, var_b))

            # Recompute 95th percentile from the active logs
            errors_a = [np.abs(r["error_a_m"]) for r in self.export_records if not math.isnan(r["error_a_m"])]
            errors_b = [np.abs(r["error_b_m"]) for r in self.export_records if not math.isnan(r["error_b_m"])]

            if errors_a:
                metrics["pct95_a"] = float(np.percentile(errors_a, 95))
            if errors_b:
                metrics["pct95_b"] = float(np.percentile(errors_b, 95))

            return metrics

    def _process_stages_detailed(self, cr: ChannelResponse, subevent: Optional[SubeventResults], config: PreprocessorConfig, bypass: dict) -> dict:
        stages_outputs = {}

        # Raw snapshot
        stages_outputs["raw_channels"] = cr.channels
        stages_outputs["raw_phase"] = cr.phase_rad
        stages_outputs["raw_amplitude"] = cr.amplitude_db
        stages_outputs["raw_quality_flags"] = cr.quality_flags

        # Stage 1: CFO correction
        pre = Preprocessor(config)
        phase_rad = cr.phase_rad.copy()
        iq_per_path = cr.iq_per_path.copy()
        
        cfo_ppm = None
        if subevent is not None:
            if getattr(subevent, 'measured_freq_offset', None) is not None:
                cfo_ppm = subevent.measured_freq_offset * 0.01
            else:
                from toolset.cs_utils.cs_step import CSStepMode0
                mode0_steps = [s for s in subevent.steps if isinstance(s, CSStepMode0) and getattr(s, 'measured_freq_offset', None) is not None]
                if mode0_steps:
                    cfo_ppm = mode0_steps[0].measured_freq_offset * 0.01

        if cfo_ppm is not None and config.enable_cfo:
            freqs_hz = (2402 + cr.channels.astype(np.float64)) * 1e6
            cfo_hz_per_chan = cfo_ppm * 1e-6 * freqs_hz
            if np.mean(np.abs(cfo_hz_per_chan)) >= config.cfo_threshold_hz:
                phase_correction = 2.0 * np.pi * cfo_ppm * 1e-6
                phase_rad = (phase_rad - phase_correction).astype(np.float32)
                iq_per_path = (iq_per_path * np.exp(-1j * 2.0 * phase_correction)).astype(np.complex64)

        stages_outputs["phase_cfo"] = phase_rad

        # Stage 2: Rejection reasons mapping
        quality_flags = cr.quality_flags.copy()
        rejected_mask = np.zeros(len(cr.channels), dtype=bool)
        rejection_reasons = ["Kept"] * len(cr.channels)

        if config.enable_rejection and len(cr.channels) >= config.min_channels_for_rejection:
            if subevent is not None:
                from toolset.cs_utils.cs_step import CSStepMode2, ToneQualityIndicator
                for i, ch in enumerate(cr.channels):
                    low_quality_count = 0
                    total_paths = 0
                    for step in subevent.steps:
                        if isinstance(step, CSStepMode2) and step.channel == ch and step.tones:
                            valid_tones = [t for t in step.tones if getattr(t, 'quality', None) is not None]
                            total_paths += len(valid_tones)
                            low_quality_count += sum(1 for t in valid_tones if t.quality >= ToneQualityIndicator.TONE_QUALITY_LOW)
                    if total_paths > 0 and (low_quality_count / total_paths) > 0.5:
                        rejected_mask[i] = True
                        rejection_reasons[i] = "Low Tone Quality indicator"
            else:
                for i in range(len(cr.channels)):
                    if quality_flags[i] >= 2:
                        rejected_mask[i] = True
                        rejection_reasons[i] = "Low Quality Flag"

            # Amplitude dips
            median_amp = np.median(cr.amplitude_db)
            for i, amp in enumerate(cr.amplitude_db):
                if amp < (median_amp - config.amplitude_dip_threshold_db):
                    rejected_mask[i] = True
                    rejection_reasons[i] = f"Amp dip ({amp - median_amp:.1f} dB)"

            # Discontinuity check
            if len(cr.channels) >= 2:
                diffs = np.diff(phase_rad)
                chan_diffs = np.diff(cr.channels)
                normalized_diffs = diffs / chan_diffs
                expected_slope = np.median(normalized_diffs)
                for k in range(1, len(phase_rad)):
                    actual_diff = phase_rad[k] - phase_rad[k-1]
                    spacing = cr.channels[k] - cr.channels[k-1]
                    discontinuity = actual_diff - expected_slope * spacing
                    discontinuity = (discontinuity + np.pi) % (2.0 * np.pi) - np.pi
                    if np.abs(discontinuity) > config.phase_discontinuity_threshold:
                        rejected_mask[k] = True
                        rejection_reasons[k] = f"Phase Jump ({np.abs(discontinuity):.2f} rad)"

        stages_outputs["rejected_mask"] = rejected_mask
        stages_outputs["rejection_reasons"] = rejection_reasons

        # Detrended CFO Phase (Row 2):
        detrended_phase = None
        detrend_slope = 0.0
        if len(cr.channels) >= 2:
            try:
                a, b = np.polyfit(cr.channels, phase_rad, 1)
                detrended_phase = phase_rad - (a * cr.channels + b)
                detrend_slope = a
            except Exception:
                pass
        
        stages_outputs["phase_detrended"] = detrended_phase
        stages_outputs["detrend_slope"] = detrend_slope

        # Stage 3, 4, 5
        cr_full = pre.preprocess(cr, subevent)
        stages_outputs["phase_unwrapped"] = cr_full.phase_rad
        stages_outputs["amplitude"] = cr_full.amplitude_db
        stages_outputs["weights"] = cr_full.weights if cr_full.weights is not None else (np.ones(len(cr.channels)) / len(cr.channels))

        if len(cr.channels) >= 2:
            freqs = (2402 + cr_full.channels.astype(np.float64)) * 1e6
            slope, intercept = np.polyfit(freqs, cr_full.phase_rad, 1)
            stages_outputs["fit_line"] = slope * freqs + intercept
        else:
            stages_outputs["fit_line"] = cr_full.phase_rad

        # Run RANSAC on unwrapped phase:
        from toolset.preprocess.pipeline import run_ransac
        best_slope, best_intercept, inlier_mask = run_ransac(
            cr.channels,
            cr_full.phase_rad,
            n_iterations=config.ransac_n_iterations,
            threshold_rad=config.ransac_inlier_threshold_rad,
            min_sample_size=config.ransac_min_sample_size
        )
        stages_outputs["ransac_inlier_mask"] = inlier_mask
        stages_outputs["ransac_fit_line"] = best_slope * cr.channels + best_intercept
        stages_outputs["ransac_slope"] = best_slope
        stages_outputs["ransac_intercept"] = best_intercept

        # STAGE B: IFFT Direct Path Extraction
        n_fft = 1024
        H = np.zeros(n_fft, dtype=complex)
        d_ifft = math.nan
        tau_direct = 0.0

        if len(cr.channels) > 0:
            ch_min = cr.channels[0]
            hann = np.hanning(len(cr.channels))
            for idx in range(len(cr.channels)):
                if inlier_mask[idx]:
                    ch = cr.channels[idx]
                    H[ch - ch_min] = (10.0 ** (cr.amplitude_db[idx] / 20.0)) * np.exp(1j * cr_full.phase_rad[idx]) * hann[idx]
                    
            h = np.fft.ifft(H)
            magnitude = np.abs(h)
            t_ns = np.arange(n_fft) * (1000.0 / float(n_fft))
            
            # Peak 1 in [0, 20 ns]
            mask1 = (t_ns >= 0.0) & (t_ns <= config.ifft_direct_path_max_ns)
            
            if np.any(mask1):
                idx1 = np.argmax(magnitude[mask1])
                peak1_idx = np.where(mask1)[0][idx1]
                tau_direct = float(t_ns[peak1_idx])
                d_ifft = tau_direct * SPEED_OF_LIGHT / 1e9

        stages_outputs["d_ifft"] = d_ifft
        stages_outputs["tau_direct"] = tau_direct

        # Compute distance from RANSAC slope
        d_ransac = -best_slope * SPEED_OF_LIGHT / (2.0 * np.pi * 1e6)
        stages_outputs["d_ransac"] = d_ransac

        # Compute inlier-only RMS residual
        inlier_indices = np.where(inlier_mask)[0]
        if len(inlier_indices) > 0:
            inlier_residuals = cr_full.phase_rad[inlier_mask] - (best_slope * cr.channels[inlier_mask] + best_intercept)
            inlier_rms = np.sqrt(np.mean(inlier_residuals**2))
        else:
            inlier_rms = 0.0
        stages_outputs["inlier_rms"] = inlier_rms

        # STAGE C: Reconstructed clean phase
        # φ_reconstructed[k] = -2π × f[k] × τ_direct
        f_rel_hz = (cr.channels - cr.channels[0]) * 1e6
        phi_reconstructed = -2.0 * np.pi * f_rel_hz * (tau_direct * 1e-9)
        stages_outputs["phase_reconstructed"] = phi_reconstructed

        return stages_outputs

    def _instantiate_estimator(self, name: str, exponent: float) -> Optional[Estimator]:
        if not name or name == "None":
            return None
        if name == "Phase Slope":
            return PhaseSlopeEstimator()
        elif name == "Weighted LS":
            return WeightedLSEstimator(exponent)
        elif name == "IFFT":
            return IFFTEstimator()
        elif name == "MUSIC":
            return MUSICEstimator()
        return None

    def _get_dynamic_estimator(self, name: str, scene: int, exponent: float) -> Optional[Estimator]:
        if not name or name == "None":
            return None
        if name != "Auto (Scene Adaptive)":
            return self._instantiate_estimator(name, exponent)
        
        if scene == SCENE_LOS:
            est = PhaseSlopeEstimator()
            est.name = "Auto (Phase Slope)"
            return est
        elif scene == SCENE_MULTIPATH:
            est = IFFTEstimator()
            est.name = "Auto (IFFT)"
            return est
        elif scene == SCENE_NLOS:
            est = IFFTEstimator()
            est.name = "Auto (IFFT)"
            return est
        else:
            est = PhaseSlopeEstimator()
            est.name = "Auto (Phase Slope)"
            return est

    def _start_result_polling(self):
        """Poll background pipeline results on Tk main thread every 10 ms."""
        if not self.running:
            return

        has_new = False
        latest_res = None
        while True:
            try:
                latest_res = self.result_queue.get_nowait()
                has_new = True
            except queue.Empty:
                break

        if has_new and latest_res is not None:
            self._latest_res = latest_res
            try:
                self._update_plots(latest_res)
            except Exception as e:
                import traceback
                print(f"[ERROR] Exception during GUI plot updates: {e}")
                traceback.print_exc()

        self.after(10, self._start_result_polling)

    def _get_rmse_color(self, rmse):
        if math.isnan(rmse):
            return ""
        if rmse < 0.3:
            return "#00ff66"  # green
        elif rmse <= 1.0:
            return "#ff9100"  # orange
        else:
            return "#ff3333"  # red

    def _update_plots(self, res: dict):
        p1 = res["pane1_data"]
        bypass = res["bypass_stages"]
        
        ch = p1["raw_channels"]
        raw_phase = p1["raw_phase"]
        corrected_phase = p1["phase_cfo"]
        unwrapped = p1["phase_unwrapped"]
        fit_line = p1["fit_line"]
        rejected_mask = p1["rejected_mask"]
        amp = p1["raw_amplitude"]
        weights = p1["weights"]

        # Color configurations
        bypass_color = "#201c10"  # soft dark golden amber highlight
        default_color = "#121212"

        # -------------------------------------------------------------
        # Row 1: Raw phase per channel (scatter by quality indicator)
        # -------------------------------------------------------------
        ax = self.axes[0]
        ax.clear()
        ax.set_facecolor(default_color)
        
        # Color tones based on quality indicators
        q_indicators = p1["raw_quality_flags"]
        colors = ["#00ff66" if q == 0 else "#ff9100" if q == 1 else "#ff3333" for q in q_indicators]
        
        ax.scatter(ch, raw_phase, c=colors, s=15, zorder=3)
        ax.set_ylabel("Raw Phase (rad)", fontsize=8)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        
        # Legend custom handles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='High Quality', markerfacecolor='#00ff66', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Medium Quality', markerfacecolor='#ff9100', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Low Quality', markerfacecolor='#ff3333', markersize=6)
        ]

        # Draw the expected LOS profile based on true distance
        try:
            val = float(self.true_dist_var.get())
        except ValueError:
            val = 5.0
        
        freqs_hz = (2402.0 + ch.astype(np.float64)) * 1e6
        expected_phase = -2.0 * np.pi * freqs_hz * 2.0 * val / SPEED_OF_LIGHT
        expected_phase_wrapped = (expected_phase + np.pi) % (2.0 * np.pi) - np.pi
        
        ax.plot(ch, expected_phase_wrapped, color="white", linestyle="--", linewidth=1.5)
        legend_elements.append(Line2D([0], [0], color='white', linestyle='--', label=f"Expected LOS (d={val:.2f}m)", linewidth=1.5))
        
        ax.legend(handles=legend_elements, loc="upper right", prop={'size': 7}, facecolor="#1e1e1e")

        # Live RMS gap
        diff = (raw_phase - expected_phase_wrapped + np.pi) % (2.0 * np.pi) - np.pi
        expected_phase_rms = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else 0.0
        ax.text(0.02, 0.08, f"Multipath deviation: {expected_phase_rms:.3f} rad RMS", color="white", fontsize=8, fontweight="bold", transform=ax.transAxes, bbox=dict(facecolor="#1e1e1e", alpha=0.85, ec="white", boxstyle="round,pad=0.3"), zorder=10)

        # -------------------------------------------------------------
        # Row 2: Phase after CFO correction (overlay raw in grey, corrected in cyan, detrended in red)
        # -------------------------------------------------------------
        ax = self.axes[1]
        ax.clear()
        ax.set_facecolor(bypass_color if bypass["cfo"] else default_color)
        
        ax.plot(ch, raw_phase, color="#555555", linestyle="--", linewidth=1.0, label="Raw (Uncompensated)", zorder=2)
        ax.plot(ch, corrected_phase, color="#00adb5", marker="o", markersize=4, label="Corrected", zorder=3)
        
        detrended_phase = p1.get("phase_detrended", None)
        if detrended_phase is not None:
            ax.plot(ch, detrended_phase, color="#ff3333", marker="x", markersize=4, label="Detrended", zorder=4)
            detrend_slope = p1.get("detrend_slope", 0.0)
            ax.text(0.02, 0.08, f"Linear trend removed: slope={detrend_slope:.4f} rad/ch", color="#ff3333", fontsize=7.5, fontweight="bold", transform=ax.transAxes, bbox=dict(facecolor="#1e1e1e", alpha=0.85, ec="#ff3333", boxstyle="round,pad=0.3"), zorder=10)

        ax.set_ylabel("CFO Phase (rad)", fontsize=8)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right", prop={'size': 7}, facecolor="#1e1e1e")

        # -------------------------------------------------------------
        # Row 3: Unwrapped phase + fit (residuals shown as error bars)
        # -------------------------------------------------------------
        ax = self.axes[2]
        ax.clear()
        ax.set_facecolor(bypass_color if (bypass["cfo"] or bypass["unwrap"]) else default_color)
        
        inlier_mask = p1.get("ransac_inlier_mask", np.ones(len(ch), dtype=bool))
        outlier_mask = ~inlier_mask
        
        # Plot inliers (purple)
        ax.scatter(ch[inlier_mask], unwrapped[inlier_mask], color="#9b5de5", s=25, label="Inliers", zorder=4)
        # Plot outliers (red)
        if np.any(outlier_mask):
            ax.scatter(ch[outlier_mask], unwrapped[outlier_mask], color="#ff3333", s=25, marker="o", label="Outliers", zorder=4)
            
        # Draw RANSAC fit line (orange)
        ransac_fit = p1.get("ransac_fit_line", fit_line)
        ax.plot(ch, ransac_fit, color="#ff7a00", linestyle="-", linewidth=1.5, label="RANSAC Fit", zorder=3)
        
        # Draw reconstructed clean path line (green)
        phase_reconstructed = p1.get("phase_reconstructed", None)
        if phase_reconstructed is not None:
            # Anchor to the first inlier's unwrapped phase to overlay it beautifully
            inlier_indices = np.where(inlier_mask)[0]
            if len(inlier_indices) > 0:
                first_idx = inlier_indices[0]
                offset = unwrapped[first_idx] - phase_reconstructed[first_idx]
                ax.plot(ch, phase_reconstructed + offset, color="#00ff66", linestyle="-.", linewidth=1.5, label="Reconstructed Path", zorder=3)

        ax.set_ylabel("Unwrapped (rad)", fontsize=8)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right", prop={'size': 7}, facecolor="#1e1e1e")

        # Persistent annotation on Row 3 plot
        d_ransac = p1.get("d_ransac", math.nan)
        d_ifft = p1.get("d_ifft", math.nan)
        inlier_rms = p1.get("inlier_rms", 0.0)
        
        ax.text(0.02, 0.08, f"d_ransac = {d_ransac:.2f} m   d_ifft = {d_ifft:.2f} m   True d = {val:.2f} m   Inlier RMS = {inlier_rms:.3f} rad", color="#ff7a00", fontsize=7.5, fontweight="bold", transform=ax.transAxes, bbox=dict(facecolor="#1e1e1e", alpha=0.85, ec="#444444", boxstyle="round,pad=0.3"), zorder=10)

        # -------------------------------------------------------------
        # Row 4: Bad tone mask (bar chart)
        # -------------------------------------------------------------
        ax = self.axes[3]
        ax.clear()
        ax.set_facecolor(bypass_color if (bypass["cfo"] or bypass["rejection"]) else default_color)
        
        bar_colors = ["#ff3333" if r else "#00ff66" for r in rejected_mask]
        bar_heights = [0.5 if r else 1.0 for r in rejected_mask]
        
        bars = ax.bar(ch, bar_heights, color=bar_colors, width=0.6, zorder=3)
        ax.set_ylabel("Tone Mask", fontsize=8)
        ax.set_ylim(0, 1.25)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)

        # -------------------------------------------------------------
        # Row 5: Amplitude with rejection threshold
        # -------------------------------------------------------------
        ax = self.axes[4]
        ax.clear()
        ax.set_facecolor(bypass_color if (bypass["cfo"] or bypass["rejection"]) else default_color)
        
        ax.plot(ch, amp, color="#ff9f1c", marker="s", markersize=4, zorder=3)
        
        median_amp = np.median(amp)
        threshold_line = median_amp - float(self.slider_amp.get())
        ax.axhline(threshold_line, color="#ff3333", linestyle=":", linewidth=1.5, label="Dip Limit")
        
        ax.set_ylabel("Amp (dB)", fontsize=8)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right", prop={'size': 7}, facecolor="#1e1e1e")

        # -------------------------------------------------------------
        # Row 6: Per-channel weight
        # -------------------------------------------------------------
        ax = self.axes[5]
        ax.clear()
        ax.set_facecolor(bypass_color if (bypass["cfo"] or bypass["weighting"]) else default_color)
        
        ax.bar(ch, weights, color="#4ecdc4", width=0.6, zorder=3)
        ax.set_ylabel("Weight w[k]", fontsize=8)
        ax.set_xlabel("BLE RF Channel Index", fontsize=9)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)

        self.canvas1.draw_idle()

        # -------------------------------------------------------------
        # PANE 3: Historical Plotting
        # -------------------------------------------------------------
        self.ax_dist.clear()
        self.ax_conf.clear()
        self.ax_error.clear()

        y_a = np.array(res["timeseries_a"])
        y_b = np.array(res["timeseries_b"])
        c_a = np.array(res["confidence_a"])
        c_b = np.array(res["confidence_b"])
        true_dist = res["true_dist"]
        x_vals = np.arange(len(y_a))

        # 1. Distance comparison timeseries
        if res["est_a_name"] != "None":
            self.ax_dist.plot(x_vals, y_a, color="#00adb5", label=res["est_a_name"], linewidth=1.5, zorder=4)
        if res["est_b_name"] != "None":
            self.ax_dist.plot(x_vals, y_b, color="#ff5722", label=res["est_b_name"], linestyle="--", linewidth=1.5, zorder=3)
        
        if self._ground_truth_m is not None and self._ground_truth_m != 0.0:
            self.ax_dist.axhline(self._ground_truth_m, color="#00ff66", linestyle="-", linewidth=1.0, label="True Dist", alpha=0.8)
            self.ax_dist.fill_between(x_vals, self._ground_truth_m - 0.5, self._ground_truth_m + 0.5, color="#00ff66", alpha=0.08, label="±0.5m Band", zorder=1)

        self.ax_dist.set_ylabel("Distance (m)", fontsize=9)
        self.ax_dist.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        self.ax_dist.legend(loc="upper right", prop={'size': 7.5}, facecolor="#1e1e1e")

        # 2. Confidences timeseries
        if res["est_a_name"] != "None":
            self.ax_conf.plot(x_vals, c_a, color="#00adb5", linewidth=1.2)
        if res["est_b_name"] != "None":
            self.ax_conf.plot(x_vals, c_b, color="#ff5722", linestyle="--", linewidth=1.2)
        self.ax_conf.set_ylabel("Confidence Score", fontsize=9)
        self.ax_conf.grid(True, color="#333333", linestyle="--", linewidth=0.5)

        # 3. Signed Error timeseries
        if self._ground_truth_m is None or self._ground_truth_m == 0.0:
            self.ax_error.text(0.5, 0.5, "Set ground truth to enable error plot", color="#888888", fontsize=10, ha="center", va="center", transform=self.ax_error.transAxes)
            self.ax_error.set_ylabel("Signed Error (m)", fontsize=9)
            self.ax_error.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        else:
            self.ax_error.axhline(0.0, color="#ffffff", linestyle="-", linewidth=0.8, alpha=0.5)
            self.ax_error.fill_between(x_vals, -0.3, 0.3, color="#888888", alpha=0.2, label="±0.3m Target", zorder=1)
            
            if res["est_a_name"] != "None":
                err_a = y_a - self._ground_truth_m
                self.ax_error.plot(x_vals, err_a, color="#00adb5", linewidth=1.5, zorder=4)
            if res["est_b_name"] != "None":
                err_b = y_b - self._ground_truth_m
                self.ax_error.plot(x_vals, err_b, color="#ff5722", linestyle="--", linewidth=1.5, zorder=3)
            
            self.ax_error.set_ylabel("Signed Error (m)", fontsize=9)
            self.ax_error.set_ylim(-1.5, 1.5)
            self.ax_error.grid(True, color="#333333", linestyle="--", linewidth=0.5)
            self.ax_error.legend(loc="upper right", prop={'size': 7.5}, facecolor="#1e1e1e")

        if res["est_a_name"] == "None" and res["est_b_name"] == "None":
            self.ax_dist.text(0.5, 0.5, "No estimator selected", color="#888888", fontsize=10, ha="center", va="center", transform=self.ax_dist.transAxes)
            self.ax_conf.text(0.5, 0.5, "No estimator selected", color="#888888", fontsize=10, ha="center", va="center", transform=self.ax_conf.transAxes)
            self.ax_error.clear()
            self.ax_error.text(0.5, 0.5, "No estimator selected", color="#888888", fontsize=10, ha="center", va="center", transform=self.ax_error.transAxes)
            self.ax_error.set_ylabel("Signed Error (m)", fontsize=9)
            self.ax_error.grid(True, color="#333333", linestyle="--", linewidth=0.5)

        self.ax_error.set_xlabel("Historical CS Subevent (Time Series Index)", fontsize=9)
        self.canvas3.draw_idle()

        # Update statistics panel labels
        stats = res.get("stats", {})
        
        # A stats
        n_a = stats.get("n_a", 0)
        rmse_a = stats.get("rmse_a", math.nan)
        bias_a = stats.get("bias_a", math.nan)
        jitter_a = stats.get("jitter_a", math.nan)
        pct95_a = stats.get("pct95_a", math.nan)
        
        self.lbl_title_a.config(text=res["est_a_name"].upper())
        self.lbl_rmse_a.config(text=f"RMSE:   {f'{rmse_a:.2f} m' if not math.isnan(rmse_a) else 'N/A'}")
        self.lbl_bias_a.config(text=f"Bias:   {f'±{bias_a:.2f} m' if not math.isnan(bias_a) else 'N/A'}")
        self.lbl_jitter_a.config(text=f"Jitter: {f'{jitter_a:.2f} m' if not math.isnan(jitter_a) else 'N/A'}")
        self.lbl_pct95_a.config(text=f"95%ile: {f'{pct95_a:.2f} m' if not math.isnan(pct95_a) else 'N/A'}")
        self.lbl_n_a.config(text=f"n =     {n_a}")

        color_a = self._get_rmse_color(rmse_a)
        if color_a:
            self.lbl_rmse_a.config(foreground=color_a)
        else:
            self.lbl_rmse_a.config(foreground="")

        # B stats
        n_b = stats.get("n_b", 0)
        rmse_b = stats.get("rmse_b", math.nan)
        bias_b = stats.get("bias_b", math.nan)
        jitter_b = stats.get("jitter_b", math.nan)
        pct95_b = stats.get("pct95_b", math.nan)
        
        self.lbl_title_b.config(text=res["est_b_name"].upper())
        self.lbl_rmse_b.config(text=f"RMSE:   {f'{rmse_b:.2f} m' if not math.isnan(rmse_b) else 'N/A'}")
        self.lbl_bias_b.config(text=f"Bias:   {f'±{bias_b:.2f} m' if not math.isnan(bias_b) else 'N/A'}")
        self.lbl_jitter_b.config(text=f"Jitter: {f'{jitter_b:.2f} m' if not math.isnan(jitter_b) else 'N/A'}")
        self.lbl_pct95_b.config(text=f"95%ile: {f'{pct95_b:.2f} m' if not math.isnan(pct95_b) else 'N/A'}")
        self.lbl_n_b.config(text=f"n =     {n_b}")

        color_b = self._get_rmse_color(rmse_b)
        if color_b:
            self.lbl_rmse_b.config(foreground=color_b)
        else:
            self.lbl_rmse_b.config(foreground="")
            
        gt_val = self._ground_truth_m
        self.lbl_shared_gt.config(text=f"Ground truth: {f'{gt_val:.2f} m' if gt_val is not None else 'None'}")

        # Update active scene badge
        active_scene = res.get("active_scene", SCENE_MULTIPATH)
        if active_scene == SCENE_LOS:
            self.lbl_scene_badge.config(text="SCENE: LOS", bg="#05c46b", fg="black")
        elif active_scene == SCENE_MULTIPATH:
            self.lbl_scene_badge.config(text="SCENE: MULTIPATH", bg="#ff9f1c", fg="black")
        elif active_scene == SCENE_NLOS:
            self.lbl_scene_badge.config(text="SCENE: NLOS", bg="#ff3f3f", fg="white")

        # Update scene distribution label & canvas
        history_scenes = res.get("history_scenes", [])
        if not history_scenes:
            history_scenes = [active_scene]
        total_scenes = len(history_scenes)
        n_los = history_scenes.count(SCENE_LOS)
        n_mp = history_scenes.count(SCENE_MULTIPATH)
        n_nlos = history_scenes.count(SCENE_NLOS)

        pct_los = (n_los / total_scenes) * 100
        pct_mp = (n_mp / total_scenes) * 100
        pct_nlos = (n_nlos / total_scenes) * 100

        self.lbl_scene_dist.config(
            text=f"Scene Dist: LOS {pct_los:.0f}% | MP {pct_mp:.0f}% | NLOS {pct_nlos:.0f}%"
        )

        # Clear and redraw custom stacked bar canvas
        self.scene_canvas.delete("all")
        canvas_w = self.scene_canvas.winfo_width()
        if canvas_w <= 1:
            canvas_w = 280  # fallback width during init
        
        w_los = int(canvas_w * (pct_los / 100.0))
        w_mp = int(canvas_w * (pct_mp / 100.0))
        w_nlos = canvas_w - w_los - w_mp

        x0 = 0
        if w_los > 0:
            self.scene_canvas.create_rectangle(x0, 0, x0 + w_los, 15, fill="#05c46b", outline="")
            x0 += w_los
        if w_mp > 0:
            self.scene_canvas.create_rectangle(x0, 0, x0 + w_mp, 15, fill="#ff9f1c", outline="")
            x0 += w_mp
        if w_nlos > 0:
            self.scene_canvas.create_rectangle(x0, 0, x0 + w_nlos, 15, fill="#ff3f3f", outline="")

    def _on_true_distance_spin_change(self, *args):
        """Callback for live true distance spinner modifications without running full pipeline."""
        try:
            val = float(self.true_dist_var.get())
        except ValueError:
            return

        if self._latest_res is None or "pane1_data" not in self._latest_res:
            return

        p1 = self._latest_res["pane1_data"]
        if "raw_channels" not in p1 or "raw_phase" not in p1:
            return

        ch = p1["raw_channels"]
        raw_phase = p1["raw_phase"]
        if len(ch) == 0:
            return

        ax = self.axes[0]
        # Store current zoom limits to avoid resetting zoom
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.clear()
        ax.set_facecolor("#121212")

        # Color tones based on quality indicators
        q_indicators = p1.get("raw_quality_flags", np.zeros(len(ch)))
        colors = ["#00ff66" if q == 0 else "#ff9100" if q == 1 else "#ff3333" for q in q_indicators]

        ax.scatter(ch, raw_phase, c=colors, s=15, zorder=3)
        ax.set_ylabel("Raw Phase (rad)", fontsize=8)
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)

        # Legend custom handles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='High Quality', markerfacecolor='#00ff66', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Medium Quality', markerfacecolor='#ff9100', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Low Quality', markerfacecolor='#ff3333', markersize=6)
        ]

        # Expected phase profile: φ_LOS[k] = -2π * f[k] * 2 * d_true / c
        freqs_hz = (2402.0 + ch.astype(np.float64)) * 1e6
        expected_phase = -2.0 * np.pi * freqs_hz * 2.0 * val / SPEED_OF_LIGHT
        expected_phase_wrapped = (expected_phase + np.pi) % (2.0 * np.pi) - np.pi

        ax.plot(ch, expected_phase_wrapped, color="white", linestyle="--", linewidth=1.5)

        legend_elements.append(Line2D([0], [0], color='white', linestyle='--', label=f"Expected LOS (d={val:.2f}m)", linewidth=1.5))
        ax.legend(handles=legend_elements, loc="upper right", prop={'size': 7}, facecolor="#1e1e1e")

        # Live RMS gap
        diff = (raw_phase - expected_phase_wrapped + np.pi) % (2.0 * np.pi) - np.pi
        expected_phase_rms = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else 0.0

        ax.text(0.02, 0.08, f"Multipath deviation: {expected_phase_rms:.3f} rad RMS", color="white", fontsize=8, fontweight="bold", transform=ax.transAxes, bbox=dict(facecolor="#1e1e1e", alpha=0.85, ec="white", boxstyle="round,pad=0.3"), zorder=10)

        # Restore limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        self.canvas1.draw_idle()

    def _on_set_ground_truth(self):
        """Set ground truth distance, save pre-reset snapshot, clear deques and statistics."""
        # Drain the result queue to prevent processing-lag freeze
        latest_res = None
        while not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                if res is not None:
                    latest_res = res
            except queue.Empty:
                break
            except Exception:
                pass

        if latest_res is not None:
            try:
                self._update_plots(latest_res)
            except Exception as e:
                import traceback
                print(f"[ERROR] Exception during GUI plot updates: {e}")
                traceback.print_exc()

        try:
            val = float(self.true_dist_spinner.get())
        except ValueError:
            val = 5.0

        # a) Store spinner value
        self._ground_truth_m = val
        self.lbl_shared_gt.config(text=f"Ground truth: {val:.2f} m")

        # Save buffer snapshot before clearing
        self._save_snapshot_npz()

        # b) Clear buffer
        self.replay_buffer.clear()

        # c) Atomic reset flag for worker thread registers
        self.reset_stats_flag = True

        # d) Session Log timestamped marker
        self.session_log.append({
            "timestamp": time.time(),
            "true_distance_m": val,
            "action": "ground_truth_set"
        })

        # e) Flash the button green for 500 ms
        self.btn_reset.config(bg="#00ff66", fg="black")
        self.after(500, lambda: self.btn_reset.config(bg="#2e2e2e", fg="white"))

        # Trigger updated plot runs
        self._submit_pipeline_job()

    def _save_snapshot_npz(self):
        """Serializes current buffer items to a pre-reset NPZ file under ./sessions/."""
        items = self.replay_buffer.get_all()
        if not items:
            return

        os.makedirs("./sessions", exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"./sessions/session_pre_reset_{timestamp_str}.npz"

        try:
            kwargs = {}
            kwargs["n_sweeps"] = np.array(len(items), dtype=np.int32)
            for idx, (cr, _) in enumerate(items):
                kwargs[f"channels_{idx}"] = cr.channels
                kwargs[f"iq_real_{idx}"] = cr.iq_per_path.real.astype(np.float32)
                kwargs[f"iq_imag_{idx}"] = cr.iq_per_path.imag.astype(np.float32)
                kwargs[f"amplitude_db_{idx}"] = cr.amplitude_db
                kwargs[f"phase_rad_{idx}"] = cr.phase_rad
                kwargs[f"quality_flags_{idx}"] = cr.quality_flags
                kwargs[f"procedure_counter_{idx}"] = np.array(cr.procedure_counter, dtype=np.int32)
                kwargs[f"role_{idx}"] = np.array(int(cr.role), dtype=np.int32)
                kwargs[f"timestamp_{idx}"] = np.array(cr.timestamp, dtype=np.float64)
                if cr.weights is not None:
                    kwargs[f"weights_{idx}"] = cr.weights
            np.savez(path, **kwargs)
            print(f"Preserved raw ChannelResponse data in snapshot: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save pre-reset snapshot: {e}")

    def _export_session_csv(self):
        """Export all recorded subevents since last reset to a CSV under ./sessions/."""
        with self.accum_lock:
            records = list(self.export_records)

        if not records:
            messagebox.showwarning("No Data to Export", "There are no subevents recorded since the last reset to export.")
            return

        os.makedirs("./sessions", exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        gt_val = self._ground_truth_m if self._ground_truth_m is not None else 0.0
        filename = f"./sessions/cs_session_{timestamp_str}_dist{gt_val:.1f}m.csv"

        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "subevent_index", "true_distance_m",
                    "estimator_a_name", "estimate_a_m", "error_a_m", "confidence_a",
                    "estimator_b_name", "estimate_b_m", "error_b_m", "confidence_b",
                    "scene"
                ])
                for r in records:
                    writer.writerow([
                        r["timestamp"], r["subevent_index"], r["true_distance_m"],
                        r["estimator_a_name"], r["estimate_a_m"], r["error_a_m"], r["confidence_a"],
                        r["estimator_b_name"], r["estimate_b_m"], r["error_b_m"], r["confidence_b"],
                        r.get("scene", "MULTIPATH")
                    ])
            messagebox.showinfo("Export Successful", f"Session exported successfully to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred while saving the CSV:\n{e}")

    def _load_replay_file(self):
        """Loads captured raw sweeps from NPZ. Supports both single sweep and session snapshots."""
        file_path = filedialog.askopenfilename(
            title="Open Ranging Dataset (.npz)",
            filetypes=[("Compressed NumPy Archive", "*.npz"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            data = np.load(file_path, allow_pickle=True)
            self.replay_buffer.clear()
            self.live_mode.set(False)

            if "n_sweeps" in data:
                # Loaded a multi-sweep session snapshot
                n_sweeps = int(data["n_sweeps"])
                last_cr = None
                for idx in range(n_sweeps):
                    iq_real = data[f"iq_real_{idx}"]
                    iq_imag = data[f"iq_imag_{idx}"]
                    iq_per_path = (iq_real + 1j * iq_imag).astype(np.complex64)
                    weights = data[f"weights_{idx}"] if f"weights_{idx}" in data else None
                    
                    cr = ChannelResponse(
                        channels=data[f"channels_{idx}"],
                        iq_per_path=iq_per_path,
                        amplitude_db=data[f"amplitude_db_{idx}"],
                        phase_rad=data[f"phase_rad_{idx}"],
                        quality_flags=data[f"quality_flags_{idx}"],
                        procedure_counter=int(data[f"procedure_counter_{idx}"]),
                        role=Role(int(data[f"role_{idx}"])),
                        timestamp=float(data[f"timestamp_{idx}"]),
                        weights=weights
                    )
                    self.replay_buffer.append(cr, None)
                    last_cr = cr
                self._last_raw_data = (last_cr, None)
            else:
                # Loaded a single-sweep NPZ file
                cr = ChannelResponse.load(file_path)
                self.replay_buffer.append(cr, None)
                for i in range(49):
                    noisy_phase = cr.phase_rad + np.random.normal(0, 0.03, len(cr.phase_rad))
                    noisy_amp = cr.amplitude_db + np.random.normal(0, 0.2, len(cr.amplitude_db))
                    mag = 2048.0 * (10.0 ** (noisy_amp / 20.0))
                    noisy_iq = (mag[:, np.newaxis] * np.exp(1j * 2.0 * noisy_phase[:, np.newaxis])).astype(np.complex64)
                    
                    cr_noisy = ChannelResponse(
                        channels=cr.channels,
                        iq_per_path=noisy_iq,
                        amplitude_db=noisy_amp,
                        phase_rad=noisy_phase,
                        quality_flags=cr.quality_flags,
                        procedure_counter=cr.procedure_counter + i + 1,
                        role=cr.role,
                        timestamp=cr.timestamp + (i + 1) * 0.1
                    )
                    self.replay_buffer.append(cr_noisy, None)
                self._last_raw_data = (cr, None)

            self.reset_stats_flag = True
            self._trigger_update()
        except Exception as e:
            messagebox.showerror("Failed to Load File", f"Error details:\n{type(e).__name__}: {e}")

    def _run_autotune_optimization(self):
        """Launches the closed-loop optimization grid search on a background thread."""
        history = self.replay_buffer.get_all()
        if not history or len(history) < 2:
            self.lbl_opt_status.config(
                text="Optimizer: Need at least 2 sweeps in history",
                fg="#ff5555",
                bg="#2e1a1a"
            )
            return

        if self._ground_truth_m is None or self._ground_truth_m == 0.0:
            self.lbl_opt_status.config(
                text="Optimizer: Set Ground Truth first to optimize!",
                fg="#ff5555",
                bg="#2e1a1a"
            )
            return

        # Snapshot all Tkinter widget values HERE on the GUI thread — NEVER read widgets from a background thread
        try:
            snapshot = {
                "smoothing": int(self.slider_smoothing.get()),
                "thresh": float(self.slider_ransac_thresh.get()),
                "exp": float(self.slider_exp.get()),
                "cfo_enabled": self.cfo_var.get() and self.stage_vars["cfo"].get(),
                "rejection": self.stage_vars["rejection"].get(),
                "amp": float(self.slider_amp.get()),
                "disc": float(self.slider_disc.get()),
                "unwrap": self.unwrap_var.get() or "weighted",
                "weighting": self.stage_vars["weighting"].get(),
                "mrc": self.stage_vars["mrc"].get(),
                "ifft_max": float(self.slider_ifft_max.get()),
                "ifft_ratio": float(self.slider_ifft_ratio.get()),
            }
        except Exception:
            snapshot = {
                "smoothing": 8, "thresh": 0.20, "exp": 2.0,
                "cfo_enabled": True, "rejection": True, "amp": -10.0,
                "disc": 1.5, "unwrap": "weighted", "weighting": True,
                "mrc": True, "ifft_max": 10.0, "ifft_ratio": 0.3,
            }

        # Disable button during run to prevent concurrent searches
        self.btn_autotune.config(state="disabled", text="⚡ Optimizing Parameters...")
        self.lbl_opt_status.config(
            text="Optimizer Status: Running Grid Search on history...",
            fg="#ffd700",
            bg="#2e2e2e"
        )

        threading.Thread(
            target=self._autotune_search_worker,
            args=(list(history), float(self._ground_truth_m), snapshot),
            daemon=True
        ).start()

    def _autotune_search_worker(self, history: List[Tuple[ChannelResponse, Optional[SubeventResults]]], ground_truth: float, snapshot: dict):
        """Runs the search worker off the main GUI thread. Uses pre-snapshotted widget values — no Tkinter calls allowed here."""
        try:
            # Unpack GUI snapshot (all values already read on GUI thread)
            current_smoothing = snapshot["smoothing"]
            current_thresh = snapshot["thresh"]
            current_exp = snapshot["exp"]
            cfo_enabled_base = snapshot["cfo_enabled"]
            rejection = snapshot["rejection"]
            amp = snapshot["amp"]
            disc = snapshot["disc"]
            unwrap = snapshot["unwrap"]
            weighting = snapshot["weighting"]
            mrc = snapshot["mrc"]
            ifft_max = snapshot["ifft_max"]
            ifft_ratio = snapshot["ifft_ratio"]

            # Pass 1: Find best estimator among ["Phase Slope", "Weighted LS", "IFFT"]
            estimators = ["Phase Slope", "Weighted LS", "IFFT"]
            best_est = "IFFT"
            min_est_rmse = float("inf")
            
            for est_name in estimators:
                sq_err_sum = 0.0
                valid_count = 0
                
                smoother = TemporalSmoother(window_size=current_smoothing)
                config = PreprocessorConfig(
                    enable_cfo=cfo_enabled_base,
                    enable_rejection=rejection,
                    amplitude_dip_threshold_db=amp,
                    phase_discontinuity_threshold=disc,
                    unwrap_strategy=UnwrapStrategy(unwrap),
                    enable_weighting=weighting,
                    enable_mrc=mrc,
                    smoothing_window_size=current_smoothing,
                    ransac_n_iterations=100,
                    ransac_inlier_threshold_rad=current_thresh,
                    ransac_min_sample_size=6,
                    ifft_direct_path_max_ns=ifft_max,
                    ifft_multipath_ratio_threshold=ifft_ratio
                )
                preprocessor = Preprocessor(config)
                
                if est_name == "Phase Slope":
                    est = PhaseSlopeEstimator()
                elif est_name == "Weighted LS":
                    est = WeightedLSEstimator(current_exp)
                else:
                    est = IFFTEstimator(config)
                
                for cr, sub in history:
                    try:
                        smoothed_cr = smoother.process(cr)
                        cleaned_cr = preprocessor.preprocess(smoothed_cr, sub)
                        res = est.estimate(cleaned_cr)
                        if res is not None and not math.isnan(res.distance_m):
                            sq_err_sum += (res.distance_m - ground_truth) ** 2
                            valid_count += 1
                    except Exception:
                        pass
                
                if valid_count > 0:
                    rmse = math.sqrt(sq_err_sum / valid_count)
                    if rmse < min_est_rmse:
                        min_est_rmse = rmse
                        best_est = est_name
            
            # Pass 2: Grid search over smoothing and RANSAC threshold using best estimator
            smoothing_options = [1, 3, 5, 8]
            thresh_options = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
            
            best_smoothing = current_smoothing
            best_thresh = current_thresh
            min_grid_rmse = float("inf")
            
            for smoothing in smoothing_options:
                for thresh in thresh_options:
                    sq_err_sum = 0.0
                    valid_count = 0
                    
                    smoother = TemporalSmoother(window_size=smoothing)
                    config = PreprocessorConfig(
                        enable_cfo=cfo_enabled_base,
                        enable_rejection=rejection,
                        amplitude_dip_threshold_db=amp,
                        phase_discontinuity_threshold=disc,
                        unwrap_strategy=UnwrapStrategy(unwrap),
                        enable_weighting=weighting,
                        enable_mrc=mrc,
                        smoothing_window_size=smoothing,
                        ransac_n_iterations=100,
                        ransac_inlier_threshold_rad=thresh,
                        ransac_min_sample_size=6,
                        ifft_direct_path_max_ns=ifft_max,
                        ifft_multipath_ratio_threshold=ifft_ratio
                    )
                    preprocessor = Preprocessor(config)
                    
                    if best_est == "Phase Slope":
                        est = PhaseSlopeEstimator()
                    elif best_est == "Weighted LS":
                        est = WeightedLSEstimator(current_exp)
                    else:
                        est = IFFTEstimator(config)
                    
                    for cr, sub in history:
                        try:
                            smoothed_cr = smoother.process(cr)
                            cleaned_cr = preprocessor.preprocess(smoothed_cr, sub)
                            res = est.estimate(cleaned_cr)
                            if res is not None and not math.isnan(res.distance_m):
                                sq_err_sum += (res.distance_m - ground_truth) ** 2
                                valid_count += 1
                        except Exception:
                            pass
                    
                    if valid_count > 0:
                        rmse = math.sqrt(sq_err_sum / valid_count)
                        if rmse < min_grid_rmse:
                            min_grid_rmse = rmse
                            best_smoothing = smoothing
                            best_thresh = thresh
            
            final_rmse = min_grid_rmse if min_grid_rmse != float("inf") else 0.0
            self.after(0, lambda: self._apply_optimal_parameters(best_smoothing, best_thresh, best_est, final_rmse))
            
        except Exception as e:
            self.after(0, lambda: self._on_autotune_failed(str(e)))

    def _apply_optimal_parameters(self, smoothing: int, thresh: float, estimator: str, rmse: float):
        """Thread-safe UI callback to apply optimized pipeline configurations and redraw."""
        self.btn_autotune.config(state="normal", text="⚡ Optimize Parameters (Auto-Tune)")
        
        # Apply optimal values to sliders
        self.slider_smoothing.set(smoothing)
        self.slider_ransac_thresh.set(thresh)
        self.est_a_var.set(estimator)
        
        self.lbl_opt_status.config(
            text=f"Auto-Tuned: {estimator} (N={smoothing}, T={thresh:.2f}) | RMSE: {rmse:.2f} m",
            fg="#00ff66",
            bg="#1a2e1d"
        )
        
        # Re-trigger pipeline recalculation and UI redraw
        self._trigger_update()

    def _on_autotune_failed(self, err_msg: str):
        """Callback for error handling during optimization run."""
        self.btn_autotune.config(state="normal", text="⚡ Optimize Parameters (Auto-Tune)")
        self.lbl_opt_status.config(
            text=f"[ERROR] Optimization failed: {err_msg}",
            fg="#ff5555",
            bg="#2e1a1a"
        )

    def _lock_calibration_offset(self):
        """Read the current mean bias from accumulated stats and lock it as the correction offset.
        After locking, all future estimates are shifted by -bias so the mean output equals ground truth."""
        if self._ground_truth_m is None or self._ground_truth_m == 0.0:
            self.lbl_cal_status.config(
                text="Calibration: Set Ground Truth first!",
                fg="#ff5555", bg="#2e1a1a"
            )
            return

        with self.accum_lock:
            bias_a = (self.sum_err_a / self.n_samples_a) if self.n_samples_a > 0 else math.nan
            bias_b = (self.sum_err_b / self.n_samples_b) if self.n_samples_b > 0 else math.nan

        if math.isnan(bias_a) and math.isnan(bias_b):
            self.lbl_cal_status.config(
                text="Calibration: Need ground truth + data first (n=0)",
                fg="#ff5555", bg="#2e1a1a"
            )
            return

        # bias = mean(estimate) - ground_truth → subtract bias to correct
        if not math.isnan(bias_a):
            self._cal_offset_a = bias_a
        if not math.isnan(bias_b):
            self._cal_offset_b = bias_b

        parts = []
        if not math.isnan(bias_a):
            parts.append(f"A: {bias_a:+.3f} m")
        if not math.isnan(bias_b):
            parts.append(f"B: {bias_b:+.3f} m")

        self.lbl_cal_status.config(
            text=f"Calibration ACTIVE — offset locked: {', '.join(parts)}",
            fg="#00ff66", bg="#1a2e1d"
        )

        # Reset accumulators so stats start fresh with the corrected output
        self.reset_stats_flag = True
        self._trigger_update()

    def _clear_calibration_offset(self):
        """Remove any active calibration offset — revert to raw estimator output."""
        self._cal_offset_a = 0.0
        self._cal_offset_b = 0.0
        self.lbl_cal_status.config(
            text="Calibration: OFF (no offset applied)",
            fg="#888888", bg="#2e2e2e"
        )
        self.reset_stats_flag = True
        self._trigger_update()

    def _handle_close(self):
        """Clean up threads gracefully upon closing."""
        self.running = False
        self.destroy()
