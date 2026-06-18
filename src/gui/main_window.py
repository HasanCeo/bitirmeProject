"""
Main GUI window for Security Camera Monitoring System
"""

import cv2
import numpy as np
import logging
import time
import os
from datetime import datetime
import threading
from threading import Thread
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

from src.detectors import MotionDetector, FireDetector, FireSmokeDetector
from src.core.metadata_manager import MetadataManager
from src.core.blacklist_manager import BlacklistManager
from src.config.settings import (
    DEFAULT_MOTION_THRESHOLD, DEFAULT_MOTION_MIN_AREA, DEFAULT_FIRE_MIN_AREA,
    DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_FPS,
    ESP32_CAM_STREAM_URL, ESP32_CAM_RECONNECT_DELAY, ESP32_CAM_TIMEOUT,
    FIRE_MODEL_PATH, FIRE_CONF_THRESHOLD, FIRE_PERSIST_K, FIRE_PERSIST_N,
    FIRE_NOTIFY_COOLDOWN, DETECTED_FIRES_DIR
)
from src.config.constants import (
    COLOR_HUMAN, COLOR_FIRE, COLOR_VEHICLE, COLOR_PET,
    DEFAULT_FRAME_SKIP_INTERVAL, DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT,
    CANDIDATE_COLLECTION_FRAMES
)
from src.utils.image_utils import resize_frame
from src.utils.esp32_stream import ESP32CamStream, test_connection


class WebcamMonitor:
    """Main application window for security camera monitoring"""
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Security Camera - Human & Fire Detection System")
        self.root.geometry("1180x920")
        self.root.minsize(1000, 760)

        # Merkezi renk paleti (Koyu Dashboard teması).
        # Tüm arayüz renkleri buradan türetilir; tek yerden değiştirilebilir.
        self.colors = {
            'bg':       '#0f172a',  # uygulama zemini (koyu lacivert)
            'card':     '#1e293b',  # paneller (slate)
            'card_alt': '#273449',  # girişler / iç yüzeyler
            'border':   '#334155',  # kenarlıklar
            'accent':   '#06b6d4',  # ana vurgu (cyan)
            'accent2':  '#3b82f6',  # ikincil vurgu (mavi)
            'text':     '#e2e8f0',  # ana metin
            'muted':    '#94a3b8',  # ikincil metin
            'success':  '#22c55e',  # çalışıyor
            'danger':   '#ef4444',  # hata / durdu
            'warning':  '#f59e0b',  # uyarı / bekliyor
        }
        self.root.configure(bg=self.colors['bg'])
        self._setup_styles()
        
        # Detection state
        self.detection_enabled = True
        self.fire_detection_enabled = True
        self.car_detection_enabled = True
        self.pet_detection_enabled = True
        self.start_hour = 0
        self.end_hour = 23
        self.running = False
        self.cap = None
        self.esp32_stream = None
        self.video_source = 0  # 0=webcam, "esp32cam"=ESP32-CAM, str=video dosyası
        self.video_paused = False
        self.total_frames = 0
        self.current_frame_num = 0
        
        # Initialize light modules (sklearn/ultralytics not loaded yet)
        self.metadata_manager = MetadataManager()
        self.blacklist_manager = BlacklistManager()
        self.image_analyzer = None
        self.photo_manager = None
        self.search_engine = None
        
        # Initialize detectors
        self.motion_detector = MotionDetector(
            threshold=DEFAULT_MOTION_THRESHOLD,
            min_area=DEFAULT_MOTION_MIN_AREA,
        )
        self._yolo_detector = None  # Loaded in background for fast startup

        # Fire detection: prefer the trained YOLO fire/smoke model. Start on the
        # basic color detector as a fallback; if trusted weights are present the
        # background loader below swaps in the accurate model.
        self.fire_detector = FireDetector(min_area=DEFAULT_FIRE_MIN_AREA)
        self._fire_yolo = FireSmokeDetector(
            FIRE_MODEL_PATH,
            conf=FIRE_CONF_THRESHOLD,
            persist_k=FIRE_PERSIST_K,
            persist_n=FIRE_PERSIST_N,
        )

        # Setup GUI
        self.setup_gui()
        
        # SCHP-equivalent garment parser (loaded lazily / in background below)
        self._human_parser = None

        # Load heavy modules (sklearn, ultralytics) in background - window appears immediately
        def _load_heavy_modules():
            try:
                from src.analysis.image_analyzer import HumanImageAnalyzer
                from src.analysis.human_parser import HumanParser
                from src.core.photo_manager import PhotoManager
                from src.core.search_engine import SearchEngine
                self.image_analyzer = HumanImageAnalyzer()
                self._human_parser = HumanParser()
                self.photo_manager = PhotoManager(
                    self.image_analyzer,
                    self.metadata_manager,
                    self.blacklist_manager,
                    human_parser=self._human_parser
                )
                self.search_engine = SearchEngine(self.image_analyzer, self.metadata_manager)
                self.root.after(0, self._on_analysis_ready)

                # Warm up the garment parser (downloads from HF on first run) so
                # the first human save isn't blocked on a multi-second load.
                self.root.after(0, lambda: self.log_info("Loading clothing parser model..."))
                ok = self._human_parser.preload()
                msg = "Clothing parser ready" if ok else "Clothing parser unavailable (using heuristic)"
                self.root.after(0, lambda m=msg: self.log_info(m))
            except Exception as e:
                logging.error(f"Failed to load analysis modules: {e}")
                self.root.after(0, lambda err=str(e): self.log_info(f"Analysis load failed: {err}"))
        Thread(target=_load_heavy_modules, daemon=True).start()

        def _load_yolo():
            try:
                from src.config.settings import MODELS_DIR, YOLO_MODEL_PATH
                import requests

                model_file = Path(YOLO_MODEL_PATH)
                if not model_file.exists():
                    # Download with progress reporting
                    model_name = "yolov8n-seg.pt"
                    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
                    dest = Path(MODELS_DIR) / model_name
                    self.root.after(0, lambda: self.log_info("Downloading YOLO model..."))
                    resp = requests.get(url, stream=True, timeout=60)
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total:
                                    pct = int(downloaded * 100 / total)
                                    mb_done = downloaded / 1_048_576
                                    mb_total = total / 1_048_576
                                    msg = f"Downloading model: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)"
                                    self.root.after(0, lambda m=msg: self.log_info(m))
                    self.root.after(0, lambda: self.log_info("Download complete. Loading model..."))

                from src.detectors.yolo_multi_detector import YoloMultiObjectDetector
                detector = YoloMultiObjectDetector()
                self._yolo_detector = detector
                self.root.after(0, self._on_model_ready)
            except Exception as e:
                msg = str(e)
                logging.error(f"Failed to load YOLO model: {msg}")
                self.root.after(0, lambda m=msg: self.log_info(f"Model load failed: {m}"))
        Thread(target=_load_yolo, daemon=True).start()

        def _load_fire_model():
            # Activate the trained fire/smoke detector if trusted weights exist;
            # otherwise keep the basic color detector (already set above).
            try:
                if self._fire_yolo.preload():
                    self.fire_detector = self._fire_yolo
                    self.root.after(0, lambda: self.log_info("Fire/smoke model active (YOLO)"))
                else:
                    self.root.after(0, lambda: self.log_info(
                        f"Fire model not found - using basic detector. "
                        f"Add weights at {FIRE_MODEL_PATH} for accurate fire/smoke detection."
                    ))
            except Exception as e:
                logging.error(f"Fire model load failed: {e}")
        Thread(target=_load_fire_model, daemon=True).start()
        
        # Variables for frame processing
        self.current_frame = None
        self.last_motion_time = None
        self.last_fire_time = None
        self.last_fire_notify = None  # cooldown for Telegram fire alerts
        self.frame_skip_counter = 0
        self.frame_skip_interval = DEFAULT_FRAME_SKIP_INTERVAL
        self.last_humans_detected = []
        self.last_fires_detected = []
        self.last_cars_detected = []
        self.last_pets_detected = []
        self.previous_human_tracks = set()
        self.previous_car_tracks = set()
        self.previous_pet_tracks = set()
        
        # Start camera automatically
        self.root.after(100, self.start_detection)
    
    def _setup_styles(self):
        """Tüm ttk widget'ları için Koyu Dashboard stillerini yapılandırır."""
        c = self.colors
        style = ttk.Style()
        style.theme_use('clam')

        base = ('Segoe UI', 10)

        # Çerçeveler
        style.configure('TFrame', background=c['card'])
        style.configure('Bg.TFrame', background=c['bg'])

        # Kart paneller (LabelFrame)
        style.configure('TLabelframe', background=c['card'], bordercolor=c['border'],
                        relief='solid', borderwidth=1)
        style.configure('TLabelframe.Label', background=c['bg'], foreground=c['accent'],
                        font=('Segoe UI', 10, 'bold'))

        # Etiketler
        style.configure('TLabel', background=c['card'], foreground=c['text'], font=base)
        style.configure('Title.TLabel', background=c['bg'], foreground=c['text'],
                        font=('Segoe UI', 17, 'bold'))
        style.configure('Subtitle.TLabel', background=c['bg'], foreground=c['muted'],
                        font=('Segoe UI', 10))
        style.configure('Heading.TLabel', background=c['card'], foreground=c['text'],
                        font=('Segoe UI', 10, 'bold'))
        style.configure('Status.TLabel', background=c['card'], foreground=c['muted'],
                        font=('Segoe UI', 9))

        # Butonlar
        style.configure('TButton', background=c['card_alt'], foreground=c['text'],
                        font=('Segoe UI', 9), borderwidth=0, padding=(10, 6))
        style.map('TButton', background=[('active', c['border']), ('pressed', c['border'])],
                  foreground=[('active', c['text'])])

        style.configure('Primary.TButton', background=c['accent2'], foreground='#ffffff',
                        font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=(10, 7))
        style.map('Primary.TButton',
                  background=[('active', c['accent']), ('pressed', c['accent'])])

        style.configure('Action.TButton', background=c['accent'], foreground='#06283d',
                        font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=(12, 6))
        style.map('Action.TButton',
                  background=[('active', c['accent2']), ('pressed', c['accent2'])],
                  foreground=[('active', '#ffffff')])

        # Onay kutuları
        style.configure('TCheckbutton', background=c['card'], foreground=c['text'], font=base)
        style.map('TCheckbutton', background=[('active', c['card'])],
                  indicatorcolor=[('selected', c['accent']), ('!selected', c['card_alt'])],
                  foreground=[('active', c['text'])])

        # Giriş alanları
        style.configure('TEntry', fieldbackground=c['card_alt'], foreground=c['text'],
                        bordercolor=c['border'], insertcolor=c['text'],
                        borderwidth=1, padding=4)
        style.configure('TSpinbox', fieldbackground=c['card_alt'], foreground=c['text'],
                        background=c['card_alt'], bordercolor=c['border'],
                        arrowcolor=c['text'], insertcolor=c['text'], borderwidth=1, padding=3)
        style.configure('TCombobox', fieldbackground=c['card_alt'], background=c['card_alt'],
                        foreground=c['text'], bordercolor=c['border'], arrowcolor=c['text'],
                        borderwidth=1, padding=3)
        style.map('TCombobox', fieldbackground=[('readonly', c['card_alt'])],
                  foreground=[('readonly', c['text'])],
                  selectbackground=[('readonly', c['card_alt'])],
                  selectforeground=[('readonly', c['text'])])

        # Combobox açılır listesi (option database ile)
        self.root.option_add('*TCombobox*Listbox.background', c['card_alt'])
        self.root.option_add('*TCombobox*Listbox.foreground', c['text'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', c['accent2'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')

        # Kaydırma çubukları
        style.configure('TScrollbar', background=c['card_alt'], troughcolor=c['card'],
                        bordercolor=c['card'], arrowcolor=c['muted'])
        style.map('TScrollbar', background=[('active', c['border'])])

    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15", style='Bg.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title / header
        header = ttk.Frame(main_frame, style='Bg.TFrame')
        header.grid(row=0, column=0, columnspan=2, pady=(0, 16), sticky="w")
        ttk.Label(header, text="📹  Security Camera Monitoring System",
                  style='Title.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Gerçek zamanlı tespit · Blacklist uyarıları · Telegram bildirimleri",
                  style='Subtitle.TLabel').grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Left column frame
        left_frame = ttk.Frame(main_frame, style='Bg.TFrame')
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 12))

        # Right column frame
        right_frame = ttk.Frame(main_frame, style='Bg.TFrame')
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display
        video_frame = ttk.LabelFrame(left_frame, text="  📷 Camera View  ", padding="8")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, text="Kamera başlatılıyor...",
                                    bg="#0a0f1a", fg=self.colors['accent'],
                                    font=('Segoe UI', 11))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(left_frame, text="  ⚙️ Controls  ", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status
        status_container = ttk.Frame(control_frame)
        status_container.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(status_container, text="Status:", style='Heading.TLabel').grid(row=0, column=0, padx=(0, 10))
        self.status_label = ttk.Label(status_container, text="● Başlatılıyor...",
                                      foreground=self.colors['warning'], style='Status.TLabel')
        self.status_label.grid(row=0, column=1, sticky="w")
        
        # Detection toggles
        self._create_toggle(control_frame, "Fire Detection:", 1, 
                          lambda: setattr(self, 'fire_detection_enabled', 
                                         self.fire_detection_var.get()))
        self._create_toggle(control_frame, "Car Detection:", 2,
                          lambda: setattr(self, 'car_detection_enabled',
                                         self.car_detection_var.get()))
        self._create_toggle(control_frame, "Pet Detection (🐱🐶):", 3,
                          lambda: setattr(self, 'pet_detection_enabled',
                                         self.pet_detection_var.get()))
        
        # Video source selection
        video_source_frame = ttk.LabelFrame(control_frame, text="  📹 Video Source  ", padding="10")
        video_source_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 10))
        
        ttk.Button(video_source_frame, text="🎥 Webcam", 
                  command=self.use_webcam, style='Primary.TButton', width=15).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(video_source_frame, text="📡 ESP32-CAM", 
                  command=self.use_esp32cam, style='Primary.TButton', width=15).grid(row=0, column=1, padx=(5, 5))
        ttk.Button(video_source_frame, text="📁 Load Video", 
                  command=self.load_video_file, style='Primary.TButton', width=15).grid(row=0, column=2, padx=(5, 5))
        ttk.Button(video_source_frame, text="⏸️ Pause/Resume", 
                  command=self.toggle_pause, style='Primary.TButton', width=15).grid(row=0, column=3, padx=(5, 0))
        
        # Time settings
        time_frame = ttk.LabelFrame(control_frame, text="  🕐 Detection Hours  ", padding="10")
        time_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(time_frame, text="Start:", style='Heading.TLabel').grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.start_hour_var = tk.StringVar(value="0")
        ttk.Spinbox(time_frame, from_=0, to=23, width=8, 
                   textvariable=self.start_hour_var, font=('Segoe UI', 9)).grid(row=0, column=1, padx=5)
        
        ttk.Label(time_frame, text="End:", style='Heading.TLabel').grid(row=0, column=2, padx=(15, 5), sticky="e")
        self.end_hour_var = tk.StringVar(value="23")
        ttk.Spinbox(time_frame, from_=0, to=23, width=8, 
                   textvariable=self.end_hour_var, font=('Segoe UI', 9)).grid(row=0, column=3, padx=5)
        
        ttk.Button(time_frame, text="Update", command=self.update_hours, 
                  style='Action.TButton').grid(row=0, column=4, padx=(20, 0))
        
        # Search frame
        search_frame = ttk.LabelFrame(left_frame, text="  🔍 Human Search  ", padding="10")
        search_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        
        ttk.Label(search_frame, text="Search Query:", style='Heading.TLabel').grid(row=0, column=0, padx=(0, 10), pady=(0, 8), sticky="w")
        self.search_query_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_query_var, 
                                font=('Segoe UI', 9), width=45)
        search_entry.grid(row=0, column=1, columnspan=4, padx=(0, 10), pady=(0, 8), sticky=(tk.W, tk.E))
        search_entry.insert(0, "e.g., red hat, blue car, red truck...")
        search_entry.bind('<FocusIn>', lambda e: search_entry.delete(0, tk.END) if search_entry.get() == "e.g., red hat, blue car, red truck..." else None)
        
        ttk.Label(search_frame, text="Time Range:", style='Heading.TLabel').grid(row=1, column=0, padx=(0, 5), sticky="e")
        ttk.Label(search_frame, text="Start:", style='Status.TLabel').grid(row=1, column=1, padx=(0, 5), sticky="e")
        self.search_start_hour_var = tk.StringVar(value="0")
        ttk.Spinbox(search_frame, from_=0, to=23, width=6, 
                   textvariable=self.search_start_hour_var, font=('Segoe UI', 9)).grid(row=1, column=2, padx=5)
        
        ttk.Label(search_frame, text="End:", style='Status.TLabel').grid(row=1, column=3, padx=(10, 5), sticky="e")
        self.search_end_hour_var = tk.StringVar(value="23")
        ttk.Spinbox(search_frame, from_=0, to=23, width=6, 
                   textvariable=self.search_end_hour_var, font=('Segoe UI', 9)).grid(row=1, column=4, padx=5)
        
        self.search_button = ttk.Button(search_frame, text="🔍 Search (loading...)", 
                  command=self.search_humans, style='Action.TButton', state='disabled')
        self.search_button.grid(row=1, column=5, padx=(15, 0))
        search_frame.columnconfigure(1, weight=1)
        
        # Search results frame
        results_frame = ttk.LabelFrame(right_frame, text="  📋 Search Results  ", padding="10")
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.results_canvas = tk.Canvas(results_frame, height=220, bg=self.colors['card'],
                                       highlightthickness=1,
                                       highlightbackground=self.colors['border'])
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_canvas.yview)
        self.results_scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Blacklist panel
        blacklist_frame = ttk.LabelFrame(right_frame, text="  🚨 Security Blacklist  ", padding="10")
        blacklist_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        input_frame = ttk.Frame(blacklist_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(input_frame, text="Description:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.blacklist_query_entry = ttk.Entry(input_frame, width=25, font=('Segoe UI', 9))
        self.blacklist_query_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(input_frame, text="Type:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(5, 5))
        self.blacklist_type_var = tk.StringVar(value='human')
        type_combo = ttk.Combobox(input_frame, textvariable=self.blacklist_type_var, 
                                 values=['human', 'vehicle', 'cat', 'dog', 'any'], width=10, state='readonly')
        type_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(input_frame, text="➕ Add", command=self.add_to_blacklist).pack(side=tk.LEFT, padx=(5, 0))
        
        list_frame = ttk.Frame(blacklist_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.blacklist_listbox = tk.Listbox(list_frame, height=4, font=('Segoe UI', 9),
                                            bg=self.colors['card_alt'], fg='#fca5a5',
                                            selectbackground=self.colors['danger'],
                                            selectforeground='#ffffff',
                                            highlightthickness=1,
                                            highlightbackground=self.colors['border'],
                                            borderwidth=0, relief=tk.FLAT,
                                            selectmode=tk.SINGLE)
        blacklist_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                           command=self.blacklist_listbox.yview)
        self.blacklist_listbox.config(yscrollcommand=blacklist_scrollbar.set)
        self.blacklist_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        blacklist_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(blacklist_frame, text="🗑️ Remove Selected", 
                  command=self.remove_from_blacklist).pack(pady=(5, 0))
        
        self.update_blacklist_display()
        
        # Info display
        info_frame = ttk.LabelFrame(right_frame, text="  📝 System Logs  ", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        
        self.info_text = tk.Text(info_frame, height=6, width=70,
                                 font=('Consolas', 9), wrap=tk.WORD,
                                 bg='#0b1220', fg=self.colors['muted'],
                                 insertbackground=self.colors['text'],
                                 relief=tk.FLAT, borderwidth=0,
                                 highlightthickness=1,
                                 highlightbackground=self.colors['border'],
                                 padx=10, pady=8)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        # Log satırları için renk etiketleri
        self.info_text.tag_config('alert', foreground=self.colors['danger'])
        self.info_text.tag_config('warn', foreground=self.colors['warning'])
        self.info_text.tag_config('ok', foreground=self.colors['success'])
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        right_frame.rowconfigure(2, weight=1)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
    
    def _create_toggle(self, parent, label_text, row, command):
        """Helper to create toggle buttons"""
        container = ttk.Frame(parent)
        container.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(container, text=label_text, style='Heading.TLabel').grid(row=0, column=0, padx=(0, 10))
        
        var_name = label_text.lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        # Initialize the toggle state from the corresponding *_enabled attribute if it exists
        enabled_attr = f"{var_name}_enabled"
        initial_value = getattr(self, enabled_attr, False)
        var = tk.BooleanVar(value=initial_value)
        setattr(self, f'{var_name}_var', var)
        
        ttk.Checkbutton(container, text="Enabled", variable=var, command=command).grid(row=0, column=1, sticky="w")
    
    def _on_analysis_ready(self):
        """Called when analysis modules (search, photo save) are loaded"""
        self.search_button.config(text="🔍 Search", state='normal')
        self.log_info("Analysis modules ready")
    
    def _on_model_ready(self):
        """Called when YOLO model is loaded"""
        self.log_info("Model ready")
    
    def update_hours(self):
        """Update detection hours"""
        try:
            start = int(self.start_hour_var.get())
            end = int(self.end_hour_var.get())
            if 0 <= start <= 23 and 0 <= end <= 23:
                self.start_hour = start
                self.end_hour = end
                self.log_info(f"Detection hours updated: {start:02d}:00 - {end:02d}:59")
            else:
                messagebox.showerror("Error", "Hours must be between 0 and 23")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
    
    def add_to_blacklist(self):
        """Add entry to blacklist"""
        query = self.blacklist_query_entry.get().strip()
        object_type = self.blacklist_type_var.get()
        
        if not query:
            messagebox.showwarning("Warning", "Please enter a description")
            return
        
        self.blacklist_manager.add_entry(query, object_type)
        self.update_blacklist_display()
        self.blacklist_query_entry.delete(0, tk.END)
        self.log_info(f"🚨 Added to blacklist: {object_type} - {query}")
        messagebox.showinfo("Success", f"Added to blacklist:\n{query}")
    
    def remove_from_blacklist(self):
        """Remove entry from blacklist"""
        selection = self.blacklist_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an entry to remove")
            return
        
        index = selection[0]
        if 0 <= index < len(self.blacklist_manager.blacklist):
            entry = self.blacklist_manager.blacklist[index]
            self.blacklist_manager.remove_entry(index)
            self.update_blacklist_display()
            self.log_info(f"Removed from blacklist: {entry['query']}")
    
    def update_blacklist_display(self):
        """Update blacklist listbox display"""
        self.blacklist_listbox.delete(0, tk.END)
        for entry in self.blacklist_manager.blacklist:
            display_text = f"[{entry['object_type'].upper()}] {entry['query']}"
            self.blacklist_listbox.insert(tk.END, display_text)
    
    def search_humans(self):
        """Search for objects matching query"""
        try:
            if self.search_engine is None:
                messagebox.showinfo("Loading", "Search is still loading. Please wait a moment.")
                return
            query = self.search_query_var.get().strip()
            if not query:
                messagebox.showwarning("Warning", "Please enter a search query")
                return
            
            start_hour = int(self.search_start_hour_var.get())
            end_hour = int(self.search_end_hour_var.get())
            
            if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                messagebox.showerror("Error", "Hours must be between 0 and 23")
                return
            
            # Clear previous results
            for widget in self.results_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Perform search
            matching_images = self.search_engine.search(query, start_hour, end_hour)
            
            # Display results
            if not matching_images:
                no_result_label = ttk.Label(self.results_scrollable_frame,
                                           text=f"❌ No search results found\n\nQuery: '{query}'\nTime range: {start_hour:02d}:00 - {end_hour:02d}:59",
                                           foreground=self.colors['muted'],
                                           font=('Segoe UI', 10))
                no_result_label.grid(row=0, column=0, padx=20, pady=30, sticky="w")
            else:
                result_label = ttk.Label(self.results_scrollable_frame,
                                        text=f"✅ {len(matching_images)} match(es) found:",
                                        font=('Segoe UI', 11, 'bold'),
                                        foreground=self.colors['success'])
                result_label.grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 15), sticky="w")
                
                # Display images in a grid
                row = 1
                col = 0
                max_cols = 4
                
                for image_path, image_file, confidence in matching_images:
                    img_frame = ttk.Frame(self.results_scrollable_frame)
                    img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                    
                    try:
                        img = cv2.imread(image_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            max_size = 150
                            if w > h:
                                new_w = max_size
                                new_h = int(h * (max_size / w))
                            else:
                                new_h = max_size
                                new_w = int(w * (max_size / h))
                            
                            img_resized = cv2.resize(img, (new_w, new_h))
                            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(img_rgb)
                            img_tk = ImageTk.PhotoImage(image=img_pil)
                            
                            img_label = tk.Label(img_frame, image=img_tk)
                            img_label.image = img_tk
                            img_label.pack()
                            
                            info_text = f"{image_file[:25]}...\nConfidence: {confidence:.1%}"
                            info_label = ttk.Label(img_frame, text=info_text,
                                                  font=('Segoe UI', 8),
                                                  foreground=self.colors['muted'],
                                                  justify=tk.CENTER)
                            info_label.pack(pady=(5, 0))
                            
                            col += 1
                            if col >= max_cols:
                                col = 0
                                row += 1
                    except Exception as e:
                        logging.error(f"Error displaying image {image_path}: {e}")
                        continue
                
                self.results_scrollable_frame.update_idletasks()
                self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
                
                self.log_info(f"Search completed: {len(matching_images)} match(es) found")
                
        except Exception as e:
            logging.error(f"Error in search_humans: {e}")
            messagebox.showerror("Error", f"An error occurred during search: {str(e)}")
    
    def is_detection_time(self):
        """Check if current time is within detection hours"""
        current_hour = datetime.now().hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= current_hour <= self.end_hour
        else:  # Overnight range
            return current_hour >= self.start_hour or current_hour <= self.end_hour
    
    def _open_video_capture(self, path):
        """
        Open a video file with a decoder that won't crash the app.

        On macOS, OpenCV's FFmpeg H.264 decoder uses frame-level multithreading
        that races with the Torch / YOLO inference threads and aborts the whole
        process ("Assertion fctx->async_lock failed at pthread_frame.c"). Apple's
        AVFoundation backend uses the OS-native decoder (no pthread_frame), so we
        prefer it and only fall back to FFmpeg for containers AVFoundation can't
        open (e.g. some .mkv / .flv files).
        """
        import sys
        if sys.platform == 'darwin':
            cap = cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                logging.info("Video opened with AVFoundation backend")
                return cap
            cap.release()
            logging.warning("AVFoundation could not open video; falling back to FFmpeg")
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    def start_detection(self):
        """Start video detection"""
        if self.video_source == "esp32cam":
            # ESP32-CAM stream modu
            self.esp32_stream = ESP32CamStream(
                stream_url=ESP32_CAM_STREAM_URL,
                reconnect_delay=ESP32_CAM_RECONNECT_DELAY,
                timeout=ESP32_CAM_TIMEOUT
            )
            self.esp32_stream.open()
            
            if not self.esp32_stream.isOpened():
                self.status_label.config(text="● ESP32-CAM Hatası", foreground=self.colors['danger'])
                self.video_label.config(text="ESP32-CAM bağlantı kurulamadı",
                                        bg="#0a0f1a", fg=self.colors['danger'])
                messagebox.showerror("Error", 
                    f"ESP32-CAM'e bağlanılamadı!\n\n"
                    f"URL: {ESP32_CAM_STREAM_URL}\n\n"
                    f"Kontrol edin:\n"
                    f"1. ESP32-CAM açık ve WiFi'ye bağlı mı?\n"
                    f"2. IP adresi doğru mu? (settings.py)\n"
                    f"3. Aynı ağda mısınız?")
                return
            
            self.cap = None
            self.total_frames = 0
            
            self.running = True
            self.video_paused = False
            self.status_label.config(text="● Çalışıyor (ESP32-CAM)", foreground=self.colors['success'])
            self.log_info(f"ESP32-CAM stream başlatıldı: {ESP32_CAM_STREAM_URL}")
            
            self.video_thread = Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            return
        
        # Webcam veya video dosyası modu (mevcut davranış)
        if isinstance(self.video_source, int):
            # Windows'ta varsayılan MSMF backend'i bazı kameralarda açılamıyor;
            # DSHOW backend'i daha güvenilir. Açılamazsa varsayılana düşeriz.
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_source)
        else:
            self.cap = self._open_video_capture(self.video_source)
        if not self.cap.isOpened():
            source_name = "video file" if isinstance(self.video_source, str) else "webcam"
            self.status_label.config(text="● Kamera Hatası", foreground=self.colors['danger'])
            self.video_label.config(text=f"Could not open {source_name}",
                                    bg="#0a0f1a", fg=self.colors['danger'])
            messagebox.showerror("Error", f"Could not open {source_name}.")
            return
        
        if self.video_source == 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_VIDEO_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_VIDEO_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, DEFAULT_VIDEO_FPS)
        
        if isinstance(self.video_source, str) and self.video_source != "esp32cam":
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_num = 0
            self.log_info(f"Video loaded: {os.path.basename(self.video_source)} ({self.total_frames} frames)")
        else:
            self.total_frames = 0
        
        self.running = True
        self.video_paused = False
        source_type = "video" if isinstance(self.video_source, str) else "camera"
        self.status_label.config(text=f"● Çalışıyor ({source_type})", foreground=self.colors['success'])
        self.log_info(f"Started on {source_type}")
        
        self.video_thread = Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop video detection"""
        self.running = False
        self.video_paused = False

        # Wait for the processing thread to fully exit BEFORE releasing the
        # capture or letting a new thread start. Otherwise the old thread keeps
        # reading from a released capture and runs Torch/MPS inference at the
        # same time as the new thread, which crashes the process (SIGTRAP).
        t = getattr(self, 'video_thread', None)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5.0)
        self.video_thread = None

        if self.cap:
            self.cap.release()
            self.cap = None
        if self.esp32_stream:
            self.esp32_stream.release()
            self.esp32_stream = None
        self.status_label.config(text="● Durdu", foreground=self.colors['danger'])
        self.video_label.config(image='', text="Kamera durdu")
        self.log_info("Stopped")
        
        self.video_source = 0
        self.total_frames = 0
        self.current_frame_num = 0
    
    def load_video_file(self):
        """Load a video file for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            if self.running:
                self.stop_detection()
                time.sleep(0.2)
            
            self.video_source = file_path
            self.log_info(f"Video: {os.path.basename(file_path)}")
            
            self.root.after(100, self.start_detection)
            messagebox.showinfo("Video Loaded", f"Video loaded: {os.path.basename(file_path)}\nProcessing started automatically.")
    
    def use_webcam(self):
        """Switch back to webcam mode"""
        if self.running:
            self.stop_detection()
        
        self.video_source = 0
        self.log_info("Webcam mode")
        self.start_detection()
    
    def use_esp32cam(self):
        """Switch to ESP32-CAM mode"""
        if self.running:
            self.stop_detection()
        
        # Önce bağlantıyı test et
        self.status_label.config(text="● ESP32-CAM bağlanıyor...", foreground=self.colors['warning'])
        self.video_label.config(text="ESP32-CAM'e bağlanılıyor...",
                                bg="#0a0f1a", fg=self.colors['warning'])
        self.root.update_idletasks()
        
        self.video_source = "esp32cam"
        self.log_info(f"ESP32-CAM mode: {ESP32_CAM_STREAM_URL}")
        self.start_detection()
    
    def toggle_pause(self):
        """Pause/resume video playback"""
        if self.running:
            self.video_paused = not self.video_paused
            status = "Duraklatıldı" if self.video_paused else "Çalışıyor"
            source_type = "video" if isinstance(self.video_source, str) else "camera"
            self.status_label.config(text=f"● {status} ({source_type})",
                                    foreground=self.colors['warning'] if self.video_paused else self.colors['success'])
            self.log_info(f"Video {status.lower()}")
    
    def _notify_fire(self, frame):
        """
        Send a Telegram fire alert (text + snapshot), like a blacklist alert.

        Reuses the blacklist's TelegramNotifier. Rate-limited by
        FIRE_NOTIFY_COOLDOWN so a sustained fire doesn't spam notifications.
        No-ops when Telegram isn't configured.
        """
        notifier = getattr(self.blacklist_manager, 'notifier', None)
        if notifier is None or not getattr(notifier, 'enabled', False):
            return

        now = datetime.now()
        if self.last_fire_notify is not None and \
           (now - self.last_fire_notify).total_seconds() < FIRE_NOTIFY_COOLDOWN:
            return
        self.last_fire_notify = now

        try:
            photo_path = str(DETECTED_FIRES_DIR / f"fire_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(photo_path, frame)  # frame is BGR
            message = (
                "🔥 YANGIN UYARISI\n"
                "Kamera görüntüsünde ateş tespit edildi!\n\n"
                f"Zaman: {now.strftime('%d.%m.%Y %H:%M:%S')}"
            )
            notifier.send_alert(message, photo_path)
            logging.info("Fire Telegram alert sent")
        except Exception as e:
            logging.error(f"Fire notification failed: {e}")

    def log_info(self, message):
        """Add message to info text and log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        # İçeriğe göre renk etiketi seç
        tag = None
        if any(s in message for s in ('🚨', 'FIRE', 'Error', 'failed', 'Failed')):
            tag = 'alert'
        elif any(s in message for s in ('⏳', 'Warning', 'updated')):
            tag = 'warn'
        elif any(s in message for s in ('✅', 'ready', 'Saved', 'Started')):
            tag = 'ok'
        self.info_text.insert(tk.END, log_message + "\n", tag)
        self.info_text.see(tk.END)
        logging.info(message)
    
    def process_video(self):
        """Process video frames in a separate thread"""
        while self.running:
            if self.video_paused:
                time.sleep(0.1)
                continue
            
            # Frame oku: ESP32-CAM veya cv2.VideoCapture
            if self.video_source == "esp32cam" and self.esp32_stream:
                ret, frame = self.esp32_stream.read()
                if not ret:
                    # ESP32-CAM'den henüz frame gelmemiş olabilir, bekle
                    # Her 100 denemede durum bilgisi logla
                    if not hasattr(self, '_esp32_wait_count'):
                        self._esp32_wait_count = 0
                    self._esp32_wait_count += 1
                    if self._esp32_wait_count % 100 == 0:
                        status = self.esp32_stream.get_status()
                        logging.warning(
                            f"ESP32-CAM frame bekleniyor "
                            f"(deneme: {self._esp32_wait_count}, "
                            f"connected: {status['connected']}, "
                            f"frame_count: {status['frame_count']})"
                        )
                    time.sleep(0.05)
                    continue
                else:
                    # Frame alındı, bekleme sayacını sıfırla
                    self._esp32_wait_count = 0
            elif self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str) and self.video_source != "esp32cam":
                        # Video bitti -> başa sar (loop). Böylece kısa klipler
                        # tekrar oynar; model arka planda yüklenirken video
                        # bitmez ve tespit sonraki turlarda çalışır.
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_num = 0
                        continue
                    break
            else:
                time.sleep(0.1)
                continue
            
            if isinstance(self.video_source, str) and self.video_source != "esp32cam":
                self.current_frame_num += 1
            
            self.frame_skip_counter += 1
            should_process_detection = (self.frame_skip_counter % self.frame_skip_interval == 0)
            
            detection_active = self.is_detection_time() if self.detection_enabled else False
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame_rgb.copy()
            
            motion_detected = False
            
            if detection_active and should_process_detection:
                motion_detected, motion_mask = self.motion_detector.detect_motion(frame)
                
                # Reset detections for this step
                self.last_humans_detected = []
                self.last_cars_detected = []
                self.last_pets_detected = []
                
                if motion_detected and self._yolo_detector is None:
                    # Model henüz arka planda yükleniyor; bir kez bilgilendir.
                    if not getattr(self, '_model_wait_warned', False):
                        self._model_wait_warned = True
                        self.log_info("⏳ Model henüz yükleniyor, tespit birazdan başlayacak...")

                if motion_detected and self._yolo_detector is not None:
                    humans, vehicles, pets = self._yolo_detector.detect_all(frame)
                    self.last_humans_detected = humans
                    
                    if self.car_detection_enabled:
                        self.last_cars_detected = vehicles
                    else:
                        self.last_cars_detected = []
                    
                    if self.pet_detection_enabled:
                        self.last_pets_detected = pets
                    else:
                        self.last_pets_detected = []
                
                if self.fire_detection_enabled:
                    self.last_fires_detected = self.fire_detector.detect_fire(frame)
                else:
                    self.last_fires_detected = []
            elif not detection_active:
                self.last_humans_detected = []
                self.last_fires_detected = []
                self.last_cars_detected = []
                self.last_pets_detected = []
            
            # Draw detections
            if detection_active:
                # Fire: detection only — no bounding box drawn on the video.
                if self.fire_detection_enabled and len(self.last_fires_detected) > 0:
                    if should_process_detection:
                        current_time = datetime.now()
                        if self.last_fire_time is None or (current_time - self.last_fire_time).seconds > 3:
                            self.log_info(f"FIRE DETECTED! Count: {len(self.last_fires_detected)}")
                            self.last_fire_time = current_time
                        # Telegram alert (with its own cooldown), like blacklist
                        self._notify_fire(frame)
                
                # Draw humans
                if len(self.last_humans_detected) > 0:
                    for detection in self.last_humans_detected:
                        x, y, w, h, track_id, mask = detection
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_HUMAN, 2)
                        cv2.putText(display_frame, f"Human ID:{track_id}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUMAN, 2)
                    
                    if should_process_detection and self.photo_manager is not None:
                        current_time = datetime.now()
                        current_human_tracks = set()
                        
                        for detection in self.last_humans_detected:
                            x, y, w, h, track_id, mask = detection
                            current_human_tracks.add(track_id)
                            
                            if self.photo_manager.should_save_human(detection, current_time):
                                self.photo_manager.add_candidate_frame(
                                    track_id, frame, detection, current_time, is_human=True
                                )
                            elif track_id in self.photo_manager.human_track_candidates:
                                num_candidates = len(self.photo_manager.human_track_candidates[track_id])
                                if num_candidates < CANDIDATE_COLLECTION_FRAMES:
                                    self.photo_manager.add_candidate_frame(
                                        track_id, frame, detection, current_time, is_human=True
                                    )
                                elif num_candidates == CANDIDATE_COLLECTION_FRAMES:
                                    filepath = self.photo_manager.get_best_candidate_and_save(track_id, is_human=True)
                                    if filepath:
                                        filename = os.path.basename(filepath)
                                        self.log_info(f"HUMAN Saved: {filename}")
                        
                        lost_human_tracks = self.previous_human_tracks - current_human_tracks
                        for lost_track_id in lost_human_tracks:
                            if lost_track_id in self.photo_manager.human_track_candidates:
                                filepath = self.photo_manager.get_best_candidate_and_save(lost_track_id, is_human=True)
                                if filepath:
                                    filename = os.path.basename(filepath)
                                    self.log_info(f"HUMAN Saved: {filename}")
                        
                        self.previous_human_tracks = current_human_tracks
                
                # Draw vehicles
                if self.car_detection_enabled and len(self.last_cars_detected) > 0:
                    for vehicle_data in self.last_cars_detected:
                        x, y, w, h, class_name, track_id, mask = vehicle_data
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_VEHICLE, 2)
                        cv2.putText(display_frame, f"{class_name.capitalize()} ID:{track_id}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VEHICLE, 2)
                    
                    if should_process_detection and self.photo_manager is not None:
                        current_time = datetime.now()
                        current_car_tracks = set()
                        
                        for vehicle_data in self.last_cars_detected:
                            x, y, w, h, class_name, track_id, mask = vehicle_data
                            current_car_tracks.add(track_id)
                            
                            if self.photo_manager.should_save_car(vehicle_data, current_time):
                                self.photo_manager.add_candidate_frame(
                                    track_id, frame, vehicle_data, current_time, is_human=False
                                )
                            elif track_id in self.photo_manager.car_track_candidates:
                                num_candidates = len(self.photo_manager.car_track_candidates[track_id])
                                if num_candidates < CANDIDATE_COLLECTION_FRAMES:
                                    self.photo_manager.add_candidate_frame(
                                        track_id, frame, vehicle_data, current_time, is_human=False
                                    )
                                elif num_candidates == CANDIDATE_COLLECTION_FRAMES:
                                    filepath = self.photo_manager.get_best_candidate_and_save(track_id, is_human=False)
                                    if filepath:
                                        filename = os.path.basename(filepath)
                                        self.log_info(f"CAR Saved: {filename}")
                        
                        lost_car_tracks = self.previous_car_tracks - current_car_tracks
                        for lost_track_id in lost_car_tracks:
                            if lost_track_id in self.photo_manager.car_track_candidates:
                                filepath = self.photo_manager.get_best_candidate_and_save(lost_track_id, is_human=False)
                                if filepath:
                                    filename = os.path.basename(filepath)
                                    self.log_info(f"CAR Saved: {filename}")
                        
                        self.previous_car_tracks = current_car_tracks
                
                # Draw pets
                if self.pet_detection_enabled and len(self.last_pets_detected) > 0:
                    for pet_data in self.last_pets_detected:
                        x, y, w, h, class_name, track_id, mask = pet_data
                        emoji = "CAT" if class_name == "cat" else "DOG"
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), COLOR_PET, 2)
                        cv2.putText(display_frame, f"{emoji} {class_name.capitalize()} ID:{track_id}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PET, 2)
                    
                    if should_process_detection and self.photo_manager is not None:
                        current_time = datetime.now()
                        current_pet_tracks = set()
                        
                        for pet_data in self.last_pets_detected:
                            x, y, w, h, class_name, track_id, mask = pet_data
                            current_pet_tracks.add(track_id)
                            
                            if self.photo_manager.should_save_pet(pet_data, current_time):
                                self.photo_manager.add_candidate_frame(
                                    track_id, frame, pet_data, current_time, is_human=False
                                )
                            elif track_id in self.photo_manager.pet_track_candidates:
                                num_candidates = len(self.photo_manager.pet_track_candidates[track_id])
                                if num_candidates < CANDIDATE_COLLECTION_FRAMES:
                                    self.photo_manager.add_candidate_frame(
                                        track_id, frame, pet_data, current_time, is_human=False
                                    )
                                elif num_candidates == CANDIDATE_COLLECTION_FRAMES:
                                    filepath = self.photo_manager.save_pet_photo(frame, pet_data, current_time)
                                    if filepath:
                                        filename = os.path.basename(filepath)
                                        emoji = "CAT" if class_name == "cat" else "DOG"
                                        self.log_info(f"{emoji} Saved: {filename}")
                        
                        lost_pet_tracks = self.previous_pet_tracks - current_pet_tracks
                        for lost_track_id in lost_pet_tracks:
                            if lost_track_id in self.photo_manager.pet_track_candidates:
                                best_candidate = self.photo_manager.pet_track_candidates[lost_track_id][0]
                                filepath = self.photo_manager.save_pet_photo(
                                    frame, best_candidate['detection_tuple'], current_time
                                )
                                if filepath:
                                    filename = os.path.basename(filepath)
                                    self.log_info(f"PET Saved: {filename}")
                        
                        self.previous_pet_tracks = current_pet_tracks
            elif not detection_active:
                cv2.putText(display_frame, "Detection Disabled (Outside Hours)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Resize for display
            display_frame = resize_frame(display_frame, DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
            
            # Convert to PhotoImage and update GUI
            if self.running:
                try:
                    image = Image.fromarray(display_frame)
                    photo = ImageTk.PhotoImage(image=image)

                    # Update GUI in main thread
                    self.root.after(0, self.update_display, photo)
                except Exception as e:
                    # Window may have been destroyed
                    logging.error(f"DIAG display error: {type(e).__name__}: {e}")
                    break
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_display(self, photo):
        """Update the video display (called from main thread)"""
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()
