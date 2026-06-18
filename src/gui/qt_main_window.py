"""
Modern PySide6 (Qt) UI for the Security Camera Monitoring System.

Reuses the existing backend unchanged (detectors, analysis, core). A single
persistent QThread (VideoWorker) owns the heavy models and the capture and does
all processing; switching webcam/video happens *inside* that one loop, so there
is no stop/restart thread churn (which previously caused native crashes). The
UI is a sidebar + stacked pages: Live, Search, Blacklist.
"""

import os
import logging
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QMutex, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QStackedWidget, QCheckBox, QSpinBox,
    QLineEdit, QComboBox, QListWidget, QListWidgetItem, QScrollArea,
    QFileDialog, QMessageBox, QSizePolicy, QPlainTextEdit, QButtonGroup
)

from src.core.metadata_manager import MetadataManager
from src.core.blacklist_manager import BlacklistManager
from src.config.constants import COLOR_HUMAN, COLOR_VEHICLE, COLOR_PET, DEFAULT_FRAME_SKIP_INTERVAL
from src.config.constants import CANDIDATE_COLLECTION_FRAMES
from src.config.settings import (
    DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_FPS,
    FIRE_MODEL_PATH, FIRE_CONF_THRESHOLD, FIRE_PERSIST_K, FIRE_PERSIST_N,
    FIRE_NOTIFY_COOLDOWN, DETECTED_FIRES_DIR,
)


# ---------------------------------------------------------------------------
# Video processing worker (single persistent thread)
# ---------------------------------------------------------------------------
class VideoWorker(QThread):
    frameReady = Signal(QImage)
    logMessage = Signal(str)
    statusChanged = Signal(str, str)   # text, level (ok/warn/danger)
    modelsReady = Signal()

    def __init__(self, metadata_manager, blacklist_manager, image_analyzer):
        super().__init__()
        self.metadata_manager = metadata_manager
        self.blacklist_manager = blacklist_manager
        self.image_analyzer = image_analyzer

        # Heavy objects created inside run() (same thread that uses them)
        self.motion_detector = None
        self.yolo = None
        self.fire = None
        self.human_parser = None
        self.photo_manager = None

        # Control state (written from the GUI thread; simple assignments are
        # atomic under the GIL)
        self._alive = True
        self._paused = False
        self.detection_enabled = True
        self.fire_enabled = True
        self.car_enabled = True
        self.pet_enabled = True
        self.start_hour = 0
        self.end_hour = 23

        self._mutex = QMutex()
        self._pending_source = 0          # initial: webcam
        self._has_pending = True
        self._current_source = None
        self._cap = None

        # processing state
        self.frame_skip_counter = 0
        self.previous_human_tracks = set()
        self.previous_car_tracks = set()
        self.previous_pet_tracks = set()
        self.last_fire_time = None
        self.last_fire_notify = None

    # ---- public control API (called from GUI thread) ----
    def request_source(self, source):
        self._mutex.lock()
        self._pending_source = source
        self._has_pending = True
        self._mutex.unlock()

    def set_paused(self, paused):
        self._paused = paused

    def stop(self):
        self._alive = False

    # ---- capture management ----
    def _open_capture(self, source):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_VIDEO_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_VIDEO_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, DEFAULT_VIDEO_FPS)
            self.statusChanged.emit("● Live (webcam)", "ok")
        else:
            # macOS-safe decoder (avoids the FFmpeg H.264 threading crash)
            cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            self.statusChanged.emit(f"● Playing video: {os.path.basename(source)}", "ok")
        self._cap = cap
        self._current_source = source
        if not cap.isOpened():
            self.statusChanged.emit("● Could not open source", "danger")
            self.logMessage.emit("Could not open video source")

    # ---- model loading ----
    def _load_models(self):
        self.logMessage.emit("Loading models...")
        from src.detectors import MotionDetector, FireSmokeDetector
        from src.detectors.yolo_multi_detector import YoloMultiObjectDetector
        from src.analysis.human_parser import HumanParser
        from src.core.photo_manager import PhotoManager
        from src.config.settings import DEFAULT_MOTION_THRESHOLD, DEFAULT_MOTION_MIN_AREA

        self.motion_detector = MotionDetector(
            threshold=DEFAULT_MOTION_THRESHOLD, min_area=DEFAULT_MOTION_MIN_AREA)
        try:
            self.yolo = YoloMultiObjectDetector()
            self.logMessage.emit("Detection model ready")
        except Exception as e:
            self.logMessage.emit(f"YOLO load failed: {e}")

        self.human_parser = HumanParser()
        if self.human_parser.preload():
            self.logMessage.emit("Clothing parser ready")
        else:
            self.logMessage.emit("Clothing parser unavailable (using heuristic)")

        self.fire = FireSmokeDetector(
            FIRE_MODEL_PATH, conf=FIRE_CONF_THRESHOLD,
            persist_k=FIRE_PERSIST_K, persist_n=FIRE_PERSIST_N)
        if self.fire.preload():
            self.logMessage.emit("Fire/smoke model active (YOLO)")
        else:
            self.fire = None
            self.logMessage.emit("Fire model not found - fire detection off")

        self.photo_manager = PhotoManager(
            self.image_analyzer, self.metadata_manager,
            self.blacklist_manager, human_parser=self.human_parser)
        self.modelsReady.emit()

    # ---- main loop ----
    def run(self):
        self._load_models()
        while self._alive:
            # source switch?
            self._mutex.lock()
            pending = self._has_pending
            src = self._pending_source
            self._has_pending = False
            self._mutex.unlock()
            if pending:
                self._open_capture(src)

            if self._paused or self._cap is None:
                self.msleep(30)
                continue

            ret, frame = self._cap.read()
            if not ret:
                # video ended -> loop it; webcam hiccup -> brief wait
                if isinstance(self._current_source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.msleep(30)
                continue

            try:
                self._process(frame)
            except Exception as e:
                logging.error(f"frame processing error: {e}")
            self.msleep(15)

        if self._cap is not None:
            self._cap.release()

    def _is_detection_time(self):
        h = datetime.now().hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= h <= self.end_hour
        return h >= self.start_hour or h <= self.end_hour

    def _process(self, frame):
        self.frame_skip_counter += 1
        process_now = (self.frame_skip_counter % DEFAULT_FRAME_SKIP_INTERVAL == 0)
        active = self.detection_enabled and self._is_detection_time()

        disp = frame.copy()  # draw on BGR copy -> correct colors
        humans, vehicles, pets, fires = [], [], [], []

        if active and process_now and self.yolo is not None:
            motion, _ = self.motion_detector.detect_motion(frame)
            if motion:
                humans, vehicles, pets = self.yolo.detect_all(frame)
                if not self.car_enabled:
                    vehicles = []
                if not self.pet_enabled:
                    pets = []
            if self.fire_enabled and self.fire is not None:
                fires = self.fire.detect_fire(frame)

        if active:
            now = datetime.now()
            # Humans
            for (x, y, w, h, tid, mask) in humans:
                cv2.rectangle(disp, (x, y), (x + w, y + h), COLOR_HUMAN, 2)
                cv2.putText(disp, f"Human {tid}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUMAN, 2)
            if process_now and humans and self.photo_manager:
                self._collect(humans, frame, now, 'human')

            # Vehicles
            for (x, y, w, h, cname, tid, mask) in vehicles:
                cv2.rectangle(disp, (x, y), (x + w, y + h), COLOR_VEHICLE, 2)
                cv2.putText(disp, f"{cname} {tid}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VEHICLE, 2)
            if process_now and vehicles and self.photo_manager:
                self._collect(vehicles, frame, now, 'car')

            # Pets
            for (x, y, w, h, cname, tid, mask) in pets:
                cv2.rectangle(disp, (x, y), (x + w, y + h), COLOR_PET, 2)
                cv2.putText(disp, f"{cname} {tid}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PET, 2)
            if process_now and pets and self.photo_manager:
                self._collect(pets, frame, now, 'pet')

            # Fire: NO bounding box, just log + Telegram alert
            if self.fire_enabled and len(fires) > 0 and process_now:
                if self.last_fire_time is None or (now - self.last_fire_time).seconds > 3:
                    self.logMessage.emit(f"🔥 FIRE DETECTED! ({len(fires)})")
                    self.last_fire_time = now
                self._notify_fire(frame)

        # emit for display
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.frameReady.emit(qimg)

    def _collect(self, detections, frame, now, kind):
        """Collect candidate frames per track and save the best one when a
        track is lost (mirrors the original processing logic)."""
        pm = self.photo_manager
        is_human = (kind == 'human')
        if kind == 'human':
            cand, prev, should = pm.human_track_candidates, self.previous_human_tracks, pm.should_save_human
        elif kind == 'car':
            cand, prev, should = pm.car_track_candidates, self.previous_car_tracks, pm.should_save_car
        else:
            cand, prev, should = pm.pet_track_candidates, self.previous_pet_tracks, pm.should_save_pet

        current = set()
        for det in detections:
            tid = det[4] if is_human else det[5]
            current.add(tid)
            if should(det, now):
                pm.add_candidate_frame(tid, frame, det, now, is_human=is_human)

        # Save the best frame of every track that has just disappeared.
        for tid in (prev - current):
            if tid in cand and cand[tid]:
                if kind == 'pet':
                    best = cand[tid][0]
                    fp = pm.save_pet_photo(best['frame'], best['detection_tuple'], now)
                    del cand[tid]
                else:
                    fp = pm.get_best_candidate_and_save(tid, is_human=is_human)
                if fp:
                    self.logMessage.emit(f"Saved: {os.path.basename(fp)}")

        if kind == 'human':
            self.previous_human_tracks = current
        elif kind == 'car':
            self.previous_car_tracks = current
        else:
            self.previous_pet_tracks = current

    def _notify_fire(self, frame):
        notifier = getattr(self.blacklist_manager, 'notifier', None)
        if notifier is None or not getattr(notifier, 'enabled', False):
            return
        now = datetime.now()
        if self.last_fire_notify is not None and \
           (now - self.last_fire_notify).total_seconds() < FIRE_NOTIFY_COOLDOWN:
            return
        self.last_fire_notify = now
        try:
            path = str(DETECTED_FIRES_DIR / f"fire_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, frame)
            msg = ("🔥 YANGIN UYARISI\nKamera görüntüsünde ateş tespit edildi!\n\n"
                   f"Zaman: {now.strftime('%d.%m.%Y %H:%M:%S')}")
            notifier.send_alert(msg, path)
            self.logMessage.emit("Fire Telegram alert sent")
        except Exception as e:
            logging.error(f"Fire notification failed: {e}")


# ---------------------------------------------------------------------------
# Modern dark stylesheet
# ---------------------------------------------------------------------------
STYLE = """
* { font-family: 'SF Pro Text', 'Segoe UI', Arial; color: #e2e8f0; }
QMainWindow, QWidget#root { background: #0b1220; }

QWidget#sidebar { background: #0f172a; border-right: 1px solid #1e293b; }
QLabel#brand { font-size: 16px; font-weight: 700; color: #38bdf8; padding: 18px 16px; }
QPushButton#nav {
    text-align: left; padding: 11px 16px; border: none; border-radius: 10px;
    background: transparent; color: #94a3b8; font-size: 13px; margin: 2px 10px;
}
QPushButton#nav:hover { background: #1e293b; color: #e2e8f0; }
QPushButton#nav:checked { background: #1d4ed8; color: #ffffff; font-weight: 600; }

QFrame#card {
    background: #111c33; border: 1px solid #1e293b; border-radius: 14px;
}
QLabel#cardTitle { font-size: 12px; font-weight: 700; color: #38bdf8; }
QLabel#pageTitle { font-size: 20px; font-weight: 700; color: #f1f5f9; }
QLabel#status { font-size: 13px; font-weight: 600; }

QLabel#video {
    background: #060a14; border: 1px solid #1e293b; border-radius: 14px;
    color: #38bdf8; font-size: 14px;
}

QPushButton {
    background: #1d4ed8; color: #fff; border: none; border-radius: 9px;
    padding: 9px 14px; font-size: 12px; font-weight: 600;
}
QPushButton:hover { background: #2563eb; }
QPushButton:pressed { background: #1e40af; }
QPushButton#ghost { background: #1e293b; color: #cbd5e1; }
QPushButton#ghost:hover { background: #334155; }
QPushButton#danger { background: #b91c1c; }
QPushButton#danger:hover { background: #dc2626; }

QLineEdit, QComboBox, QSpinBox {
    background: #0b1426; border: 1px solid #243044; border-radius: 8px;
    padding: 7px 9px; color: #e2e8f0; selection-background-color: #1d4ed8;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #38bdf8; }

QCheckBox { spacing: 8px; font-size: 13px; }
QCheckBox::indicator { width: 38px; height: 20px; border-radius: 10px; background: #334155; }
QCheckBox::indicator:checked { background: #22c55e; }

QPlainTextEdit#log {
    background: #060a14; border: 1px solid #1e293b; border-radius: 10px;
    color: #94a3b8; font-family: 'Menlo','Consolas','monospace'; font-size: 11px;
}
QListWidget {
    background: #0b1426; border: 1px solid #243044; border-radius: 10px;
    padding: 4px; font-size: 12px;
}
QListWidget::item { padding: 7px; border-radius: 6px; color: #fca5a5; }
QListWidget::item:selected { background: #b91c1c; color: #fff; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #334155; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QFrame#thumb { background: #0b1426; border: 1px solid #243044; border-radius: 10px; }
QLabel#thumbCap { color: #cbd5e1; font-size: 10px; }
QLabel#thumbConf { color: #22c55e; font-size: 10px; font-weight: 700; }
"""


def _card(title=None):
    f = QFrame(); f.setObjectName("card")
    lay = QVBoxLayout(f); lay.setContentsMargins(16, 14, 16, 16); lay.setSpacing(10)
    if title:
        t = QLabel(title); t.setObjectName("cardTitle"); lay.addWidget(t)
    return f, lay


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Camera Monitoring System")
        self.resize(1280, 800)

        # Lightweight backend shared with the worker / used by the GUI thread
        from src.analysis.image_analyzer import HumanImageAnalyzer
        from src.core.search_engine import SearchEngine
        self.metadata_manager = MetadataManager()
        self.blacklist_manager = BlacklistManager()
        self.image_analyzer = HumanImageAnalyzer()
        self.search_engine = SearchEngine(self.image_analyzer, self.metadata_manager)

        self._last_frame = None
        self._build_ui()

        # Worker
        self.worker = VideoWorker(self.metadata_manager, self.blacklist_manager,
                                  self.image_analyzer)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.logMessage.connect(self._append_log)
        self.worker.statusChanged.connect(self._on_status)
        self.worker.modelsReady.connect(lambda: self.search_btn.setEnabled(True))
        self.worker.start()

    # ---------- UI construction ----------
    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)

        # Sidebar
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(210)
        sl = QVBoxLayout(side); sl.setContentsMargins(0, 8, 0, 16); sl.setSpacing(2)
        brand = QLabel("📹  SecureCam"); brand.setObjectName("brand")
        sl.addWidget(brand)
        self.nav_group = QButtonGroup(self)
        self.stack = QStackedWidget()
        for i, (label, builder) in enumerate([
            ("📡  Live", self._build_live),
            ("🔍  Search", self._build_search),
            ("🚨  Blacklist", self._build_blacklist),
        ]):
            btn = QPushButton(label); btn.setObjectName("nav"); btn.setCheckable(True)
            btn.clicked.connect(lambda _, idx=i: self.stack.setCurrentIndex(idx))
            self.nav_group.addButton(btn, i); sl.addWidget(btn)
            self.stack.addWidget(builder())
        self.nav_group.button(0).setChecked(True)
        sl.addStretch(1)
        h.addWidget(side)

        wrap = QWidget(); wl = QVBoxLayout(wrap); wl.setContentsMargins(18, 18, 18, 18)
        wl.addWidget(self.stack)
        h.addWidget(wrap, 1)

    def _build_live(self):
        page = QWidget(); lay = QHBoxLayout(page); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(16)

        # Left: video + source buttons
        left = QVBoxLayout(); left.setSpacing(12)
        self.video = QLabel("Starting camera…"); self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setMinimumSize(640, 420)
        left.addWidget(self.video, 1)

        srcbar = QHBoxLayout(); srcbar.setSpacing(8)
        b_cam = QPushButton("🎥  Webcam"); b_cam.clicked.connect(lambda: self.worker.request_source(0))
        b_vid = QPushButton("📁  Load Video"); b_vid.setObjectName("ghost"); b_vid.clicked.connect(self._load_video)
        self.b_pause = QPushButton("⏸  Pause"); self.b_pause.setObjectName("ghost"); self.b_pause.clicked.connect(self._toggle_pause)
        self.status = QLabel("● Starting…"); self.status.setObjectName("status")
        srcbar.addWidget(b_cam); srcbar.addWidget(b_vid); srcbar.addWidget(self.b_pause)
        srcbar.addStretch(1); srcbar.addWidget(self.status)
        left.addLayout(srcbar)
        lay.addLayout(left, 1)

        # Right: controls column
        col = QVBoxLayout(); col.setSpacing(14)

        card1, c1 = _card("DETECTION")
        self.cb_fire = QCheckBox("Fire / smoke"); self.cb_fire.setChecked(True)
        self.cb_car = QCheckBox("Vehicles"); self.cb_car.setChecked(True)
        self.cb_pet = QCheckBox("Pets (cat/dog)"); self.cb_pet.setChecked(True)
        self.cb_fire.toggled.connect(lambda v: setattr(self.worker, 'fire_enabled', v))
        self.cb_car.toggled.connect(lambda v: setattr(self.worker, 'car_enabled', v))
        self.cb_pet.toggled.connect(lambda v: setattr(self.worker, 'pet_enabled', v))
        for cb in (self.cb_fire, self.cb_car, self.cb_pet):
            c1.addWidget(cb)
        col.addWidget(card1)

        card2, c2 = _card("ACTIVE HOURS")
        hrow = QHBoxLayout()
        self.sp_start = QSpinBox(); self.sp_start.setRange(0, 23)
        self.sp_end = QSpinBox(); self.sp_end.setRange(0, 23); self.sp_end.setValue(23)
        apply_btn = QPushButton("Apply"); apply_btn.clicked.connect(self._apply_hours)
        hrow.addWidget(QLabel("Start")); hrow.addWidget(self.sp_start)
        hrow.addWidget(QLabel("End")); hrow.addWidget(self.sp_end); hrow.addWidget(apply_btn)
        c2.addLayout(hrow)
        col.addWidget(card2)

        card3, c3 = _card("SYSTEM LOG")
        self.log = QPlainTextEdit(); self.log.setObjectName("log"); self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        c3.addWidget(self.log)
        col.addWidget(card3, 1)

        colw = QWidget(); colw.setLayout(col); colw.setFixedWidth(310)
        lay.addWidget(colw)
        return page

    def _build_search(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setSpacing(14)
        lay.addWidget(self._title("Search detections"))

        card, c = _card()
        row = QHBoxLayout(); row.setSpacing(8)
        self.search_in = QLineEdit(); self.search_in.setPlaceholderText("e.g. red hat, blue car, black shirt…")
        self.search_in.returnPressed.connect(self._do_search)
        self.s_start = QSpinBox(); self.s_start.setRange(0, 23)
        self.s_end = QSpinBox(); self.s_end.setRange(0, 23); self.s_end.setValue(23)
        self.search_btn = QPushButton("Search"); self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._do_search)
        row.addWidget(self.search_in, 1)
        row.addWidget(QLabel("Hours")); row.addWidget(self.s_start); row.addWidget(QLabel("–")); row.addWidget(self.s_end)
        row.addWidget(self.search_btn)
        c.addLayout(row)
        lay.addWidget(card)

        self.results_area = QScrollArea(); self.results_area.setWidgetResizable(True)
        self.results_host = QWidget(); self.results_grid = QGridLayout(self.results_host)
        self.results_grid.setContentsMargins(2, 2, 2, 2); self.results_grid.setSpacing(12)
        self.results_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_area.setWidget(self.results_host)
        lay.addWidget(self.results_area, 1)
        self.results_info = QLabel("Enter a query to search saved detections.")
        self.results_grid.addWidget(self.results_info, 0, 0)
        return page

    def _build_blacklist(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setSpacing(14)
        lay.addWidget(self._title("Security blacklist"))

        card, c = _card("ADD ENTRY")
        row = QHBoxLayout(); row.setSpacing(8)
        self.bl_query = QLineEdit(); self.bl_query.setPlaceholderText("Description, e.g. red jacket")
        self.bl_type = QComboBox(); self.bl_type.addItems(['human', 'vehicle', 'cat', 'dog', 'any'])
        add_btn = QPushButton("➕  Add"); add_btn.clicked.connect(self._add_blacklist)
        row.addWidget(self.bl_query, 1); row.addWidget(self.bl_type); row.addWidget(add_btn)
        c.addLayout(row)
        lay.addWidget(card)

        card2, c2 = _card("WATCHLIST")
        self.bl_list = QListWidget(); self.bl_list.setMinimumHeight(260)
        c2.addWidget(self.bl_list)
        rm = QPushButton("🗑  Remove selected"); rm.setObjectName("danger"); rm.clicked.connect(self._remove_blacklist)
        c2.addWidget(rm, 0, Qt.AlignLeft)
        lay.addWidget(card2, 1)
        self._refresh_blacklist()
        return page

    def _title(self, text):
        lbl = QLabel(text); lbl.setObjectName("pageTitle"); return lbl

    # ---------- slots ----------
    def _on_frame(self, qimg):
        self._last_frame = qimg
        self._render_frame()

    def _render_frame(self):
        if self._last_frame is None:
            return
        pix = QPixmap.fromImage(self._last_frame).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_frame()

    def _on_status(self, text, level):
        color = {'ok': '#22c55e', 'warn': '#f59e0b', 'danger': '#ef4444'}.get(level, '#94a3b8')
        self.status.setText(text); self.status.setStyleSheet(f"color: {color};")

    def _append_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All files (*)")
        if path:
            self.worker.request_source(path)
            self._append_log(f"Video: {os.path.basename(path)}")

    def _toggle_pause(self):
        paused = not self.worker._paused
        self.worker.set_paused(paused)
        self.b_pause.setText("▶  Resume" if paused else "⏸  Pause")

    def _apply_hours(self):
        self.worker.start_hour = self.sp_start.value()
        self.worker.end_hour = self.sp_end.value()
        self._append_log(f"Active hours: {self.sp_start.value():02d}:00–{self.sp_end.value():02d}:59")

    def _do_search(self):
        # clear grid
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        query = self.search_in.text().strip()
        if not query:
            return
        results = self.search_engine.search(query, self.s_start.value(), self.s_end.value())
        if not results:
            lbl = QLabel(f"No results for '{query}'."); self.results_grid.addWidget(lbl, 0, 0)
            return
        cols = 4
        for i, (path, fname, conf) in enumerate(results):
            self.results_grid.addWidget(self._thumb(path, fname, conf), i // cols, i % cols)

    def _thumb(self, path, fname, conf):
        f = QFrame(); f.setObjectName("thumb"); f.setFixedSize(170, 190)
        v = QVBoxLayout(f); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(4)
        img = QLabel(); img.setAlignment(Qt.AlignCenter)
        pix = QPixmap(path)
        if not pix.isNull():
            img.setPixmap(pix.scaled(150, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img.setText("(missing)")
        cap = QLabel(fname[:22]); cap.setObjectName("thumbCap")
        cf = QLabel(f"{conf*100:.0f}% match"); cf.setObjectName("thumbConf")
        v.addWidget(img, 1); v.addWidget(cap); v.addWidget(cf)
        return f

    def _refresh_blacklist(self):
        self.bl_list.clear()
        for e in self.blacklist_manager.blacklist:
            self.bl_list.addItem(QListWidgetItem(f"[{e['object_type'].upper()}]  {e['query']}"))

    def _add_blacklist(self):
        q = self.bl_query.text().strip()
        if not q:
            QMessageBox.warning(self, "Blacklist", "Please enter a description.")
            return
        self.blacklist_manager.add_entry(q, self.bl_type.currentText())
        self.bl_query.clear(); self._refresh_blacklist()
        self._append_log(f"🚨 Blacklist + {self.bl_type.currentText()}: {q}")

    def _remove_blacklist(self):
        idx = self.bl_list.currentRow()
        if idx < 0:
            return
        if 0 <= idx < len(self.blacklist_manager.blacklist):
            entry = self.blacklist_manager.blacklist[idx]
            self.blacklist_manager.remove_entry(idx)
            self._refresh_blacklist()
            self._append_log(f"Blacklist − {entry['query']}")

    def closeEvent(self, e):
        try:
            self.worker.stop()
            self.worker.wait(4000)
        except Exception:
            pass
        super().closeEvent(e)
