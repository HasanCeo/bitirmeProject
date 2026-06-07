"""
ESP32-CAM MJPEG HTTP Stream Okuyucu

ESP32-CAM'den MJPEG stream okuyarak cv2.VideoCapture uyumlu arayüz sağlar.
Otomatik yeniden bağlanma (reconnect) mekanizması içerir.

Kullanım:
    stream = ESP32CamStream("http://192.168.1.100:81/stream")
    stream.open()
    
    while stream.isOpened():
        ret, frame = stream.read()
        if ret:
            cv2.imshow("ESP32-CAM", frame)
    
    stream.release()
"""

import cv2
import numpy as np
import logging
import time
import urllib.request
from threading import Thread, Lock, Event


class ESP32CamStream:
    """
    ESP32-CAM MJPEG stream okuyucu.
    
    cv2.VideoCapture ile aynı arayüzü sağlar:
        - open()      : Stream bağlantısını aç
        - read()      : (ret, frame) döner
        - isOpened()  : Bağlantı durumu
        - release()   : Bağlantıyı kapat
    """
    
    def __init__(self, stream_url, reconnect_delay=3, timeout=10):
        """
        Args:
            stream_url: ESP32-CAM MJPEG stream URL'i (ör: "http://192.168.1.100:81/stream")
            reconnect_delay: Bağlantı koparsa yeniden deneme aralığı (saniye)
            timeout: HTTP bağlantı zaman aşımı (saniye)
        """
        self.stream_url = stream_url
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout
        
        self._frame = None
        self._frame_lock = Lock()
        self._opened = False
        self._running = False
        self._stop_event = Event()
        self._thread = None
        self._stream = None
        self._connected = False
        self._last_frame_time = 0
        self._frame_count = 0
        self._reconnect_count = 0
        
        self.logger = logging.getLogger("ESP32CamStream")
    
    def open(self):
        """Stream bağlantısını başlat"""
        if self._running:
            self.logger.warning("Stream zaten çalışıyor")
            return True
        
        self._stop_event.clear()
        self._running = True
        self._opened = True
        
        # Arka plan thread'inde stream okuma başlat
        self._thread = Thread(target=self._read_stream_loop, daemon=True)
        self._thread.start()
        
        # İlk bağlantının kurulmasını bekle (max 5 saniye)
        wait_start = time.time()
        while not self._connected and time.time() - wait_start < 5:
            time.sleep(0.1)
        
        if self._connected:
            self.logger.info(f"ESP32-CAM bağlantısı kuruldu: {self.stream_url}")
        else:
            self.logger.warning(f"ESP32-CAM ilk bağlantı bekleniyor: {self.stream_url}")
        
        return self._opened
    
    def read(self):
        """
        Güncel frame'i oku.
        
        Returns:
            tuple: (ret, frame) - ret: frame geçerli mi, frame: numpy array (BGR)
        """
        with self._frame_lock:
            if self._frame is not None:
                return True, self._frame.copy()
        
        return False, None
    
    def isOpened(self):
        """Bağlantı açık mı"""
        return self._opened and self._running
    
    def release(self):
        """Bağlantıyı kapat ve kaynakları serbest bırak"""
        self.logger.info("ESP32-CAM stream kapatılıyor...")
        self._running = False
        self._opened = False
        self._stop_event.set()
        
        # Stream bağlantısını kapat
        self._close_stream()
        
        # Thread'in bitmesini bekle
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        
        self._connected = False
        self.logger.info("ESP32-CAM stream kapatıldı")
    
    def get_status(self):
        """Stream durum bilgisi döner"""
        return {
            "connected": self._connected,
            "url": self.stream_url,
            "frame_count": self._frame_count,
            "reconnect_count": self._reconnect_count,
            "last_frame_time": self._last_frame_time,
        }
    
    # ==================== İÇ METOTLAR ====================
    
    def _read_stream_loop(self):
        """Arka plan thread'i: sürekli stream oku, bağlantı koparsa yeniden bağlan"""
        while self._running and not self._stop_event.is_set():
            try:
                self._connect_stream()
                
                if not self._stream:
                    self.logger.warning(
                        f"ESP32-CAM bağlantı kurulamadı, {self.reconnect_delay}s sonra tekrar denenecek..."
                    )
                    self._connected = False
                    self._stop_event.wait(timeout=self.reconnect_delay)
                    self._reconnect_count += 1
                    continue
                
                self._connected = True
                self.logger.info("ESP32-CAM stream okuma başladı")
                
                # MJPEG stream'inden frame'leri oku
                bytes_buffer = b''
                while self._running and not self._stop_event.is_set():
                    chunk = self._stream.read(4096)
                    if not chunk:
                        self.logger.warning("ESP32-CAM stream'den veri alınamadı")
                        break
                    
                    bytes_buffer += chunk
                    
                    # JPEG frame sınırlarını bul
                    while True:
                        # JPEG başlangıcını bul (SOI marker: 0xFF 0xD8)
                        start = bytes_buffer.find(b'\xff\xd8')
                        if start == -1:
                            # Başlangıç bulunamadı, buffer'ı temizle (son 2 byte hariç)
                            if len(bytes_buffer) > 2:
                                bytes_buffer = bytes_buffer[-2:]
                            break
                        
                        # JPEG bitişini bul (EOI marker: 0xFF 0xD9)
                        end = bytes_buffer.find(b'\xff\xd9', start + 2)
                        if end == -1:
                            # Henüz tam frame yok, daha fazla veri bekle
                            # Başlangıçtan önceki veriyi temizle
                            if start > 0:
                                bytes_buffer = bytes_buffer[start:]
                            break
                        
                        # Tam JPEG frame bulundu
                        jpg_data = bytes_buffer[start:end + 2]
                        bytes_buffer = bytes_buffer[end + 2:]
                        
                        # JPEG'i decode et
                        frame = self._decode_jpeg(jpg_data)
                        if frame is not None:
                            with self._frame_lock:
                                self._frame = frame
                            self._frame_count += 1
                            self._last_frame_time = time.time()
                
            except Exception as e:
                self.logger.error(f"ESP32-CAM stream hatası: {e}")
                self._connected = False
                self._close_stream()
                
                if self._running:
                    self.logger.info(
                        f"Yeniden bağlanma denemesi ({self._reconnect_count + 1})..."
                    )
                    self._stop_event.wait(timeout=self.reconnect_delay)
                    self._reconnect_count += 1
        
        self._connected = False
        self._close_stream()
    
    def _connect_stream(self):
        """HTTP bağlantısı kur"""
        self._close_stream()
        
        try:
            self.logger.info(f"ESP32-CAM'e bağlanılıyor: {self.stream_url}")
            req = urllib.request.Request(self.stream_url)
            self._stream = urllib.request.urlopen(req, timeout=self.timeout)
            self.logger.info("HTTP bağlantısı kuruldu")
        except Exception as e:
            self.logger.error(f"ESP32-CAM bağlantı hatası: {e}")
            self._stream = None
    
    def _close_stream(self):
        """HTTP bağlantısını kapat"""
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
    
    def _decode_jpeg(self, jpg_data):
        """
        JPEG byte verisini numpy array'e dönüştür.
        
        Args:
            jpg_data: JPEG formatında byte verisi
            
        Returns:
            numpy array (BGR formatında) veya None
        """
        try:
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            self.logger.error(f"JPEG decode hatası: {e}")
            return None


def test_connection(stream_url, timeout=5):
    """
    ESP32-CAM bağlantısını test et.
    
    Args:
        stream_url: Stream URL'i
        timeout: Zaman aşımı (saniye)
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Status URL'ini oluştur (stream URL'inden)
        # http://IP:81/stream -> http://IP/status
        base_url = stream_url.rsplit(":", 1)[0]  # Port'u çıkar
        # Eğer port varsa IP kısmını al
        if "//" in base_url:
            protocol_and_ip = base_url
        else:
            protocol_and_ip = base_url
            
        # IP'yi çıkar
        parts = stream_url.split("//")
        if len(parts) > 1:
            host_part = parts[1].split("/")[0]  # IP:PORT
            ip = host_part.split(":")[0]         # sadece IP
            status_url = f"http://{ip}/status"
        else:
            status_url = stream_url.replace("/stream", "/status")
        
        req = urllib.request.Request(status_url)
        response = urllib.request.urlopen(req, timeout=timeout)
        data = response.read().decode('utf-8')
        response.close()
        
        return True, f"ESP32-CAM bağlantısı başarılı: {data}"
    except urllib.error.URLError as e:
        return False, f"ESP32-CAM bağlantı hatası: {e.reason}"
    except Exception as e:
        return False, f"ESP32-CAM test hatası: {e}"
