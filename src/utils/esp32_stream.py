"""
ESP32-CAM MJPEG HTTP Stream Okuyucu

ESP32-CAM'den MJPEG stream okuyarak cv2.VideoCapture uyumlu arayüz sağlar.
Hem JPEG hem de ham RGB565 piksel formatını destekler.
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
import requests
from threading import Thread, Lock, Event


class ESP32CamStream:
    """
    ESP32-CAM MJPEG/raw stream okuyucu.
    
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
        self._response = None
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
        
        # İlk bağlantının kurulmasını bekle (max 8 saniye)
        wait_start = time.time()
        while not self._connected and time.time() - wait_start < 8:
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
                
                if not self._response:
                    self.logger.warning(
                        f"ESP32-CAM bağlantı kurulamadı, {self.reconnect_delay}s sonra tekrar denenecek..."
                    )
                    self._connected = False
                    self._stop_event.wait(timeout=self.reconnect_delay)
                    self._reconnect_count += 1
                    continue
                
                self._connected = True
                content_type = self._response.headers.get('Content-Type', '')
                self.logger.info(f"ESP32-CAM Content-Type: {content_type}")
                self.logger.info("ESP32-CAM stream okuma başladı")
                
                # Boundary'yi Content-Type'dan al
                boundary = None
                if 'boundary=' in content_type:
                    boundary = content_type.split('boundary=')[1].strip()
                    self.logger.info(f"Multipart boundary: {boundary}")
                
                if boundary:
                    self._read_multipart_stream(boundary)
                else:
                    self.logger.error("Boundary bulunamadı, stream okunamıyor")
                
                # Stream sona erdi
                if self._running:
                    self.logger.warning("ESP32-CAM stream sona erdi, yeniden bağlanılacak...")
                
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"ESP32-CAM bağlantı hatası: {e}")
                self._connected = False
                self._close_stream()
                
                if self._running:
                    self.logger.info(
                        f"Yeniden bağlanma denemesi ({self._reconnect_count + 1})..."
                    )
                    self._stop_event.wait(timeout=self.reconnect_delay)
                    self._reconnect_count += 1
                    
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
    
    def _read_multipart_stream(self, boundary):
        """
        Multipart stream'i oku ve frame'leri parse et.
        Hem JPEG hem de raw RGB565 formatını destekler.
        
        State machine:
        1. FIND_BOUNDARY: Boundary'yi bul
        2. PARSE_HEADER: Header'ları parse et, content_length al
        3. READ_DATA: content_length kadar veri topla
        4. DECODE: Frame'i decode et
        """
        boundary_bytes = ('--' + boundary).encode()
        raw = self._response.raw
        
        buffer = b''
        READ_SIZE = 32768  # 32KB per read for faster accumulation
        decode_fail_count = 0
        
        # State machine değişkenleri
        state = 'FIND_BOUNDARY'
        content_length = 0
        content_type_part = 'image/jpeg'
        data_start = 0  # Buffer içinde frame verisinin başlangıcı
        
        while self._running and not self._stop_event.is_set():
            try:
                chunk = raw.read(READ_SIZE)
            except Exception as e:
                self.logger.error(f"Stream read hatası: {e}")
                break
            
            if not chunk:
                self.logger.warning("Stream'den veri alınamadı (boş chunk)")
                break
            
            buffer += chunk
            
            # State machine döngüsü
            while True:
                if state == 'FIND_BOUNDARY':
                    boundary_pos = buffer.find(boundary_bytes)
                    if boundary_pos == -1:
                        # Boundary yok, son kısmı sakla (boundary parçalanmış olabilir)
                        keep = len(boundary_bytes) + 10
                        if len(buffer) > keep:
                            buffer = buffer[-keep:]
                        break  # Daha fazla veri oku
                    
                    # Boundary bulundu, öncesini temizle
                    buffer = buffer[boundary_pos:]
                    state = 'PARSE_HEADER'
                
                elif state == 'PARSE_HEADER':
                    # Boundary'den sonraki header'ları bul
                    header_start = len(boundary_bytes)
                    header_end = buffer.find(b'\r\n\r\n', header_start)
                    if header_end == -1:
                        break  # Header henüz tam gelmemiş, daha fazla veri oku
                    
                    # Header'ları parse et
                    header_data = buffer[header_start:header_end].decode('ascii', errors='ignore')
                    data_start = header_end + 4  # \r\n\r\n sonrası
                    
                    content_length = None
                    content_type_part = 'image/jpeg'
                    for line in header_data.split('\r\n'):
                        line = line.strip()
                        if line.lower().startswith('content-length:'):
                            try:
                                content_length = int(line.split(':')[1].strip())
                            except ValueError:
                                pass
                        elif line.lower().startswith('content-type:'):
                            content_type_part = line.split(':')[1].strip()
                    
                    if content_length is None:
                        # Content-Length yoksa, boundary ararak boyut tahmin et
                        self.logger.warning("Content-Length bulunamadı, boundary ile boyut tahmin ediliyor")
                        state = 'FIND_BOUNDARY'
                        buffer = buffer[data_start:]  # Header sonrasından devam
                        continue
                    
                    state = 'READ_DATA'
                
                elif state == 'READ_DATA':
                    # Yeterli veri toplandı mı?
                    available = len(buffer) - data_start
                    if available < content_length:
                        # Henüz yeterli veri yok, daha fazla oku
                        # Buffer'ı KESMEDEN devam et!
                        break
                    
                    # Tam frame verisi toplandı!
                    frame_data = buffer[data_start:data_start + content_length]
                    
                    # Buffer'dan işlenmiş kısmı çıkar
                    buffer = buffer[data_start + content_length:]
                    
                    # Frame'i decode et
                    frame = self._decode_frame(frame_data, content_length, content_type_part)
                    
                    if frame is not None:
                        with self._frame_lock:
                            self._frame = frame
                        self._frame_count += 1
                        self._last_frame_time = time.time()
                        
                        # İlk frame geldiğinde özel log
                        if self._frame_count == 1:
                            self.logger.info(
                                f"ESP32-CAM ilk frame alındı! "
                                f"(boyut: {frame.shape[1]}x{frame.shape[0]}, "
                                f"content_length: {content_length}, "
                                f"format: {content_type_part})"
                            )
                        # Her 30 frame'de durum bilgisi
                        elif self._frame_count % 30 == 0:
                            self.logger.info(
                                f"ESP32-CAM frame #{self._frame_count} alındı "
                                f"(decode fail: {decode_fail_count})"
                            )
                    else:
                        decode_fail_count += 1
                        if decode_fail_count % 10 == 0:
                            self.logger.warning(
                                f"Frame decode başarısız: {decode_fail_count} kez "
                                f"(data boyutu: {content_length} bytes, "
                                f"format: {content_type_part})"
                            )
                    
                    # Sonraki frame için boundary ara
                    state = 'FIND_BOUNDARY'
                    continue  # Hemen sonraki boundary'yi ara
    
    def _decode_frame(self, frame_data, content_length, content_type):
        """
        Frame verisini decode et. Hem JPEG hem de ham RGB565 formatını destekler.
        
        Args:
            frame_data: Ham frame verisi (bytes)
            content_length: Veri boyutu
            content_type: İçerik türü (image/jpeg vb.)
            
        Returns:
            numpy array (BGR formatında) veya None
        """
        try:
            # Önce JPEG olarak dene
            if frame_data[:2] == b'\xff\xd8':
                return self._decode_jpeg(frame_data)
            
            # RGB565 formatı kontrolü: 640x480x2 = 614400, 320x240x2 = 153600, vb.
            known_resolutions = [
                (640, 480),   # VGA
                (320, 240),   # QVGA
                (800, 600),   # SVGA
                (1024, 768),  # XGA
                (1280, 720),  # HD
                (1280, 1024), # SXGA
                (1600, 1200), # UXGA
                (160, 120),   # QQVGA
                (176, 144),   # QCIF
                (352, 288),   # CIF
                (400, 296),   # HQVGA
            ]
            
            for w, h in known_resolutions:
                if content_length == w * h * 2:  # RGB565 = 2 bytes per pixel
                    return self._decode_rgb565(frame_data, w, h)
            
            # Bilinmeyen boyut, yine de JPEG olarak dene
            result = self._decode_jpeg(frame_data)
            if result is not None:
                return result
            
            # Son çare: RGB565 olarak en yakın çözünürlüğü tahmin et
            for w, h in known_resolutions:
                expected = w * h * 2
                if abs(content_length - expected) < 100:
                    self.logger.info(f"Yakın RGB565 boyutu tahmin edildi: {w}x{h}")
                    return self._decode_rgb565(frame_data[:expected], w, h)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Frame decode hatası: {e}")
            return None
    
    def _decode_jpeg(self, jpg_data):
        """JPEG byte verisini numpy array'e dönüştür."""
        try:
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            self.logger.error(f"JPEG decode hatası: {e}")
            return None
    
    def _decode_rgb565(self, raw_data, width, height):
        """
        RGB565 formatındaki ham veriyi BGR numpy array'e dönüştür.
        
        ESP32-CAM RGB565 format: Her piksel 2 byte, big-endian byte sırası.
        Bit düzeni (16 bit): RRRRR GGGGGG BBBBB
        """
        try:
            # RGB565 verisini big-endian uint16 array'e dönüştür
            # ESP32 big-endian gönderir, numpy default little-endian okur
            pixels = np.frombuffer(raw_data, dtype='>u2').reshape((height, width))
            
            # RGB565'ten RGB888'e dönüştür
            # uint16 olarak tut (overflow önleme)
            r = (pixels >> 11) & 0x1F   # 5 bit, 0-31 arası
            g = (pixels >> 5) & 0x3F    # 6 bit, 0-63 arası
            b = pixels & 0x1F           # 5 bit, 0-31 arası
            
            # 5/6 bit'i 8 bit'e ölçekle
            r = (r * 255 // 31).astype(np.uint8)
            g = (g * 255 // 63).astype(np.uint8)
            b = (b * 255 // 31).astype(np.uint8)
            
            # BGR formatında birleştir (OpenCV formatı)
            frame = np.stack([b, g, r], axis=2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"RGB565 decode hatası: {e}")
            return None
    
    def _connect_stream(self):
        """HTTP bağlantısı kur (requests ile streaming)"""
        self._close_stream()
        
        try:
            self.logger.info(f"ESP32-CAM'e bağlanılıyor: {self.stream_url}")
            self._response = requests.get(
                self.stream_url,
                stream=True,
                timeout=(self.timeout, 30),
                headers={'Connection': 'keep-alive'}
            )
            self._response.raise_for_status()
            self.logger.info(f"HTTP bağlantısı kuruldu (status: {self._response.status_code})")
        except Exception as e:
            self.logger.error(f"ESP32-CAM bağlantı hatası: {e}")
            self._response = None
    
    def _close_stream(self):
        """HTTP bağlantısını kapat"""
        if self._response:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None


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
        parts = stream_url.split("//")
        if len(parts) > 1:
            host_part = parts[1].split("/")[0]
            ip = host_part.split(":")[0]
            status_url = f"http://{ip}/status"
        else:
            status_url = stream_url.replace("/stream", "/status")
        
        response = requests.get(status_url, timeout=timeout)
        response.raise_for_status()
        data = response.text
        
        return True, f"ESP32-CAM bağlantısı başarılı: {data}"
    except requests.exceptions.ConnectionError as e:
        return False, f"ESP32-CAM bağlantı hatası: {e}"
    except Exception as e:
        return False, f"ESP32-CAM test hatası: {e}"
