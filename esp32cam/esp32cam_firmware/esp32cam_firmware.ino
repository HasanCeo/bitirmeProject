/*
 * ESP32-CAM Güvenlik Kamerası Firmware
 * 
 * Bu firmware ESP32-CAM (AI-Thinker) modülü için MJPEG HTTP stream sunucusu sağlar.
 * Python güvenlik kamerası yazılımı bu stream'i okuyarak görüntü işleme yapar.
 * 
 * Endpoints:
 *   /stream   - MJPEG video akışı (sürekli)
 *   /capture  - Tek JPEG frame yakalama
 *   /status   - Kamera durumu (JSON)
 *   /led      - LED flash aç/kapa (?state=on veya ?state=off)
 * 
 * Kurulum:
 *   1. Arduino IDE'de ESP32 board paketini kurun
 *   2. Board: "AI Thinker ESP32-CAM" seçin
 *   3. Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)" seçin
 *   4. Aşağıdaki WIFI_SSID ve WIFI_PASSWORD değerlerini kendi ağ bilgilerinizle değiştirin
 *   5. Upload Speed: 115200
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "esp_timer.h"

// ==================== KULLANICI AYARLARI ====================
// WiFi bilgilerinizi buraya girin
const char* WIFI_SSID     = "hasanbakiiseri";
const char* WIFI_PASSWORD = "asasasas.";

// Kamera çözünürlüğü ayarı
// FRAMESIZE_QVGA   (320x240)  - Düşük çözünürlük, hızlı
// FRAMESIZE_VGA    (640x480)  - Orta çözünürlük (önerilen)
// FRAMESIZE_SVGA   (800x600)  - Yüksek çözünürlük
// FRAMESIZE_XGA    (1024x768) - Çok yüksek, yavaş
#define CAMERA_RESOLUTION FRAMESIZE_VGA

// JPEG sıkıştırma kalitesi (10-63, düşük = daha iyi kalite ama daha büyük dosya)
#define JPEG_QUALITY 12

// Stream FPS sınırı (ms cinsinden frame arası bekleme)
#define FRAME_DELAY_MS 33  // ~30 FPS

// ==================== AI-THINKER ESP32-CAM PIN TANIMLAMALARI ====================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// LED Flash pin (AI-Thinker board)
#define LED_GPIO_NUM       4

// ==================== GLOBAL DEĞİŞKENLER ====================
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// Stream istatistikleri
unsigned long frame_count = 0;
unsigned long last_frame_time = 0;
float current_fps = 0.0;

// ==================== MULTIPART STREAM TANIMLARI ====================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %lu\r\n\r\n";

// ==================== KAMERA BAŞLATMA ====================
bool initCamera() {
    camera_config_t config;

config.ledc_channel = LEDC_CHANNEL_0;

config.ledc_timer = LEDC_TIMER_0;

config.pin_d0 = Y2_GPIO_NUM;

config.pin_d1 = Y3_GPIO_NUM;

config.pin_d2 = Y4_GPIO_NUM;

config.pin_d3 = Y5_GPIO_NUM;

config.pin_d4 = Y6_GPIO_NUM;

config.pin_d5 = Y7_GPIO_NUM;

config.pin_d6 = Y8_GPIO_NUM;

config.pin_d7 = Y9_GPIO_NUM;

config.pin_xclk = XCLK_GPIO_NUM;

config.pin_pclk = PCLK_GPIO_NUM;

config.pin_vsync = VSYNC_GPIO_NUM;

config.pin_href = HREF_GPIO_NUM;

config.pin_sccb_sda = SIOD_GPIO_NUM;

config.pin_sccb_scl = SIOC_GPIO_NUM;

config.pin_pwdn = PWDN_GPIO_NUM;

config.pin_reset = RESET_GPIO_NUM;

//config.xclk_freq_hz = 20000000; high

config.xclk_freq_hz = 10000000; // Try reducing the clock frequency to reduce frame rate

config.pixel_format = PIXFORMAT_RGB565;

config.frame_size = FRAMESIZE_QQVGA; // Lower resolution to reduce lag

//config.frame_size = FRAMESIZE_QVGA; //this is lower than VGA but higher than QQVGA

//config.frame_size = FRAMESIZE_VGA; //higher frame rate //buffers a little

config.jpeg_quality = 10;

config.fb_count = 2;

config.grab_mode = CAMERA_GRAB_LATEST;

config.fb_location = CAMERA_FB_IN_PSRAM;

    // PSRAM varsa yüksek çözünürlük kullan
    if (psramFound()) {
        config.frame_size = CAMERA_RESOLUTION;
        config.fb_count   = 2;
        Serial.println("PSRAM bulundu - çift frame buffer aktif");
    } else {
        config.frame_size = FRAMESIZE_QVGA;
        config.fb_count   = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
        Serial.println("PSRAM bulunamadı - düşük çözünürlük modunda");
    }

    // Kamerayı başlat
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Kamera başlatma hatası: 0x%x\n", err);
        return false;
    }

    // Kamera sensör ayarları
    sensor_t* s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_brightness(s, 0);     // -2 ~ 2
        s->set_contrast(s, 0);       // -2 ~ 2
        s->set_saturation(s, 0);     // -2 ~ 2
        s->set_whitebal(s, 1);       // 0 = kapalı, 1 = açık
        s->set_awb_gain(s, 1);       // 0 = kapalı, 1 = açık
        s->set_wb_mode(s, 0);        // 0 ~ 4
        s->set_exposure_ctrl(s, 1);  // 0 = kapalı, 1 = açık
        s->set_aec2(s, 0);           // AEC DSP
        s->set_gain_ctrl(s, 1);      // AGC
        s->set_gainceiling(s, (gainceiling_t)6);  // 0 ~ 6
        s->set_bpc(s, 0);            // Black pixel correction
        s->set_wpc(s, 1);            // White pixel correction
        s->set_raw_gma(s, 1);        // Gamma correction
        s->set_lenc(s, 1);           // Lens correction
        s->set_hmirror(s, 0);        // Yatay ayna
        s->set_vflip(s, 0);          // Dikey çevirme
        s->set_dcw(s, 1);            // Downsize EN
    }

    Serial.println("Kamera başarıyla başlatıldı");
    return true;
}

// ==================== HTTP HANDLER: MJPEG STREAM ====================
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[128];

    Serial.println("Stream istemcisi bağlandı");

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    // CORS header
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "30");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Frame yakalanamadı");
            res = ESP_FAIL;
            break;
        }

        // FPS hesaplama
        unsigned long now = millis();
        if (last_frame_time > 0) {
            float delta = (now - last_frame_time) / 1000.0;
            if (delta > 0) {
                current_fps = 1.0 / delta;
            }
        }
        last_frame_time = now;
        frame_count++;

        // Boundary gönder
        size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len, now);
        
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res != ESP_OK) {
            esp_camera_fb_return(fb);
            break;
        }

        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res != ESP_OK) {
            esp_camera_fb_return(fb);
            break;
        }

        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        if (res != ESP_OK) {
            esp_camera_fb_return(fb);
            break;
        }

        esp_camera_fb_return(fb);

        // Frame rate sınırlama
        delay(FRAME_DELAY_MS);
    }

    Serial.println("Stream istemcisi ayrıldı");
    return res;
}

// ==================== HTTP HANDLER: TEK FRAME YAKALAMA ====================
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;

    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Frame yakalanamadı");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);

    return res;
}

// ==================== HTTP HANDLER: DURUM BİLGİSİ ====================
static esp_err_t status_handler(httpd_req_t *req) {
    char json_response[256];

    sensor_t *s = esp_camera_sensor_get();
    
    snprintf(json_response, sizeof(json_response),
        "{"
        "\"status\":\"ok\","
        "\"framesize\":%d,"
        "\"quality\":%d,"
        "\"brightness\":%d,"
        "\"contrast\":%d,"
        "\"fps\":%.1f,"
        "\"frame_count\":%lu,"
        "\"free_heap\":%u,"
        "\"psram_free\":%u"
        "}",
        s->status.framesize,
        s->status.quality,
        s->status.brightness,
        s->status.contrast,
        current_fps,
        frame_count,
        ESP.getFreeHeap(),
        ESP.getFreePsram()
    );

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

// ==================== HTTP HANDLER: LED KONTROLÜ ====================
static esp_err_t led_handler(httpd_req_t *req) {
    char buf[32];
    int buf_len = httpd_req_get_url_query_len(req) + 1;
    
    bool led_on = false;
    
    if (buf_len > 1 && buf_len <= sizeof(buf)) {
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            char param[16];
            if (httpd_query_key_value(buf, "state", param, sizeof(param)) == ESP_OK) {
                if (strcmp(param, "on") == 0) {
                    led_on = true;
                }
            }
        }
    }

    digitalWrite(LED_GPIO_NUM, led_on ? HIGH : LOW);

    char response[64];
    snprintf(response, sizeof(response), "{\"led\":\"%s\"}", led_on ? "on" : "off");
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, strlen(response));
}

// ==================== HTTP SUNUCU BAŞLATMA ====================
void startHTTPServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 7;
    config.max_uri_handlers = 8;

    // Ana sunucu (capture, status, led)
    Serial.printf("HTTP sunucu port %d'de başlatılıyor...\n", config.server_port);
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        // /capture endpoint
        httpd_uri_t capture_uri = {
            .uri       = "/capture",
            .method    = HTTP_GET,
            .handler   = capture_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(camera_httpd, &capture_uri);

        // /status endpoint
        httpd_uri_t status_uri = {
            .uri       = "/status",
            .method    = HTTP_GET,
            .handler   = status_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(camera_httpd, &status_uri);

        // /led endpoint
        httpd_uri_t led_uri = {
            .uri       = "/led",
            .method    = HTTP_GET,
            .handler   = led_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(camera_httpd, &led_uri);

        Serial.println("HTTP sunucu başlatıldı");
    }

    // Stream sunucusu (ayrı port: 81)
    config.server_port = 81;
    config.ctrl_port = 32769;
    
    Serial.printf("Stream sunucu port %d'de başlatılıyor...\n", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_uri_t stream_uri = {
            .uri       = "/stream",
            .method    = HTTP_GET,
            .handler   = stream_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(stream_httpd, &stream_uri);

        Serial.println("Stream sunucu başlatıldı");
    }
}

// ==================== SETUP ====================
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();
    Serial.println("==========================================");
    Serial.println("  ESP32-CAM Güvenlik Kamerası Başlatılıyor");
    Serial.println("==========================================");

    // LED pin ayarı
    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);

    // Kamerayı başlat
    if (!initCamera()) {
        Serial.println("HATA: Kamera başlatılamadı! Yeniden başlatılıyor...");
        delay(1000);
        ESP.restart();
    }

    // WiFi bağlantısı
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    WiFi.setSleep(false);  // WiFi uyku modunu kapat (düşük gecikme için)

    Serial.printf("WiFi'ye bağlanılıyor: %s", WIFI_SSID);
    
    int retry_count = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        retry_count++;
        
        if (retry_count > 40) {  // 20 saniye sonra timeout
            Serial.println("\nWiFi bağlantı zaman aşımı! Yeniden başlatılıyor...");
            ESP.restart();
        }
    }

    Serial.println();
    Serial.println("==========================================");
    Serial.println("  WiFi Bağlantısı Başarılı!");
    Serial.println("==========================================");
    Serial.printf("  IP Adresi  : %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Sinyal Gücü: %d dBm\n", WiFi.RSSI());
    Serial.println("------------------------------------------");
    Serial.printf("  Stream URL : http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Capture URL: http://%s/capture\n", WiFi.localIP().toString().c_str());
    Serial.printf("  Status URL : http://%s/status\n", WiFi.localIP().toString().c_str());
    Serial.printf("  LED URL    : http://%s/led?state=on\n", WiFi.localIP().toString().c_str());
    Serial.println("==========================================");

    // HTTP sunucuyu başlat
    startHTTPServer();

    // Başarılı başlatma göstergesi: LED 3 kez yanıp sönsün
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_GPIO_NUM, HIGH);
        delay(200);
        digitalWrite(LED_GPIO_NUM, LOW);
        delay(200);
    }
}

// ==================== LOOP ====================
void loop() {
    // WiFi bağlantı kontrolü
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi bağlantısı koptu! Yeniden bağlanılıyor...");
        
        // LED hızlı yanıp sönsün (hata göstergesi)
        for (int i = 0; i < 5; i++) {
            digitalWrite(LED_GPIO_NUM, HIGH);
            delay(100);
            digitalWrite(LED_GPIO_NUM, LOW);
            delay(100);
        }

        WiFi.reconnect();
        
        int retry = 0;
        while (WiFi.status() != WL_CONNECTED && retry < 20) {
            delay(500);
            retry++;
        }

        if (WiFi.status() == WL_CONNECTED) {
            Serial.printf("WiFi yeniden bağlandı! IP: %s\n", WiFi.localIP().toString().c_str());
        } else {
            Serial.println("WiFi yeniden bağlanılamadı. Restart...");
            ESP.restart();
        }
    }

    // Her 30 saniyede bir durum bilgisi
    static unsigned long last_status_time = 0;
    if (millis() - last_status_time > 30000) {
        last_status_time = millis();
        Serial.printf("[Durum] FPS: %.1f | Frame: %lu | Heap: %u | PSRAM: %u\n",
                      current_fps, frame_count, ESP.getFreeHeap(), ESP.getFreePsram());
    }

    delay(1000);
}
