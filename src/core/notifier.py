"""Telegram tabanlı anlık bildirim yöneticisi.

Blacklist eşleşmesi olduğunda kullanıcının telefonuna (Telegram üzerinden)
metin + yakalanan fotoğraf gönderir. Ağ isteği video akışını kilitlemesin diye
gönderim ayrı bir thread'de yapılır.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import requests

from src.config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ENABLED,
    TELEGRAM_TIMEOUT,
)


class TelegramNotifier:
    """Telegram Bot API ile bildirim gönderir."""

    def __init__(self):
        self.enabled = TELEGRAM_ENABLED
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.timeout = TELEGRAM_TIMEOUT
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        if self.enabled:
            logging.info("Telegram bildirimleri aktif")
        else:
            logging.info(
                "Telegram bildirimleri pasif "
                "(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ayarlanmamış)"
            )

    def send_alert(self, message: str, photo_path: Optional[str] = None):
        """Bildirimi arka planda (non-blocking) gönderir.

        Args:
            message: Gönderilecek metin (caption olarak da kullanılır).
            photo_path: Varsa gönderilecek fotoğrafın yolu.
        """
        if not self.enabled:
            return

        thread = threading.Thread(
            target=self._send,
            args=(message, photo_path),
            daemon=True,
        )
        thread.start()

    def _send(self, message: str, photo_path: Optional[str]):
        """Asıl gönderim (thread içinde çalışır)."""
        try:
            if photo_path and Path(photo_path).exists():
                self._send_photo(message, photo_path)
            else:
                self._send_message(message)
        except Exception as e:
            # Bildirim hatası uygulamanın akışını bozmamalı.
            logging.error(f"Telegram bildirimi gönderilemedi: {e}")

    def _send_message(self, message: str):
        """Sadece metin gönderir."""
        response = requests.post(
            f"{self.base_url}/sendMessage",
            data={"chat_id": self.chat_id, "text": message},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def _send_photo(self, caption: str, photo_path: str):
        """Fotoğraf + caption gönderir."""
        with open(photo_path, "rb") as photo:
            response = requests.post(
                f"{self.base_url}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": photo},
                timeout=self.timeout,
            )
        response.raise_for_status()
