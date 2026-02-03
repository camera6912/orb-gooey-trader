"""Campfire notifier (copied from orb-trader).

This module only posts plaintext messages to a Campfire room bot endpoint.
"""

from __future__ import annotations

from loguru import logger
import requests


class CampfireNotifier:
    def __init__(self, base_url: str, room_id: str, bot_key: str):
        self.base_url = (base_url or "").rstrip("/")
        self.room_id = str(room_id or "").strip()
        self.bot_key = str(bot_key or "").strip()
        self.enabled = bool(self.base_url and self.room_id and self.bot_key)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/rooms/{self.room_id}/{self.bot_key}/messages"

    def send_message(self, message: str) -> bool:
        if not self.enabled:
            logger.debug("Campfire notifier disabled")
            return False

        try:
            resp = requests.post(
                self.endpoint,
                data=message.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=10,
            )
            ok = resp.status_code == 201
            if ok:
                logger.info("Campfire message sent")
            else:
                logger.error(f"Campfire failed: {resp.status_code} {resp.text}")
            return ok
        except Exception as e:
            logger.error(f"Campfire error: {e}")
            return False


def notifier_from_config(settings: dict | None, secrets: dict | None = None) -> CampfireNotifier:
    s = settings or {}
    camp = (s.get("campfire") or {}) if isinstance(s, dict) else {}

    base_url = str(camp.get("url") or "").strip()
    room_id = str(camp.get("room_id") or "").strip()

    sec = secrets or {}
    sec_camp = (sec.get("campfire") or {}) if isinstance(sec, dict) else {}
    bot_key = str(sec_camp.get("bot_key") or sec_camp.get("api_token") or "").strip()

    return CampfireNotifier(base_url=base_url, room_id=room_id, bot_key=bot_key)
