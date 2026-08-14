from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading

log = logging.getLogger(__name__)


@dataclass
class MqttConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    prefix: str = "tundranvr"
    camera: str = "cam"


class MqttBus:
    """Publish event/state/verdict. No-op when disabled or broker missing."""

    def __init__(self, cfg: MqttConfig) -> None:
        self.cfg = cfg
        self._client = None
        self._lock = threading.Lock()
        if cfg.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.warning("paho-mqtt not installed; MQTT publish disabled")
            return
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
        try:
            client.connect(self.cfg.host, int(self.cfg.port), keepalive=30)
            client.loop_start()
            self._client = client
            log.info("MQTT connected %s:%s", self.cfg.host, self.cfg.port)
        except Exception as exc:
            log.warning("MQTT broker unreachable: %s", exc)
            self._client = None

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
        self._client = None

    def publish(self, kind: str, payload: dict) -> None:
        topic = f"{self.cfg.prefix}/{self.cfg.camera}/{kind}"
        body = json.dumps(payload, default=str)
        with self._lock:
            if self._client is None:
                log.debug("MQTT skip %s %s", topic, body[:120])
                return
            try:
                self._client.publish(topic, body, qos=0)
            except Exception as exc:
                log.debug("MQTT publish failed: %s", exc)
