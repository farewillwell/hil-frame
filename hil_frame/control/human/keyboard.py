from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from hil_frame.common import monotonic_ns

from .base import HumanControlPhase, HumanControlSnapshot, HumanControllerBase

LOGGER = logging.getLogger(__name__)


KEY_TO_AXIS = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "d": (1, 1.0),
    "a": (1, -1.0),
    "q": (2, 1.0),
    "e": (2, -1.0),
    "i": (3, 1.0),
    "k": (3, -1.0),
    "l": (4, 1.0),
    "j": (4, -1.0),
    "u": (5, 1.0),
    "o": (5, -1.0),
    "x": (6, 1.0),
    "z": (6, -1.0),
}


@dataclass
class KeyboardScales:
    translation: float = 1.0
    rotation: float = 1.0
    gripper: float = 1.0
    deadzone: float = 0.0


class ManualHumanController(HumanControllerBase):
    """Thread-safe controller used by tests and non-keyboard integrations."""

    def __init__(self, action_dim: int = 7) -> None:
        self.action_dim = action_dim
        self._phase = HumanControlPhase.IDLE
        self._action = np.zeros(action_dim, dtype=np.float32)
        self._quit_requested = False
        self._lock = threading.RLock()

    def set_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != (self.action_dim,):
            raise ValueError(f"human action shape {arr.shape} does not match {(self.action_dim,)}")
        with self._lock:
            self._action = arr.copy()

    def request_quit(self) -> None:
        with self._lock:
            self._quit_requested = True

    def enter_control(self) -> None:
        with self._lock:
            self._phase = HumanControlPhase.ACTIVE

    def exit_control(self) -> None:
        with self._lock:
            self._phase = HumanControlPhase.IDLE

    def get_action(self) -> np.ndarray:
        with self._lock:
            return self._action.copy()

    def stop_and_reset(self) -> None:
        with self._lock:
            self._phase = HumanControlPhase.STOPPED
            self._action = np.zeros(self.action_dim, dtype=np.float32)

    def get_snapshot(self) -> HumanControlSnapshot:
        with self._lock:
            return HumanControlSnapshot(
                phase=self._phase,
                action=self._action.copy(),
                timestamp_ns=monotonic_ns(),
                quit_requested=self._quit_requested,
            )

    def acknowledge_reset(self) -> None:
        with self._lock:
            if self._phase == HumanControlPhase.STOPPED:
                self._phase = HumanControlPhase.IDLE
            self._action = np.zeros(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        return None


class KeyboardController(ManualHumanController):
    """pynput keyboard backend with explicit phase edge keys."""

    def __init__(self, action_dim: int = 7, scales: KeyboardScales | None = None) -> None:
        super().__init__(action_dim=action_dim)
        self.scales = scales or KeyboardScales()
        self._pressed: set[str] = set()
        self._listener = None
        self._edge_latch: set[str] = set()
        self._start_listener()

    def _start_listener(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            raise RuntimeError("pynput keyboard backend unavailable; provide a different HumanControllerBase.") from exc

        def name_key(key) -> str | None:
            try:
                return key.char.lower()
            except AttributeError:
                if key == keyboard.Key.enter:
                    return "enter"
                if key == keyboard.Key.backspace:
                    return "backspace"
                if key == keyboard.Key.f12:
                    return "f12"
                if key == keyboard.Key.esc:
                    return "esc"
            return None

        def on_press(key) -> None:
            name = name_key(key)
            if name is None:
                return
            with self._lock:
                self._pressed.add(name)
                if name in {"enter", "backspace", "f12", "esc"} and name not in self._edge_latch:
                    self._edge_latch.add(name)
                    if name == "enter":
                        self._phase = HumanControlPhase.ACTIVE
                    elif name == "backspace":
                        self._phase = HumanControlPhase.IDLE
                    elif name == "f12":
                        self._phase = HumanControlPhase.STOPPED
                        self._action = np.zeros(self.action_dim, dtype=np.float32)
                        self._pressed.clear()
                    elif name == "esc":
                        self._quit_requested = True
                self._action = self._action_from_pressed_locked()

        def on_release(key) -> None:
            name = name_key(key)
            if name is None:
                return
            with self._lock:
                self._pressed.discard(name)
                self._edge_latch.discard(name)
                self._action = self._action_from_pressed_locked()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def _action_from_pressed_locked(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        for key, (axis, sign) in KEY_TO_AXIS.items():
            if key in self._pressed:
                scale = self.scales.translation if axis < 3 else self.scales.rotation
                if axis == 6:
                    scale = self.scales.gripper
                action[axis] += sign * scale
        if self.scales.deadzone > 0:
            action[np.abs(action) < self.scales.deadzone] = 0.0
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()

