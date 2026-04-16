#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline text-to-speech service backed by Piper.
Runs synthesis and playback in a dedicated worker thread.
"""

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import hashlib
from pathlib import Path

from config import TTSConfig


class TTSService:
    """Queue-based Piper TTS service with graceful fallback behavior."""

    def __init__(self):
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._current_process = None
        self._speaking = False

        self.binary = shutil.which(TTSConfig.PIPER_BINARY) or TTSConfig.PIPER_BINARY
        self.playback_binary = shutil.which(TTSConfig.PLAYBACK_BINARY) or TTSConfig.PLAYBACK_BINARY
        self.model_path = Path(TTSConfig.MODEL_PATH)
        self.model_config_path = Path(TTSConfig.MODEL_CONFIG_PATH)
        self.cache_enabled = bool(TTSConfig.CACHE_ENABLED)
        self.cache_dir = Path(TTSConfig.CACHE_DIR)
        self.prewarm_cache_on_start = bool(TTSConfig.PREWARM_CACHE_ON_START)
        self.standard_cache_texts = {self._normalize_text(text) for text in TTSConfig.STANDARD_CACHE_TEXTS}

        self.enabled = bool(TTSConfig.ENABLED)
        self._validate_runtime()

        self._worker = threading.Thread(target=self._run, daemon=True, name="TTSService")
        self._worker.start()
        self._prewarm_thread = None
        if self.enabled and self.cache_enabled and self.prewarm_cache_on_start:
            self._prewarm_thread = threading.Thread(
                target=self.prewarm_cache,
                daemon=True,
                name="TTSCachePrewarm",
            )
            self._prewarm_thread.start()

    def _validate_runtime(self):
        if not self.enabled:
            return

        missing = []
        if not shutil.which(self.binary) and not Path(self.binary).exists():
            missing.append(f"piper binary not found: {self.binary}")
        if not self.model_path.exists():
            missing.append(f"model missing: {self.model_path}")
        if not shutil.which(self.playback_binary) and not Path(self.playback_binary).exists():
            missing.append(f"playback binary not found: {self.playback_binary}")

        if missing:
            self.enabled = False
            print(f"[TTS] Disabled ({'; '.join(missing)})")
            return

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str, interrupt: bool = False) -> bool:
        text = (text or "").strip()
        if not text or not self.enabled:
            return False

        if interrupt:
            self._clear_queue()
            self._terminate_current_process()

        self._queue.put(text)
        return True

    def is_speaking(self) -> bool:
        with self._state_lock:
            return self._speaking

    def stop(self):
        self._stop_event.set()
        self._clear_queue()
        self._terminate_current_process()
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        if self._prewarm_thread and self._prewarm_thread.is_alive():
            self._prewarm_thread.join(timeout=2.0)

    def _set_speaking(self, value: bool):
        with self._state_lock:
            self._speaking = value

    def _clear_queue(self):
        with self._queue.mutex:
            self._queue.queue.clear()

    def _set_current_process(self, process):
        with self._state_lock:
            self._current_process = process

    def _terminate_current_process(self):
        with self._state_lock:
            process = self._current_process

        if not process:
            return

        try:
            process.terminate()
            process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        finally:
            self._set_current_process(None)
            self._set_speaking(False)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                self._synthesize_and_play(text)
            except Exception as exc:
                print(f"[TTS] Playback failed: {exc}")
            finally:
                self._set_current_process(None)
                self._set_speaking(False)

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().split())

    def _should_cache_text(self, text: str) -> bool:
        if not self.cache_enabled:
            return False
        return self._normalize_text(text) in self.standard_cache_texts

    def _get_cache_path(self, text: str) -> Path:
        normalized = self._normalize_text(text)
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.wav"

    def _play_wav(self, wav_path: str):
        playback_cmd = [str(self.playback_binary), wav_path]
        playback_process = subprocess.Popen(
            playback_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_current_process(playback_process)
        playback_process.wait()

    def _synthesize_to_file(self, text: str, wav_path: str, track_process: bool = True) -> bool:
        piper_cmd = [str(self.binary), "--model", str(self.model_path), "--output_file", wav_path]
        if self.model_config_path.exists():
            piper_cmd.extend(["--config", str(self.model_config_path)])

        piper_process = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if track_process:
            self._set_current_process(piper_process)
        piper_process.communicate(input=text.encode("utf-8"))
        return piper_process.returncode in (0, None)

    def _ensure_cached_file(self, text: str):
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            return cache_path

        fd, temp_cache_path = tempfile.mkstemp(
            prefix="alarm_tts_cache_",
            suffix=".wav",
            dir=str(self.cache_dir),
        )
        os.close(fd)

        try:
            if not self._synthesize_to_file(text, temp_cache_path, track_process=False):
                return None
            os.replace(temp_cache_path, str(cache_path))
            return cache_path
        finally:
            if os.path.exists(temp_cache_path):
                try:
                    os.remove(temp_cache_path)
                except OSError:
                    pass

    def prewarm_cache(self):
        for text in sorted(self.standard_cache_texts):
            if self._stop_event.is_set():
                return
            try:
                self._ensure_cached_file(text)
            except Exception as exc:
                print(f"[TTS] Cache prewarm failed for '{text}': {exc}")

    def _synthesize_and_play(self, text: str):
        self._set_speaking(True)

        if self._should_cache_text(text):
            cache_path = self._ensure_cached_file(text)
            if cache_path:
                self._play_wav(str(cache_path))
                return

        fd, wav_path = tempfile.mkstemp(prefix="alarm_tts_", suffix=".wav")
        os.close(fd)

        try:
            if not self._synthesize_to_file(text, wav_path):
                return
            if self._stop_event.is_set():
                return
            self._play_wav(wav_path)
        finally:
            self._set_current_process(None)
            try:
                os.remove(wav_path)
            except OSError:
                pass
