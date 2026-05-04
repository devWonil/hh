#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

"""Background thread that converts text to speech via EdgeTTS and plays it."""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import edge_tts
import pygame

import config
import logging

logger = logging.getLogger(__name__)


@dataclass
class TTSRequest:
	"""Container for queued TTS requests."""

	text: str
	rate: str
	pitch: str
	voice: str
	volume: float
	wait_event: Optional[threading.Event] = None


class TTSThread(threading.Thread):
	"""Thread that handles EdgeTTS synthesis, caching, and playback."""

	def __init__(
		self,
		*,
		default_voice: Optional[str] = None,
		default_rate: Optional[str] = None,
		default_pitch: Optional[str] = None,
		default_volume: Optional[float] = None,
		cache_dir: Optional[str | Path] = None,
	) -> None:
		super().__init__(name="EdgeTTS", daemon=True)
		self._queue: queue.Queue[Optional[TTSRequest]] = queue.Queue()
		self._stop_event = threading.Event()

		self.default_voice = default_voice or getattr(config, "EDGE_TTS_VOICE", "ko-KR-SunHiNeural")
		self.default_rate = default_rate or getattr(config, "EDGE_TTS_RATE", "+0%")
		self.default_pitch = default_pitch or getattr(config, "EDGE_TTS_PITCH", "+0Hz")
		self.default_volume = (
			self._clamp_volume(default_volume)
			if default_volume is not None
			else self._clamp_volume(getattr(config, "EDGE_TTS_VOLUME", 0.8))
		)
		cache_root = cache_dir or getattr(config, "EDGE_TTS_CACHE_DIR", ".cache/tts")
		self.cache_dir = Path(cache_root)
		self.cache_dir.mkdir(parents=True, exist_ok=True)

		self._current_channel: Optional[pygame.mixer.Channel] = None
		self._current_sound: Optional[pygame.mixer.Sound] = None
		self._play_lock = threading.Lock()

	# ============================ Public API ============================
	def speak(
		self,
		text: str,
		*,
		rate: Optional[str] = None,
		pitch: Optional[str] = None,
		voice: Optional[str] = None,
		volume: Optional[float] = None,
		wait: bool = False,
	) -> Optional[threading.Event]:
		"""Queue a text for synthesis and playback."""

		normalized_text = (text or "").strip()
		if not normalized_text:
			logger.warning("TTS request skipped: empty text")
			return None

		req = TTSRequest(
			text=normalized_text,
			rate=rate or self.default_rate,
			pitch=pitch or self.default_pitch,
			voice=voice or self.default_voice,
			volume=self._clamp_volume(volume if volume is not None else self.default_volume),
		)

		if wait:
			req.wait_event = threading.Event()

		self._queue.put(req)

		if wait:
			req.wait_event.wait()
			return req.wait_event

		return None

	def stop(self) -> None:
		"""Stop playback and terminate the worker thread."""

		self._stop_event.set()
		self.stop_playback()
		self._queue.put(None)

	def stop_playback(self) -> None:
		"""Stop the currently playing sound, if any."""

		with self._play_lock:
			if self._current_channel is not None:
				self._current_channel.stop()
			self._current_channel = None
			self._current_sound = None

	def get_cache_path(self, text: str, *, rate: Optional[str] = None, pitch: Optional[str] = None) -> Path:
		"""Return the cache path for the provided text/rate/pitch combination."""

		return self._build_cache_path(text, rate or self.default_rate, pitch or self.default_pitch)

	# ============================ Thread loop ===========================
	def run(self) -> None:
		self._loop = asyncio.new_event_loop()
		try:
			while True:
				try:
					request = self._queue.get(timeout=0.1)
				except queue.Empty:
					if self._stop_event.is_set():
						break
					continue

				if request is None:
					self._queue.task_done()
					if self._stop_event.is_set():
						break
					continue

				try:
					cache_path = self._ensure_audio(request)
					if cache_path is not None:
						self._play_audio(cache_path, request.volume)
				except Exception as exc: # pragma: no cover - defensive log
					logger.error(f"EdgeTTS synthesis failed: {exc}")
				finally:
					if request.wait_event is not None:
						request.wait_event.set()
					self._queue.task_done()
		finally:
			self._loop.close()

	# ============================ Internals =============================
	def _ensure_audio(self, request: TTSRequest) -> Optional[Path]:
		cache_path = self._build_cache_path(request.text, request.rate, request.pitch)
		if cache_path.exists():
			return cache_path

		logger.info(f"Generating TTS audio: {cache_path.name}")
		try:
			self._loop.run_until_complete(
				self._synthesize_async(
					text=request.text,
					voice=request.voice,
					rate=request.rate,
					pitch=request.pitch,
					output_path=cache_path,
				)
			)
		except Exception as exc:
			logger.error(f"Failed to synthesize speech: {exc}")
			return None

		return cache_path

	async def _synthesize_async(
		self,
		*,
		text: str,
		voice: str,
		rate: str,
		pitch: str,
		output_path: Path,
	) -> None:
		communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
		await communicate.save(str(output_path))

	def _play_audio(self, audio_path: Path, volume: float) -> None:
		self._ensure_mixer_ready()
		try:
			sound = pygame.mixer.Sound(str(audio_path))
		except pygame.error as exc:
			logger.error(f"Unable to load synthesized audio: {exc}")
			if audio_path.exists():
				try:
					audio_path.unlink()
					logger.warning(f"Deleted potentially corrupt cache file: {audio_path}")
				except OSError as e:
					logger.error(f"Failed to delete cache file {audio_path}: {e}")
			return

		sound.set_volume(volume)
		with self._play_lock:
			if self._current_channel is not None:
				self._current_channel.stop()
			self._current_sound = sound
			self._current_channel = sound.play()

	def _build_cache_path(self, text: str, rate: str, pitch: str) -> Path:
		text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
		rate_fragment = self._sanitize_fragment(rate)
		pitch_fragment = self._sanitize_fragment(pitch)
		filename = f"{text_hash}_{rate_fragment}_{pitch_fragment}.wav"
		return self.cache_dir / filename

	def _sanitize_fragment(self, fragment: str) -> str:
		cleaned = fragment.replace("/", "_").replace("\\", "_").replace(":", "_").strip()
		return cleaned or "default"

	def _ensure_mixer_ready(self) -> None:
		if pygame.mixer.get_init() is None:
			try:
				pygame.mixer.init()
			except pygame.error as exc:
				logger.error(f"pygame mixer init failed: {exc}")
				raise

	def _clamp_volume(self, volume: float) -> float:
		if volume is None:
			return 0.8
		return max(0.0, min(1.0, volume))
