#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

"""Background thread that processes VLM (Vision-Language Model) requests for work feedback."""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import cv2
import numpy as np

import logging

import config
from utils.sulivan_logger import get_log_dir

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# ================================ Data Classes ================================
@dataclass
class LPIPSImageInfo:
    """Container for LPIPS reference and comparison images."""
    
    name: str  # Name of the LPIPS object
    reference_image: Optional[np.ndarray] = None  # Reference image (BGR format)
    comparison_image: Optional[np.ndarray] = None  # Latest comparison image (BGR format)
    last_score: Optional[float] = None  # Latest LPIPS score
    threshold: float = 0.1  # Threshold for completion


@dataclass
class VLMRequest:
    """Container for queued VLM requests."""

    frame: np.ndarray  # Camera frame (BGR format from OpenCV)
    step_info: Dict[str, Any]  # Current step configuration
    custom_prompt: Optional[str] = None  # Optional custom prompt for this request
    lpips_images: Optional[list] = None  # List of LPIPSImageInfo objects
    wait_event: Optional[threading.Event] = None


@dataclass
class VLMResponse:
    """Container for VLM responses."""

    success: bool
    text: str
    step_name: str
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


# ================================ Provider Base ================================
class VLMProviderBase(ABC):
    """Abstract base class for VLM providers.
    
    Extend this class to add support for different VLM backends
    (e.g., local models like LLaVA, Qwen-VL, or other cloud APIs).
    """

    @abstractmethod
    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        additional_images: Optional[list] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the VLM.
        
        Args:
            image: BGR image as numpy array (main camera frame)
            prompt: Text prompt for the model
            additional_images: List of (label, BGR image) tuples for additional context
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass


# ================================ Gemini Provider ================================
class GeminiVLMProvider(VLMProviderBase):
    """VLM provider using Google Gemini Vision API (google-genai package)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._initialized = False
        self._init_error: Optional[str] = None
        
        # Rate limit handling
        self._rate_limit_until: float = 0  # Timestamp until which we should not make requests
        self._consecutive_rate_limits: int = 0
        self._max_retries: int = 3
        self._base_retry_delay: float = 5.0  # Base delay in seconds

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the Gemini client."""
        if not self._api_key:
            self._init_error = "Gemini API key not provided"
            return

        try:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
            self._initialized = True
            logger.info(f"Gemini VLM provider initialized with model: {self._model}")
        except ImportError:
            self._init_error = "google-genai package not installed. Run: pip install google-genai"
            logger.error(self._init_error)
        except Exception as e:
            self._init_error = f"Failed to initialize Gemini client: {e}"
            logger.error(self._init_error)

    def is_available(self) -> bool:
        return self._initialized and self._client is not None

    def is_rate_limited(self) -> bool:
        """Check if we're currently in a rate limit cooldown period."""
        return time.time() < self._rate_limit_until

    def get_rate_limit_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        remaining = self._rate_limit_until - time.time()
        return max(0, remaining)

    def _parse_retry_delay(self, error_message: str) -> float:
        """Extract retry delay from error message if available."""
        import re
        # Look for patterns like "retry in 33.226011401s" or "retry_delay { seconds: 33 }"
        match = re.search(r'retry in (\d+\.?\d*)s', str(error_message))
        if match:
            return float(match.group(1))
        match = re.search(r'seconds:\s*(\d+)', str(error_message))
        if match:
            return float(match.group(1))
        return self._base_retry_delay

    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        additional_images: Optional[list] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        if not self.is_available():
            raise RuntimeError(self._init_error or "Gemini provider not available")

        # Check if we're in cooldown period
        if self.is_rate_limited():
            remaining = self.get_rate_limit_remaining()
            raise RuntimeError(f"Rate limited. Please wait {remaining:.1f} seconds before retrying.")

        try:
            from PIL import Image as PILImage

            # Convert BGR (OpenCV) to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            # Build contents list with prompt and images
            contents = [prompt, pil_image]
            
            # Add additional images if provided
            if additional_images:
                for label, img in additional_images:
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = PILImage.fromarray(img_rgb)
                        contents.append(f"\n[{label}]")
                        contents.append(pil_img)

            # Generate response using new google-genai API
            from google.genai import types
            
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            # Success - reset rate limit counter
            self._consecutive_rate_limits = 0
            return response.text.strip()

        except Exception as e:
            error_str = str(e)
            
            # Check if this is a rate limit error (HTTP 429)
            if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                self._consecutive_rate_limits += 1
                
                # Parse retry delay from error message or use exponential backoff
                retry_delay = self._parse_retry_delay(error_str)
                # Add exponential backoff for consecutive failures
                retry_delay = retry_delay * (1.5 ** (self._consecutive_rate_limits - 1))
                # Cap at 5 minutes
                retry_delay = min(retry_delay, 300)
                
                self._rate_limit_until = time.time() + retry_delay
                logger.warning(f"Rate limited. Cooling down for {retry_delay:.1f} seconds.")
                raise RuntimeError(f"Rate limited. Please wait {retry_delay:.1f} seconds before retrying.")
            
            logger.error(f"Gemini generation failed: {e}")
            raise


# ================================ Local Model Provider (Placeholder) ================================
class LocalVLMProvider(VLMProviderBase):
    """Placeholder for local VLM models (LLaVA, Qwen-VL, etc.).
    
    This class can be extended to support local models in the future.
    """

    def __init__(self, model_path: str, **kwargs):
        self._model_path = model_path
        self._model = None
        self._initialized = False
        logger.warning("LocalVLMProvider is not yet implemented")

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        additional_images: Optional[list] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        raise NotImplementedError("Local VLM provider is not yet implemented")


# ================================ VLM Thread ================================
class VLMThread(threading.Thread):
    """Thread that handles VLM requests for work feedback."""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_prompt_template: Optional[str] = None,
        auto_check_interval: Optional[int] = None,
    ) -> None:
        super().__init__(name="VLMWorker", daemon=True)
        self._queue: queue.Queue[Optional[VLMRequest]] = queue.Queue()
        self._stop_event = threading.Event()

        # Configuration
        self._provider_name = provider or getattr(config, "VLM_PROVIDER", "gemini")
        self._api_key = api_key or getattr(config, "VLM_API_KEY", "")
        self._model = model or getattr(config, "VLM_MODEL", "gemini-2.5-flash")
        self._auto_check_interval = auto_check_interval or getattr(config, "VLM_AUTO_CHECK_INTERVAL", 0)
        
        self._default_prompt_template = default_prompt_template or getattr(
            config,
            "VLM_DEFAULT_PROMPT_TEMPLATE",
            """당신은 작업 가이드 시스템의 AI 어시스턴트입니다.
작업자가 현재 수행 중인 작업 단계: {step_name}

이번 단계에서 해야 할 것:
{step_description}

완료 조건:
{completion_conditions}

{custom_prompt}

위 정보와 제공된 카메라 이미지를 보고, 작업자가 현재 작업을 올바르게 수행하고 있는지 평가하고,
도움이 될 수 있는 피드백을 한국어로 한 문장으로 간결하게 제공하세요.
문제가 없다면 "잘 하고 있습니다"라고만 답하세요."""
        )

        # Initialize provider
        self._provider: Optional[VLMProviderBase] = None
        self._init_provider()

        # State
        self._latest_response: Optional[VLMResponse] = None
        self._response_lock = threading.Lock()
        self._done_event = threading.Event()
        self._last_request_time: float = 0
        self._is_processing = False

        self.ready = False
        self._ready_event = threading.Event()

    def _init_provider(self) -> None:
        """Initialize the VLM provider based on configuration."""
        provider_name = self._provider_name.lower()

        if provider_name == "gemini":
            self._provider = GeminiVLMProvider(self._api_key, self._model)
        elif provider_name == "local":
            model_path = getattr(config, "VLM_LOCAL_MODEL_PATH", "")
            self._provider = LocalVLMProvider(model_path)
        else:
            logger.error(f"Unknown VLM provider: {provider_name}")
            self._provider = None

    def is_available(self) -> bool:
        """Check if VLM is available for requests."""
        return self._provider is not None and self._provider.is_available()

    def is_processing(self) -> bool:
        """Check if a request is currently being processed."""
        return self._is_processing

    def is_rate_limited(self) -> bool:
        """Check if we're currently in a rate limit cooldown period."""
        if self._provider is None:
            return False
        if hasattr(self._provider, 'is_rate_limited'):
            return self._provider.is_rate_limited()
        return False

    def get_rate_limit_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self._provider is None:
            return 0
        if hasattr(self._provider, 'get_rate_limit_remaining'):
            return self._provider.get_rate_limit_remaining()
        return 0

    # ============================ Public API ============================
    def request_feedback(
        self,
        frame: np.ndarray,
        step_info: Dict[str, Any],
        *,
        custom_prompt: Optional[str] = None,
        lpips_images: Optional[list] = None,
        wait: bool = False,
    ) -> Optional[threading.Event]:
        """Queue a VLM feedback request.
        
        Args:
            frame: Current camera frame (BGR format)
            step_info: Dictionary containing step information:
                - name: Step name
                - description: What needs to be done (optional)
                - completion_events: List of completion conditions (optional)
                - vlm_prompt: Custom prompt for this step (optional)
            custom_prompt: Override prompt for this specific request
            lpips_images: List of LPIPSImageInfo objects for context
            wait: If True, block until response is ready
            
        Returns:
            threading.Event if wait=False, else None after completion
        """
        if not self.is_available():
            logger.warning("VLM provider not available, skipping request")
            return None

        if frame is None:
            logger.warning("VLM request skipped: no frame provided")
            return None

        req = VLMRequest(
            frame=frame,  # frame is already a copy from getImage()
            step_info=step_info,
            custom_prompt=custom_prompt,
            lpips_images=lpips_images,
        )

        if wait:
            req.wait_event = threading.Event()

        self._done_event.clear()
        self._queue.put(req)

        if wait and req.wait_event is not None:
            req.wait_event.wait()
            return None

        return req.wait_event

    def get_latest_response(self) -> Optional[VLMResponse]:
        """Get the most recent VLM response."""
        with self._response_lock:
            return self._latest_response

    def get_auto_check_interval(self) -> int:
        """Get the auto check interval in milliseconds."""
        return self._auto_check_interval

    def set_auto_check_interval(self, interval_ms: int) -> None:
        """Set the auto check interval in milliseconds."""
        self._auto_check_interval = max(1000, interval_ms)  # Minimum 1 second

    def stop(self) -> None:
        """Stop the VLM worker thread."""
        self._stop_event.set()
        self._queue.put(None)

    # ============================ Thread loop ===========================
    def run(self) -> None:
        self.ready = True
        self._ready_event.set()
        logger.info("VLM thread started")

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

            self._is_processing = True
            try:
                response = self._process_request(request)
                with self._response_lock:
                    self._latest_response = response
            except Exception as exc:
                logger.error(f"VLM processing failed: {exc}")
                with self._response_lock:
                    self._latest_response = VLMResponse(
                        success=False,
                        text="",
                        step_name=request.step_info.get("name", "Unknown"),
                        error=str(exc),
                    )
            finally:
                self._is_processing = False
                self._done_event.set()
                if request.wait_event is not None:
                    request.wait_event.set()
                self._queue.task_done()
                self._last_request_time = time.time()

        logger.info("VLM thread stopped")

    # ============================ Internals =============================
    def _process_request(self, request: VLMRequest) -> VLMResponse:
        """Process a single VLM request."""
        step_info = request.step_info
        step_name = step_info.get("name", "Unknown")
        request_timestamp = datetime.now()

        # Build prompt
        prompt = self._build_prompt(step_info, request.custom_prompt, request.lpips_images)
        
        # Build additional images list from LPIPS images
        additional_images = self._build_additional_images(request.lpips_images)

        logger.debug(f"Processing VLM request for step: {step_name}")

        try:
            # Get max_tokens from config
            max_tokens = getattr(config, "VLM_MAX_TOKENS", 128)
            temperature = getattr(config, "VLM_TEMPERATURE", 0.7)

            response_text = self._provider.generate(
                request.frame,
                prompt,
                additional_images=additional_images,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            logger.info(f"VLM response for '{step_name}': {response_text}")
            
            # Save VLM request log
            self._save_vlm_log(
                request=request,
                prompt=prompt,
                response_text=response_text,
                success=True,
                error=None,
                timestamp=request_timestamp,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return VLMResponse(
                success=True,
                text=response_text,
                step_name=step_name,
            )

        except Exception as e:
            logger.error(f"VLM generation error: {e}")
            
            # Save VLM request log even on failure
            self._save_vlm_log(
                request=request,
                prompt=prompt,
                response_text="",
                success=False,
                error=str(e),
                timestamp=request_timestamp,
                max_tokens=getattr(config, "VLM_MAX_TOKENS", 128),
                temperature=getattr(config, "VLM_TEMPERATURE", 0.7),
            )
            
            return VLMResponse(
                success=False,
                text="",
                step_name=step_name,
                error=str(e),
            )

    def _save_vlm_log(
        self,
        request: VLMRequest,
        prompt: str,
        response_text: str,
        success: bool,
        error: Optional[str],
        timestamp: datetime,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Save VLM request log to {get_log_dir()}/vlm folder."""
        try:
            # Create vlm log directory
            vlm_log_dir = os.path.join(get_log_dir(), "vlm")
            os.makedirs(vlm_log_dir, exist_ok=True)
            
            # Generate timestamp string for filenames
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            step_name = request.step_info.get("name", "Unknown").replace(" ", "_").replace("/", "_")
            base_filename = f"{timestamp_str}_{step_name}"
            
            # Save camera frame
            if request.frame is not None:
                frame_path = os.path.join(vlm_log_dir, f"{base_filename}_frame.png")
                cv2.imwrite(frame_path, request.frame)
            
            # Save LPIPS images
            if request.lpips_images:
                for i, lpips_info in enumerate(request.lpips_images):
                    lpips_name = lpips_info.name.replace(" ", "_").replace("/", "_")
                    if lpips_info.reference_image is not None:
                        ref_path = os.path.join(vlm_log_dir, f"{base_filename}_lpips_{lpips_name}_reference.png")
                        cv2.imwrite(ref_path, lpips_info.reference_image)
                    if lpips_info.comparison_image is not None:
                        comp_path = os.path.join(vlm_log_dir, f"{base_filename}_lpips_{lpips_name}_comparison.png")
                        cv2.imwrite(comp_path, lpips_info.comparison_image)
            
            # Build metadata JSON
            metadata = {
                "timestamp": timestamp.isoformat(),
                "step_name": request.step_info.get("name", "Unknown"),
                "step_description": request.step_info.get("description", ""),
                "completion_events": request.step_info.get("completion_events", []),
                "tasks": request.step_info.get("tasks", ""),
                "explain_for_vlm": request.step_info.get("explain_for_vlm", ""),
                "runtime": {
                    "step_runtime": request.step_info.get("step_runtime", 0),
                    "scenario_runtime": request.step_info.get("scenario_runtime", 0),
                },
                "custom_prompt": request.custom_prompt,
                "vlm_config": {
                    "provider": self._provider_name,
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                "lpips_info": [],
                "prompt": prompt,
                "response": {
                    "success": success,
                    "text": response_text,
                    "error": error,
                },
            }
            
            # Add LPIPS info to metadata
            if request.lpips_images:
                for lpips_info in request.lpips_images:
                    metadata["lpips_info"].append({
                        "name": lpips_info.name,
                        "threshold": lpips_info.threshold,
                        "last_score": lpips_info.last_score,
                        "has_reference": lpips_info.reference_image is not None,
                        "has_comparison": lpips_info.comparison_image is not None,
                    })
            
            # Save metadata JSON
            json_path = os.path.join(vlm_log_dir, f"{base_filename}_metadata.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"VLM log saved to {vlm_log_dir}/{base_filename}_*")
            
        except Exception as e:
            logger.error(f"Failed to save VLM log: {e}")

    def _build_additional_images(self, lpips_images: Optional[list]) -> Optional[list]:
        """Build list of additional images for VLM context."""
        if not lpips_images:
            return None
        
        additional = []
        for lpips_info in lpips_images:
            if lpips_info.reference_image is not None:
                additional.append((f"{lpips_info.name} - 목표 조립 상태 (Reference)", lpips_info.reference_image))
            if lpips_info.comparison_image is not None:
                score_text = f", 유사도 점수: {lpips_info.last_score:.4f}/{lpips_info.threshold}" if lpips_info.last_score is not None else ""
                additional.append((f"{lpips_info.name} - 현재 조립 상태 (Current){score_text}", lpips_info.comparison_image))
        
        return additional if additional else None

    def _build_prompt(self, step_info: Dict[str, Any], custom_prompt: Optional[str], lpips_images: Optional[list] = None) -> str:
        """Build the prompt for the VLM request."""
        step_name = step_info.get("name", "Unknown")
        
        # Get step description from step_info
        step_description = step_info.get("description", "")
        
        # Get tasks (what needs to be done to complete the step)
        tasks_text = step_info.get("tasks", "- 특정 조건 없음")
        
        # Get runtime info
        step_runtime = step_info.get("step_runtime", 0)
        scenario_runtime = step_info.get("scenario_runtime", 0)
        
        # Build runtime context with detail level based on elapsed time
        runtime_context = self._build_runtime_context(step_runtime, scenario_runtime)

        # Use custom prompt if provided, otherwise use step's vlm_prompt or empty
        extra_prompt = custom_prompt or step_info.get("vlm_prompt", "")
        
        # Add scenario explanation if available
        explain_for_vlm = step_info.get("explain_for_vlm", "")
        if explain_for_vlm:
            extra_prompt = f"[시나리오 설명]\n{explain_for_vlm}\n\n{extra_prompt}"
        
        # Add runtime context
        if runtime_context:
            extra_prompt = f"{extra_prompt}\n\n{runtime_context}"
        
        # Add LPIPS image context if available
        lpips_context = ""
        if lpips_images:
            lpips_descriptions = []
            for lpips_info in lpips_images:
                desc = f"- '{lpips_info.name}': "
                if lpips_info.last_score is not None:
                    if lpips_info.last_score < lpips_info.threshold:
                        desc += f"조립 완료 (유사도: {lpips_info.last_score:.4f}, 기준: {lpips_info.threshold})"
                    else:
                        desc += f"조립 미완료 (유사도: {lpips_info.last_score:.4f}, 기준: {lpips_info.threshold} 이하 필요)"
                else:
                    desc += "비교 대기 중"
                lpips_descriptions.append(desc)
            
            lpips_context = "\n\n참조 이미지 비교 정보:\n" + "\n".join(lpips_descriptions)
            lpips_context += "\n(첨부된 Reference 이미지는 목표 조립 상태, Current 이미지는 현재 상태입니다. 두 이미지를 비교하여 조립 진행 상황을 판단해주세요.)"

        # Format the template
        prompt = self._default_prompt_template.format(
            step_name=step_name,
            step_description=step_description,
            completion_conditions=tasks_text,
            custom_prompt=extra_prompt + lpips_context,
        )

        return prompt

    def _build_runtime_context(self, step_runtime: float, scenario_runtime: float) -> str:
        """Build runtime context with detail level based on elapsed time.
        
        Args:
            step_runtime: Current step elapsed time in seconds
            scenario_runtime: Total scenario elapsed time in seconds
            
        Returns:
            Runtime context string with appropriate detail level
        """
        # Format times as MM:SS
        step_time_str = f"{int(step_runtime // 60):02d}:{int(step_runtime % 60):02d}"
        scenario_time_str = f"{int(scenario_runtime // 60):02d}:{int(scenario_runtime % 60):02d}"
        
        context = f"[소요 시간]\n현재 단계: {step_time_str}, 전체 시나리오: {scenario_time_str}\n\n"
        
        # Determine detail level based on step runtime
        if step_runtime < 30:
            # Under 30 seconds: Brief feedback
            context += "[피드백 수준: 간단]\n작업이 시작된 지 얼마 되지 않았습니다. 간단하게 현재 상황만 알려주세요."
        elif step_runtime < 60:
            # 30-60 seconds: Normal feedback
            context += "[피드백 수준: 보통]\n작업이 진행 중입니다. 현재 상태를 평가하고 필요한 조언을 제공해주세요."
        elif step_runtime < 120:
            # 1-2 minutes: Detailed feedback
            context += "[피드백 수준: 상세]\n이 단계에서 시간이 많이 소요되고 있습니다. 작업자가 어려움을 겪고 있을 수 있으니, 현재 상태를 자세히 분석하고 구체적인 단계별 가이드를 제공해주세요."
        else:
            # Over 2 minutes: Very detailed feedback with encouragement
            context += "[피드백 수준: 매우 상세]\n이 단계에서 상당한 시간이 소요되고 있습니다. 작업자가 어려움을 겪고 있을 가능성이 높습니다.\n"
            context += "다음을 포함하여 매우 상세하게 안내해주세요:\n"
            context += "1. 현재 화면에서 보이는 조립 상태 분석\n"
            context += "2. 다음으로 해야 할 정확한 동작 설명\n"
            context += "3. 흔히 발생하는 실수와 해결 방법\n"
            context += "4. 격려의 메시지"
        
        return context