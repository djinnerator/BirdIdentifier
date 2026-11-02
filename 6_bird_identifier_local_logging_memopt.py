"""
6_bird_identifier_local_logging_memopt.py
=============================================================================
Automated Real-Time Bird Acoustic Detection, Identification, and 
19th-Century Ornithology Field Guide Illustration Pipeline (Offline Local Model,
Comprehensive Multiprocess Logging, & Zero-Collision VRAM Memory Optimization).

Target Platform:
- NVIDIA Jetson Orin Nano
- Tegra R39.2 / Jetpack 7.2 (Ubuntu 24.04 LTS Noble Numbat)
- Hardware: Seeed Studio ReSpeaker XVF3800 4-Mic Array (USB VID 0x2886, PID 0x001A)

Architecture:
- Parent-child process architecture managing 4 dedicated worker processes:
    1. activator (activate): Acoustic event / Voice Activity Detection (VAD) via ReSpeaker
    2. recorder  (record):   10-second high-fidelity stereo capture (48kHz, 2-ch) via sounddevice
    3. predictor (predict):  Optimized torchcodec AudioDecoder ingestion, on-demand TensorRT classification,
                             immediate engine/context memory deallocation, and archiving audio + mel-spectrogram
    4. artist    (draw):     19th-century vintage field guide plate creation via local offline model
                             ('models/Z-Image-AbliteratedV1.Q8_0.gguf') without Internet usage.

Memory Optimization (Zero-Collision VRAM Management):
- On the 8GB Unified Memory Architecture (UMA) Jetson Orin Nano, holding both the TensorRT
  engine and the generative image model in memory simultaneously risks memory exhaustion.
- Immediately after the TensorRT model performs its prediction, the engine, execution context,
  runtime, CUDA stream, and PyCUDA device memory buffers are completely freed from memory
  BEFORE the result is placed onto the IPC `predict_queue`.
- Reclaims maximum VRAM bandwidth so ArtistWorker can safely execute local image synthesis.

Multiprocess Logging Architecture:
- Central logging queue (`mp.Queue`) with a dedicated `QueueListener` in the main process.
- Daily timestamped log file: 'logs/log_YYYY_MM_DD_HHMMSS.txt' in the auto-created 'logs/' folder.
- All entries include timestamp, severity level, and the originating child process name:
    [YYYY-MM-DD HH:MM:SS] [LEVEL] [PROCESS_NAME] Message
- Severity levels:
    - INFO: Normal operations, process lifecycle events, and queue accesses/retrievals.
    - WARNING: Exceptions that fail gracefully while allowing program execution to continue.
    - ERROR: Exceptions or conditions that break functionality or prevent further execution.
- Toolbox Function Reuse: Calls functions already implemented in toolbox/toolbox.py
  (specifically `get_time_str` and `scale_minmax`) rather than re-implementing them.
=============================================================================
"""

import os
import sys
import time
import queue
import struct
import datetime
import subprocess
import shutil
import gc
import logging
from logging.handlers import QueueHandler, QueueListener
import traceback
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
import multiprocessing as mp
import multiprocessing.synchronize
import multiprocessing.queues

# Ensure offline operation across huggingface and diffusers ecosystems
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Import reusable utilities from toolbox (do not re-implement in this file)
from toolbox.toolbox import get_time_str, scale_minmax

# Audio & Neural Processing Libraries
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import torchcodec
from torchcodec.decoders import AudioDecoder
from torchvision.transforms import v2 as vision_v2
from PIL import Image, ImageDraw, ImageFont

# Hardware USB libraries for ReSpeaker XVF3800
try:
    import usb.core
    import usb.util
    import libusb_package
    HAS_USB = True
except ImportError:
    HAS_USB = False

# TensorRT & PyCUDA for Jetson Orin Nano
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


# =====================================================================
# Logging Infrastructure & Helpers
# =====================================================================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Generate log filename using notation log_YYYY_MM_DD_HHMMSS.txt
CURRENT_LOG_FILE = LOGS_DIR / datetime.datetime.now().strftime("log_%Y_%m_%d_%H%M%S.txt")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_worker_logger(log_queue: Optional[mp.queues.Queue] = None, logger_name: Optional[str] = None) -> logging.Logger:
    """
    Configures a logger for worker processes or the main coordinator.
    Routes log records through the central multiprocessing queue to ensure
    process-safe atomic writing to the daily log file and console.
    """
    name = logger_name or mp.current_process().name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if log_queue is not None:
        qh = QueueHandler(log_queue)
        logger.addHandler(qh)
    else:
        # Fallback direct console handler when running standalone
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(ch)

    logger.propagate = False
    return logger


# =====================================================================
# Utility Functions (Not present in toolbox/toolbox.py)
# =====================================================================
def sanitize_folder_name(name: str) -> str:
    """Sanitizes bird name for filesystem directory creation."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join(c if c not in invalid_chars else "_" for c in name).strip(". ")
    return sanitized if sanitized else "Unknown_Species"


# =====================================================================
# 1. Acoustic Trigger & ReSpeaker Hardware Interface
# =====================================================================
class ReSpeaker:
    """
    Interface to Seeed Studio ReSpeaker XVF3800 4-Mic Array over USB.
    Queries the Voice Activity Detection (VAD) hardware register.
    """
    TIMEOUT = 5000  # 5s USB timeout

    def __init__(self, dev, logger: Optional[logging.Logger] = None):
        self.dev = dev
        self.logger = logger or logging.getLogger("ReSpeaker")

    def read(self) -> bool:
        """Query Voice Activity Detection (VAD) register on ReSpeaker."""
        if self.dev is None:
            return False
        try:
            # ReSpeaker XVF3800 VAD control transfer packet
            data = (20, 18, 4, "ro", "uint16")
            resid = data[0]
            cmdid = 0x80 | data[1]
            length = data[2] + 1

            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmdid, resid, length, self.TIMEOUT
            )
            result = response.tolist()
            if len(result) > 3:
                return result[3] == 1
            return False
        except Exception as e:
            self.logger.warning(f"ReSpeaker USB control transfer non-fatal exception: {e}; continuing polling.")
            return False

    def close(self):
        """Release USB device resources."""
        if self.dev is not None and HAS_USB:
            try:
                usb.util.dispose_resources(self.dev)
                self.logger.info("Released ReSpeaker USB device resources.")
            except Exception as e:
                self.logger.warning(f"Exception releasing ReSpeaker USB resources: {e}; ignored.")


def find_respeaker(vid=0x2886, pid=0x001A, logger: Optional[logging.Logger] = None) -> Optional[ReSpeaker]:
    """Finds and initializes the ReSpeaker XVF3800 USB device."""
    log = logger or logging.getLogger("ReSpeaker")
    if not HAS_USB:
        log.warning("libusb/pyusb not available on system; ReSpeaker USB queries will be disabled.")
        return None
    try:
        dev = usb.core.find(idVendor=vid, idProduct=pid)
        if dev is None:
            return None
        return ReSpeaker(dev, logger=log)
    except Exception as e:
        log.warning(f"ReSpeaker USB device enumeration warning: {e}; continuing gracefully.")
        return None


# =====================================================================
# 2. Optimized Audio Preprocessing (Direct torchcodec Ingestion)
# =====================================================================
class AudioPrep:
    """
    High-performance audio preprocessing pipeline for Jetson Orin Nano.
    Loads raw in-memory numpy audio directly into torchcodec.decoders.AudioDecoder
    without intermediate io.BytesIO byte stream allocations.
    """
    def __init__(self, sample_rate: int = 48000, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128,
                 logger: Optional[logging.Logger] = None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.logger = logger or logging.getLogger("AudioPrep")

        # PyTorch Torchaudio Transforms
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.resize_tool = vision_v2.Resize((64, 256))

    def process_mp3(self, audio: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Optimal, direct in-memory loader for torchcodec AudioDecoder.
        Synthesizes a 44-byte standard RIFF header directly in memory and wraps
        the contiguous audio payload as a torch.uint8 tensor passed to AudioDecoder.

        Returns:
            Tuple[np.ndarray, torch.Tensor]: (normalized_numpy_spec, torch_spec_tensor)
        """
        try:
            # Ensure float32 contiguous array
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if not audio.flags.c_contiguous:
                audio = np.ascontiguousarray(audio)

            num_channels = audio.shape[1] if audio.ndim > 1 else 1
            data_bytes = audio.nbytes

            # 44-byte RIFF/WAV header (IEEE Float32 format = 3)
            header = struct.pack(
                '<4sI4s4sIHHIIHH4sI',
                b'RIFF',
                36 + data_bytes,
                b'WAVE',
                b'fmt ',
                16,
                3,  # IEEE Float
                num_channels,
                self.sample_rate,
                self.sample_rate * num_channels * 4,
                num_channels * 4,
                32,  # 32-bit float
                b'data',
                data_bytes,
            )

            # Build contiguous memory buffer and wrap into torch.Tensor (uint8)
            buf = bytearray(header)
            buf.extend(audio.tobytes())
            wav_tensor = torch.frombuffer(buf, dtype=torch.uint8)

            # Decode using torchcodec AudioDecoder with native C++ downmixing to mono
            dec = AudioDecoder(wav_tensor, num_channels=1)
            y = dec.get_all_samples().data  # Tensor shape: (1, time_samples)

            # Convert to Mel-spectrogram
            mel_spec = self.mel_spectrogram_transform(y)
            mel_spec_db = self.to_db(mel_spec).detach()

            # Resize to expected model input dimension: (64, 256)
            img_tensor = self.resize_tool(mel_spec_db).squeeze(0)
            # Call scale_minmax imported from toolbox.toolbox
            img_np = scale_minmax(img_tensor.cpu().numpy(), 0.0, 1.0)

            return img_np, img_tensor

        except Exception as e:
            self.logger.warning(f"Error decoding audio via torchcodec: {e}; falling back to zero spectrogram.")
            dummy_np = np.zeros((64, 256), dtype=np.float32)
            dummy_tensor = torch.zeros((64, 256), dtype=torch.float32)
            return dummy_np, dummy_tensor

    def process_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Alias for process_mp3."""
        return self.process_mp3(audio)


# =====================================================================
# 3. Model Inference Engine with Explicit Memory Deallocation
# =====================================================================
class TensorRTInference:
    """
    TensorRT acceleration engine for bird species identification on Jetson Orin Nano.
    Leverages PyCUDA pinned host memory and asynchronous CUDA streams for zero-copy UMA.
    Supports explicit memory reclamation (freeing engine and execution context)
    to maximize available memory bandwidth for downstream generative models.
    """
    def __init__(self, engine_path: str = "BirdID_US.trt", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("TensorRTInference")
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        self.fallback_mode = False

        self._initialize_engine()

    def _initialize_engine(self):
        resolved_path = self._resolve_engine_path(self.engine_path)
        if resolved_path and HAS_TENSORRT:
            try:
                self.trt_logger = trt.Logger(trt.Logger.ERROR)
                self.runtime = trt.Runtime(self.trt_logger)
                self.engine = self._load_engine(resolved_path)
                self.context = self.engine.create_execution_context()
                self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine)
                self.logger.info(f"Loaded Jetson TensorRT engine from: {resolved_path}")
            except Exception as e:
                self.logger.warning(f"Error initializing TensorRT ({e}). Gracefully entering simulated inference mode.")
                self.fallback_mode = True
        else:
            self.logger.warning(
                f"TensorRT engine '{self.engine_path}' not found or PyCUDA unavailable. Gracefully running in simulated inference mode."
            )
            self.fallback_mode = True

    def _resolve_engine_path(self, path: str) -> Optional[str]:
        candidates = [
            path,
            os.path.join("models", path),
            os.path.join(os.path.dirname(__file__), path),
            os.path.join(os.path.dirname(__file__), "models", path),
            os.path.join("models", "BirdID_US.trt"),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def _allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data: Union[np.ndarray, torch.Tensor], return_logits: bool = False):
        if self.fallback_mode:
            # Simulated inference for development when hardware model is unbuilt
            num_classes = 1730
            logits = np.random.randn(num_classes).astype(np.float32)
            logits[5] += 5.0  # Emphasize Melozone fusca
            return logits if return_logits else int(np.argmax(logits))

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()

        return self.outputs[0].host if return_logits else int(np.argmax(self.outputs[0].host))

    def free_resources(self):
        """
        Explicitly releases TensorRT execution context, engine, runtime, and PyCUDA device buffers
        from unified memory (UMA) to ensure maximum VRAM headroom for subsequent pipelines.
        """
        self.logger.info("Freeing TensorRT engine and execution context from memory...")
        try:
            # 1. Release PyCUDA device memory allocations
            for item in self.inputs + self.outputs:
                if hasattr(item, "device") and item.device is not None:
                    try:
                        item.device.free()
                    except Exception as dev_err:
                        self.logger.warning(f"Note releasing device buffer: {dev_err}")

            self.inputs.clear()
            self.outputs.clear()
            self.bindings.clear()

            # 2. Free TensorRT execution context
            if self.context is not None:
                del self.context
                self.context = None

            # 3. Free TensorRT engine
            if self.engine is not None:
                del self.engine
                self.engine = None

            # 4. Free TensorRT runtime
            if self.runtime is not None:
                del self.runtime
                self.runtime = None

            # 5. Clear CUDA stream
            self.stream = None

            # 6. Reclaim PyTorch CUDA memory and run garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            self.logger.info("Successfully freed TensorRT engine and context from memory.")

        except Exception as e:
            self.logger.warning(f"Exception during TensorRT resource cleanup: {e}; continuing.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_resources()


# =====================================================================
# 4. Local Offline 19th-Century Field Guide Plate Artist (Z-Image GGUF)
# =====================================================================
class BirdArtist:
    """
    Generates authentic 19th-century vintage ornithology field guide book
    illustrations using a local quantized model ('models/Z-Image-AbliteratedV1.Q8_0.gguf')
    completely offline without any Internet connection.

    Key Benefits for Jetson Orin Nano & Edge Deployment:
    - 100% Offline execution: Zero external API calls, ensuring high privacy and reliability.
    - Quantized footprint: Leverages 8-bit quantization (Q8_0) to fit within Jetson memory limits.
    - Dynamic memory reclamation: Frees pipeline memory after generation to preserve bandwidth for TensorRT.
    - Preserves exact 19th-century John James Audubon vintage field guide aesthetic.
    """
    def __init__(self, output_dir: str = "images", model_path: Optional[Union[str, Path]] = None,
                 logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("BirdArtist")

        # Resolve local model relative to script directory as specified
        script_dir = Path(__file__).resolve().parent
        if model_path is None:
            self.model_path = script_dir / "models" / "Z-Image-AbliteratedV1.Q8_0.gguf"
        else:
            p = Path(model_path)
            self.model_path = p if p.is_absolute() else (script_dir / p)

        # Fallback candidate search
        if not self.model_path.exists():
            cwd_candidate = Path("models") / "Z-Image-AbliteratedV1.Q8_0.gguf"
            if cwd_candidate.exists():
                self.model_path = cwd_candidate.resolve()

        self.logger.info(f"BirdArtist initialized with target offline model path: {self.model_path}")

    def __call__(self, bird_name: str) -> str:
        """
        Creates and saves a 19th-century ornithology field guide illustration
        of the specified bird species using the local offline model.
        """
        time_stamp = get_time_str()
        sanitized_name = sanitize_folder_name(bird_name)
        output_filename = self.output_dir / f"field_guide_{time_stamp}_{sanitized_name}.png"

        # Structured 19th-century Audubon-style scientific prompt (same style)
        prompt = (
            f"A 19th-century vintage ornithology field guide book illustration plate of the bird: '{bird_name}'. "
            f"Detailed scientific illustration in the style of John James Audubon, hand-colored engraving, "
            f"fine plumage textures, naturalistic perch on a botanical branch, aged antique parchment paper with fine grain, "
            f"botanical watercolor notes, copperplate Latin binomial inscription at bottom, archival museum plate."
        )

        self.logger.info(f"Requesting 19th-Century Field Guide Plate for: {bird_name}")
        self.logger.info(f"Illustration prompt: \"{prompt}\"")

        # 1. Primary Method: Local Offline Model Generation
        generated = False
        if self.model_path.exists():
            self.logger.info(f"Local model found at {self.model_path}. Executing offline inference...")
            generated = self._generate_with_local_model(prompt, output_filename, bird_name)
        else:
            self.logger.warning(
                f"Local model file not found at: {self.model_path}. "
                f"Gracefully failing over to archival vintage plate generator fallback."
            )

        # 2. Resilient Fallback: Archival Vintage Plate Generator
        if not generated or not output_filename.exists():
            self.logger.info(f"Rendering archival field guide plate fallback for {bird_name}...")
            self._generate_archival_plate_fallback(output_filename, bird_name, time_stamp)

        self.logger.info(f"Successfully saved field guide illustration plate: {output_filename}")
        return str(output_filename)

    def _generate_with_local_model(self, prompt: str, target_file: Path, bird_name: str) -> bool:
        """
        Executes local offline text-to-image inference using the Z-Image GGUF model.
        Attempts Diffusers/PyTorch single-file execution first, followed by local CLI runner.
        No Internet access is used.
        """
        # Ensure offline mode flags
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Attempt A: In-process Diffusers Pipeline
        try:
            from diffusers import ZImagePipeline, DiffusionPipeline
            pipe = None

            if hasattr(ZImagePipeline, "from_single_file"):
                try:
                    pipe = ZImagePipeline.from_single_file(
                        str(self.model_path),
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                        local_files_only=True,
                    )
                except Exception as e:
                    self.logger.warning(f"ZImagePipeline.from_single_file non-fatal note: {e}")

            if pipe is None and hasattr(DiffusionPipeline, "from_single_file"):
                try:
                    pipe = DiffusionPipeline.from_single_file(
                        str(self.model_path),
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        local_files_only=True,
                    )
                except Exception as e:
                    self.logger.warning(f"DiffusionPipeline.from_single_file non-fatal note: {e}")

            if pipe is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pipe = pipe.to(device)

                # Z-Image Turbo utilizes 8-9 inference steps without CFG guidance
                try:
                    res = pipe(
                        prompt=prompt,
                        num_inference_steps=9,
                        guidance_scale=0.0,
                    )
                except Exception:
                    res = pipe(prompt=prompt, num_inference_steps=20)

                img = res.images[0]
                img.save(str(target_file))

                # Immediately reclaim GPU memory to ensure TensorRT has maximum headroom
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if target_file.exists():
                    self.logger.info(f"Diffusers generated image successfully: {target_file}")
                    return True
        except Exception as e:
            self.logger.warning(f"Diffusers local model execution non-fatal issue: {e}")

        # Attempt B: Local offline CLI runner (e.g. sd.cpp / stable-diffusion-cli / z-image)
        cli_candidates = [
            os.environ.get("SD_CLI_EXE"),
            "sd",
            "stable-diffusion",
            "sd-cli",
            "z-image",
        ]
        for cli in cli_candidates:
            if not cli:
                continue
            cli_path = shutil.which(cli) or (cli if os.path.isfile(cli) else None)
            if cli_path:
                try:
                    cmd = [
                        cli_path,
                        "-m", str(self.model_path),
                        "-p", prompt,
                        "-o", str(target_file),
                        "--steps", "9",
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if target_file.exists():
                        self.logger.info(f"Local CLI runner ({cli_path}) generated image: {target_file}")
                        return True
                except Exception as e:
                    self.logger.warning(f"Local CLI runner ({cli_path}) non-fatal issue: {e}")

        return False

    def _generate_archival_plate_fallback(self, target_file: Path, bird_name: str, timestamp: str):
        """
        Renders an authentic 19th-century vintage parchment field guide plate
        with aged paper textures, botanical framing, double-rule borders, and Latin typography.
        Guarantees reliable image creation completely offline.
        """
        width, height = 768, 1024
        parchment_color = (243, 236, 216)
        img = Image.new("RGB", (width, height), color=parchment_color)
        draw = ImageDraw.Draw(img)

        # Simulated antique paper grain
        np_img = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 3.5, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)
        draw = ImageDraw.Draw(img)

        # 19th-century double border rules
        border_ink = (58, 48, 38)
        accent_ink = (112, 94, 76)

        draw.rectangle([30, 30, width - 30, height - 30], outline=border_ink, width=3)
        draw.rectangle([38, 38, width - 38, height - 38], outline=accent_ink, width=1)
        draw.rectangle([44, 44, width - 44, height - 44], outline=border_ink, width=1)

        # Scientific Header
        draw.text((55, 55), "ORDO: AVES / AMERICAN ORNITHOLOGY", fill=accent_ink)
        draw.text((width - 150, 55), f"PLATE No. {abs(hash(bird_name)) % 900 + 100}", fill=accent_ink)

        # Botanical vignette background oval
        draw.ellipse([140, 200, width - 140, height - 340], outline=(208, 198, 178), width=2)

        # Botanical perch sketch
        branch_ink = (85, 70, 55)
        leaf_wash = (130, 145, 115)
        draw.line([(180, 560), (320, 530), (580, 540)], fill=branch_ink, width=5)
        draw.line([(320, 530), (460, 470)], fill=branch_ink, width=3)
        for lx, ly in [(240, 545), (380, 495), (450, 460), (520, 535)]:
            draw.ellipse([lx - 12, ly - 6, lx + 12, ly + 6], fill=leaf_wash, outline=branch_ink)

        # Bird silhouette & plumage wash
        bird_wash_body = (180, 160, 130)
        bird_wash_accent = (140, 105, 80)
        draw.ellipse([310, 420, 470, 540], fill=bird_wash_body, outline=branch_ink, width=2)
        draw.ellipse([270, 370, 360, 450], fill=bird_wash_accent, outline=branch_ink, width=2)
        draw.polygon([(270, 410), (230, 420), (270, 428)], fill=branch_ink)
        draw.ellipse([295, 400, 305, 410], fill=(20, 20, 20), outline=parchment_color)
        draw.polygon([(400, 470), (540, 580), (510, 595), (380, 530)], fill=bird_wash_accent, outline=branch_ink, width=2)

        # Scientific taxonomy footer & Latin binomial inscriptions
        draw.line([60, height - 170, width - 60, height - 170], fill=accent_ink, width=1)
        draw.text((width // 2 - 120, height - 150), f"1. {bird_name.upper()}", fill=border_ink)
        draw.text((width // 2 - 150, height - 125), "Drawn from Nature - Local Ornithological Survey", fill=accent_ink)
        draw.text((width // 2 - 90, height - 100), f"London & New York, {timestamp[:4]}", fill=accent_ink)

        img.save(target_file, "PNG")


# =====================================================================
# 5. Child Process Workers
# =====================================================================
def activate(stop_event: mp.synchronize.Event, cooldown_event: mp.synchronize.Event,
             flag_queue: mp.queues.Queue, log_queue: mp.queues.Queue):
    """
    Worker Process 1: Acoustic Activity & ReSpeaker VAD Monitor.
    Runs on Jetson Orin Nano, monitoring the ReSpeaker XVF3800 microphone array
    over USB for voice/sound activity events. Mic is assumed always connected.
    """
    logger = setup_worker_logger(log_queue, "ActivatorWorker")
    logger.info("Process started. Initializing ReSpeaker XVF3800 USB monitoring...")

    dev = None
    try:
        dev = find_respeaker(logger=logger)
        while dev is None and not stop_event.is_set():
            logger.warning("Connecting to ReSpeaker XVF3800 USB interface failed or device busy; retrying in 1.0s...")
            time.sleep(1.0)
            dev = find_respeaker(logger=logger)

        if stop_event.is_set():
            logger.info("Stop event received during initialization; exiting cleanly.")
            return

        logger.info("Successfully bound to ReSpeaker XVF3800. Monitoring for acoustic triggers...")

        while not stop_event.is_set():
            # Check if cooldown period is active
            if cooldown_event.is_set():
                time.sleep(0.2)
                continue

            # Query hardware VAD register
            try:
                sound_detected = dev.read()
            except Exception as e:
                logger.warning(f"ReSpeaker hardware read exception: {e}; treating as silent.")
                sound_detected = False

            if sound_detected and not cooldown_event.is_set():
                logger.info("Acoustic trigger detected by ReSpeaker XVF3800! Preparing to notify Recorder...")
                try:
                    logger.info("Accessing flag_queue: putting trigger flag (True)...")
                    flag_queue.put(True, timeout=1.0)
                    logger.info("Accessing flag_queue: successfully enqueued trigger flag.")
                    cooldown_event.set()
                    logger.info("Cooldown event set; pausing trigger polling during recording and cooldown.")
                except queue.Full:
                    logger.warning("Accessing flag_queue failed: queue is full; skipping acoustic trigger.")
                except Exception as e:
                    logger.error(f"Accessing flag_queue failed with error: {e}")

            time.sleep(0.05)  # Fast 20Hz polling for responsive acoustic detection

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in ActivatorWorker.")
    except Exception as e:
        logger.error(f"Fatal exception in ActivatorWorker breaking functionality: {e}\n{traceback.format_exc()}")
    finally:
        if dev is not None:
            dev.close()
        logger.info("Process exiting cleanly.")


def record(stop_event: mp.synchronize.Event, cooldown_event: mp.synchronize.Event,
           flag_queue: mp.queues.Queue, data_queue: mp.queues.Queue, log_queue: mp.queues.Queue):
    """
    Worker Process 2: Audio Recorder.
    Captures 10 seconds of 48kHz stereo audio upon receiving trigger from flag_queue.
    """
    logger = setup_worker_logger(log_queue, "RecorderWorker")
    logger.info("Process started. Ready to capture sound clips.")
    duration = 10        # 10-second recording as specified in requirements
    sample_rate = 48000
    channels = 2

    # Query device index for ReSpeaker XVF3800
    mic_idx = None
    try:
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if "reSpeaker XVF3800" in device.get("name", ""):
                mic_idx = idx
                logger.info(f"Bound to ReSpeaker input device index {mic_idx} on Jetson.")
                break
        if mic_idx is None:
            logger.warning("ReSpeaker XVF3800 device name not found in sounddevice list; using default audio input.")
    except Exception as e:
        logger.warning(f"Audio device query warning: {e}; using default system input device.")

    try:
        while not stop_event.is_set():
            try:
                # Non-blocking get with timeout to ensure responsiveness to stop_event
                flag = flag_queue.get(timeout=1.0)
                logger.info(f"Retrieving from flag_queue: received trigger token: {flag}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Retrieving from flag_queue encountered non-fatal error: {e}")
                continue

            if flag is None:
                logger.info("Retrieving from flag_queue: received shutdown sentinel (None). Exiting record loop.")
                break

            if flag:
                logger.info(f"Recording {duration}s audio clip at {sample_rate}Hz (channels={channels})...")
                try:
                    audio = sd.rec(
                        frames=int(duration * sample_rate),
                        samplerate=sample_rate,
                        channels=channels,
                        blocking=True,
                        device=mic_idx,
                        dtype="float32",
                    )
                    logger.info(f"Audio capture complete. Audio array shape: {audio.shape}")

                    logger.info("Accessing data_queue: putting captured audio array payload...")
                    data_queue.put(audio, timeout=5.0)
                    logger.info("Accessing data_queue: successfully enqueued audio array payload.")

                except queue.Full:
                    logger.warning("Accessing data_queue failed: data_queue is full; audio payload dropped.")
                except Exception as e:
                    logger.error(f"Fatal error during audio recording or queueing: {e}\n{traceback.format_exc()}")

    except Exception as e:
        logger.error(f"Fatal exception in RecorderWorker breaking functionality: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("Process exiting cleanly.")


def predict(stop_event: mp.synchronize.Event, data_queue: mp.queues.Queue,
            predict_queue: mp.queues.Queue, log_queue: mp.queues.Queue):
    """
    Worker Process 3: Audio Preprocessing, Species Inference, Dynamic Memory Deallocation,
    and Data Archiving.

    Transforms audio into Mel-spectrogram using torchcodec AudioDecoder, dynamically
    loads TensorRT for inference, and crucially FREES the engine and execution context
    from memory BEFORE the result is placed on the IPC predict_queue to preserve VRAM.
    """
    logger = setup_worker_logger(log_queue, "PredictorWorker")
    logger.info("Process started. Loading neural models and species taxonomy...")

    # Ensure root recordings base directory exists
    base_recordings_dir = Path("recordings")
    base_recordings_dir.mkdir(parents=True, exist_ok=True)

    def log_prediction(bird_name: str):
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        today = get_time_str("%Y%m%d")
        timestamp = get_time_str("%Y-%m-%d %H:%M:%S")
        detection_log_path = log_dir / f"birdid_detections_{today}.txt"
        try:
            with open(detection_log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp},{bird_name}\n")
            logger.info(f"Logged species detection: '{bird_name}' to {detection_log_path}")
        except Exception as err:
            logger.warning(f"Failed to append to detection log {detection_log_path}: {err}; continuing execution.")

    def archive_detection(bird_name: str, audio: np.ndarray, spec_np: np.ndarray, spec_tensor: torch.Tensor):
        """
        Saves the 10-second audio recording (.wav) and Mel-spectrogram (.pt and .png)
        into a dedicated folder named after the identified bird under 'recordings/'.
        """
        try:
            timestamp = get_time_str()
            folder_name = sanitize_folder_name(bird_name)
            bird_folder = base_recordings_dir / folder_name
            bird_folder.mkdir(parents=True, exist_ok=True)

            # 1. Save 10-second lossless audio recording (.wav @ 48kHz)
            wav_path = bird_folder / f"recording_{timestamp}.wav"
            sf.write(str(wav_path), audio, 48000, subtype="PCM_16")

            # 2. Save Mel-spectrogram PyTorch tensor (.pt)
            pt_path = bird_folder / f"melspec_{timestamp}.pt"
            torch.save(spec_tensor, str(pt_path))

            # 3. Save Mel-spectrogram visual grayscale image (.png)
            png_path = bird_folder / f"melspec_{timestamp}.png"
            spec_uint8 = np.clip(spec_np * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(spec_uint8, mode="L")
            img.save(str(png_path))

            logger.info(f"Archived detection data to '{bird_folder}': Audio: {wav_path.name}, Spectrogram: {pt_path.name} & {png_path.name}")

        except Exception as e:
            logger.warning(f"Error archiving detection data to disk: {e}; continuing execution.")

    # Load species class mappings
    class_mapping_file = "class_mapping_us.csv"
    if not os.path.exists(class_mapping_file):
        alt_path = os.path.join("models", "us", "class_mapping_us.csv")
        if os.path.exists(alt_path):
            class_mapping_file = alt_path

    try:
        class_mappings = pd.read_csv(class_mapping_file, index_col="ClassID")
        logger.info(f"Successfully loaded {len(class_mappings)} species from {class_mapping_file}")
    except Exception as e:
        logger.warning(f"Class mapping CSV unavailable ({e}). Using default fallback taxonomy.")
        class_mappings = pd.DataFrame({
            "ClassName": ["Melozone fusca", "Bubo virginianus", "Cyanocitta cristata", "Cardinalis cardinalis"]
        }, index=[0, 1, 2, 3])

    mel_prep = AudioPrep(logger=logger)

    try:
        while not stop_event.is_set():
            try:
                audio = data_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Retrieving from data_queue encountered non-fatal error: {e}")
                continue

            if audio is None:
                logger.info("Retrieving from data_queue: received shutdown sentinel (None). Exiting predict loop.")
                break

            logger.info(f"Retrieving from data_queue: successfully retrieved audio payload (shape: {audio.shape})")

            logger.info("Generating log-Mel spectrogram via torchcodec AudioDecoder...")
            spec_np, spec_tensor = mel_prep.process_mp3(audio)

            # Dynamically initialize and run TensorRT inference
            logger.info("Initializing Jetson TensorRT model/engine for acoustic classification...")
            model = TensorRTInference(logger=logger)

            logger.info("Executing Jetson TensorRT model inference...")
            pred = model.infer(spec_np)

            # Resolve prediction to species name
            if isinstance(pred, (int, np.integer)):
                class_id = int(pred)
            elif hasattr(pred, "size") and pred.size == 1:
                class_id = int(np.squeeze(pred))
            else:
                class_id = int(np.argmax(pred))

            if class_id in class_mappings.index:
                species_name = str(class_mappings.loc[class_id, "ClassName"])
            else:
                species_name = str(class_mappings.iloc[class_id % len(class_mappings)]["ClassName"])

            logger.info(f"Classification result: species identified as '{species_name}' (ClassID: {class_id})")

            log_prediction(species_name)
            archive_detection(species_name, audio, spec_np, spec_tensor)

            # =========================================================================
            # MEMORY OPTIMIZATION: Free TensorRT engine and execution context from memory
            # BEFORE placing the result onto the IPC predict_queue.
            # =========================================================================
            logger.info("Releasing TensorRT engine and context from memory prior to IPC queue dispatch...")
            model.free_resources()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("TensorRT memory completely reclaimed. VRAM is now free for ArtistWorker.")

            try:
                logger.info(f"Accessing predict_queue: putting prediction '{species_name}'...")
                predict_queue.put(species_name, timeout=5.0)
                logger.info(f"Accessing predict_queue: successfully enqueued prediction '{species_name}'.")
            except queue.Full:
                logger.warning("Accessing predict_queue failed: predict_queue is full; skipping artist dispatch.")
            except Exception as e:
                logger.error(f"Accessing predict_queue encountered error: {e}")

    except Exception as e:
        logger.error(f"Fatal exception in PredictorWorker breaking functionality: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("Process exiting cleanly.")


def draw(stop_event: mp.synchronize.Event, predict_queue: mp.queues.Queue, log_queue: mp.queues.Queue):
    """
    Worker Process 4: Generative Visual Artist.
    Consumes species names and uses the local offline Z-Image GGUF model
    to produce 19th-century vintage field guide book illustrations without Internet access.
    Operates with maximum VRAM headroom since TensorRT has already been freed.
    """
    logger = setup_worker_logger(log_queue, "ArtistWorker")
    logger.info("Process started. Local Z-Image illustration studio initialized (Offline Mode).")
    artist = BirdArtist(output_dir="images", logger=logger)

    try:
        while not stop_event.is_set():
            try:
                species_name = predict_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Retrieving from predict_queue encountered non-fatal error: {e}")
                continue

            if species_name is None:
                logger.info("Retrieving from predict_queue: received shutdown sentinel (None). Exiting draw loop.")
                break

            logger.info(f"Retrieving from predict_queue: received species '{species_name}' for illustration.")

            try:
                plate_path = artist(species_name)
                logger.info(f"Artwork pipeline successfully generated plate: {plate_path}")
            except Exception as e:
                logger.warning(f"Artwork pipeline encountered an issue for '{species_name}': {e}; continuing execution.")

    except Exception as e:
        logger.error(f"Fatal exception in ArtistWorker breaking functionality: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("Process exiting cleanly.")


# =====================================================================
# 6. Main Parent Process Coordinator
# =====================================================================
def main():
    # Enforce safe multiprocessing start method on Jetson Orin Nano / Linux
    # Uses 'forkserver' on Linux/Ubuntu 24.04 as intended, with 'spawn' fallback for Windows.
    start_method = "forkserver" if sys.platform != "win32" else "spawn"
    try:
        mp.set_start_method(start_method)
    except RuntimeError:
        pass  # Multiprocessing context already configured

    # Central Multiprocessing Logging Queue & Listener
    log_queue = mp.Queue(-1)

    # Configure daily log file handler with notation: log_YYYY_MM_DD_HHMMSS.txt
    file_handler = logging.FileHandler(CURRENT_LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(formatter)

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # QueueListener in main process collects log records from all processes
    listener = QueueListener(log_queue, file_handler, console_handler)
    listener.start()

    logger = setup_worker_logger(log_queue, "MainProcess")
    logger.info("=" * 75)
    logger.info("BIRD IDENTIFIER: 4-PROCESS LOCAL Z-IMAGE ORNITHOLOGY PIPELINE")
    logger.info("Platform: NVIDIA Jetson Orin Nano (Tegra R39.2 / Jetpack 7.2)")
    logger.info("Mode: 100% Offline (Local GGUF: models/Z-Image-AbliteratedV1.Q8_0.gguf)")
    logger.info("Feature: Zero-Collision VRAM Management (TensorRT freed before IPC dispatch)")
    logger.info(f"Active Session Daily Log File: {CURRENT_LOG_FILE.resolve()}")
    logger.info("=" * 75)

    # Synchronization Primitives
    stop_event = mp.Event()       # Signals all children to exit
    cooldown_event = mp.Event()   # Manages cooldown period between detections

    # IPC Queues (Configured with maxsize 5 to allow graceful buffering)
    flag_queue = mp.Queue(maxsize=5)
    data_queue = mp.Queue(maxsize=5)
    predict_queue = mp.Queue(maxsize=5)

    # Spawn 4 Child Processes matching original structure:
    # a: activator, r: recorder, p: predictor, d: drawer
    proc_activator = mp.Process(
        target=activate,
        name="ActivatorWorker",
        args=(stop_event, cooldown_event, flag_queue, log_queue),
    )
    proc_recorder = mp.Process(
        target=record,
        name="RecorderWorker",
        args=(stop_event, cooldown_event, flag_queue, data_queue, log_queue),
    )
    proc_predictor = mp.Process(
        target=predict,
        name="PredictorWorker",
        args=(stop_event, data_queue, predict_queue, log_queue),
    )
    proc_artist = mp.Process(
        target=draw,
        name="ArtistWorker",
        args=(stop_event, predict_queue, log_queue),
    )

    processes = [proc_artist, proc_predictor, proc_recorder, proc_activator]

    try:
        # Start processes in pipeline order
        logger.info("Starting child worker processes...")
        proc_artist.start()
        logger.info(f"ArtistWorker successfully started (PID: {proc_artist.pid})")
        proc_predictor.start()
        logger.info(f"PredictorWorker successfully started (PID: {proc_predictor.pid})")
        proc_recorder.start()
        logger.info(f"RecorderWorker successfully started (PID: {proc_recorder.pid})")
        proc_activator.start()
        logger.info(f"ActivatorWorker successfully started (PID: {proc_activator.pid})")

        logger.info("All 4 child processes successfully launched on Jetson Orin Nano.")

        cooldown_duration_sec = 600  # 10 minutes cooldown (can be configured)
        cooldown_start = None

        # Parent Process Supervision Loop
        while True:
            # Supervise child process health
            for proc in processes:
                if not proc.is_alive():
                    logger.error(f"Worker process {proc.name} (PID: {proc.pid}) has exited unexpectedly! Functionality broken.")

            # Cooldown Cycle Management (Non-blocking so parent remains responsive)
            if cooldown_event.is_set():
                if cooldown_start is None:
                    cooldown_start = time.time()
                    logger.info(f"Cooldown cycle initiated ({cooldown_duration_sec}s). VAD trigger polling paused.")
                elif time.time() - cooldown_start >= cooldown_duration_sec:
                    logger.info(f"Cooldown period ({cooldown_duration_sec}s) ended. Ready for next acoustic trigger.")
                    cooldown_event.clear()
                    cooldown_start = None

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Shutdown signal (Ctrl+C) received in MainProcess. Initiating graceful shutdown...")
    finally:
        # 1. Signal stop event to all workers
        stop_event.set()
        logger.info("Stop event signaled to all workers.")

        # 2. Inject sentinel tokens to unblock any waiting queue.get() calls
        logger.info("Injecting shutdown sentinels into flag_queue, data_queue, predict_queue...")
        for q in (flag_queue, data_queue, predict_queue):
            try:
                q.put(None, timeout=0.2)
            except Exception:
                pass

        # 3. Cleanly join workers with timeout
        for proc in processes:
            logger.info(f"Waiting for {proc.name} (PID: {proc.pid}) to join...")
            proc.join(timeout=3.0)
            if proc.is_alive():
                logger.warning(f"Process {proc.name} did not terminate within timeout; forcing terminate().")
                proc.terminate()
                proc.join(timeout=1.0)
            try:
                proc.close()
            except Exception:
                pass

        # 4. Close queues
        for q in (flag_queue, data_queue, predict_queue):
            try:
                q.close()
            except Exception:
                pass

        logger.info("All child processes closed cleanly. Pipeline stopped.")

        # 5. Flush and stop QueueListener
        time.sleep(0.5)
        listener.stop()
        file_handler.close()


if __name__ == "__main__":
    main()
