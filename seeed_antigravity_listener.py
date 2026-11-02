"""
seeed_antigravity_listener.py
=============================================================================
Acoustic Detection, Multiprocessing Voice Recording, and Google Antigravity
Picture Generation Pipeline for NVIDIA Jetson Orin Nano.

Target Platform & Hardware:
---------------------------
- Platform: NVIDIA Jetson Orin Nano
- OS / BSP: Jetpack 7.2 / Jetson Linux R39.2 (Ubuntu 24.04 LTS Noble Numbat)
- Hardware: Seeed Studio USB Microphone (VendorID: 0x2886, ProductID: 0x001A)
            (e.g., ReSpeaker XVF3800 / 4-Mic USB Array)

System Architecture (Multiprocessing):
--------------------------------------
1. PARENT PROCESS:
   - Identifies and binds to the Seeed USB microphone (0x2886:0x001A) via USB
     and ALSA / sounddevice.
   - Monitors for audio/voice activity using hardware VAD (Voice Activity Detection)
     and real-time acoustic energy (RMS) thresholding.
   - Upon activity, records the user's speech until silence is reached.
   - Dispatches the recorded audio to the child process via an IPC Queue.
   - TIMEOUT PERIOD: The parent process enters a timeout state where the microphone
     stops listening completely and remains timed out UNTIL the child process saves
     the image into 'AgyPics'.
   - Once the image is saved, resumes microphone listening.

2. CHILD PROCESS:
   - Receives audio recordings asynchronously from the IPC Queue.
   - Prompts Google Antigravity to execute the recording as a prompt (supporting
     both the native Google Antigravity Python SDK and the 'agy' CLI).
   - Inspects Antigravity's response:
     * ONLY if Antigravity responds with a picture (image bytes, base64 image data,
       or generated image file), saves it in the 'AgyPics' folder and signals
       the parent process that the image is saved.
     * IGNORES any other response (text, code, explanations, etc.).
   - Cleans up temporary audio files.

Ubuntu 24.04 / Jetson Setup Note:
---------------------------------
For non-root USB access to the Seeed microphone on Ubuntu 24.04, install udev rule:
  echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="001a", MODE="0666", GROUP="audio"' | \\
    sudo tee /etc/udev/rules.d/99-respeaker.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
=============================================================================
"""

import os
import sys
import time
import datetime
import re
import base64
import shutil
import signal
import subprocess
import json
import traceback
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import sounddevice as sd
import soundfile as sf
import multiprocessing as mp
import multiprocessing.synchronize
import multiprocessing.queues

# Optional USB hardware control via pyusb / libusb
try:
    import usb.core
    import usb.util
    HAS_PYUSB = True
except ImportError:
    HAS_PYUSB = False

# Optional Google Antigravity Python SDK
try:
    import asyncio
    import google.antigravity
    from google.antigravity import Agent, LocalAgentConfig
    from google.antigravity.types import Audio
    HAS_AGY_SDK = True
except ImportError:
    HAS_AGY_SDK = False

# Optional SpeechRecognition for CLI text-prompt fallback
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False


# =============================================================================
# Constants & Configuration
# =============================================================================
SEEED_VENDOR_ID = 0x2886
SEEED_PRODUCT_ID = 0x001A

SAMPLE_RATE = 48000          # 16 kHz standard for voice/speech AI prompts
CHANNELS = 1                 # Mono channel recording
SUBTYPE = "PCM_16"           # 16-bit linear PCM
CHUNK_SIZE = 1024            # ~64ms buffer chunk at 16kHz
SILENCE_DURATION = 1.2       # Seconds of silence to trigger end-of-utterance
MIN_RECORD_DURATION = 1.0    # Minimum recording length in seconds
MAX_RECORD_DURATION = 20.0   # Maximum recording limit safeguard
COOLDOWN_TIMEOUT = 5.0       # Mandatory 5-second silence period after recording

OUTPUT_PICS_DIR = "AgyPics"
TEMP_AUDIO_DIR = "temp_recordings"


# =============================================================================
# Helper Utilities
# =============================================================================
def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current timestamp formatted as string."""
    return datetime.datetime.now().strftime(fmt)


def compute_rms(audio_chunk: np.ndarray) -> float:
    """Computes the Root Mean Square (RMS) energy of an audio buffer."""
    if len(audio_chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float64)))))


# =============================================================================
# Hardware USB & ALSA Discovery for Seeed Microphone (0x2886:0x001A)
# =============================================================================
class SeeedHardwareVAD:
    """
    Direct interface to Seeed Studio USB Microphone Array over USB (pyusb).
    Queries the hardware Voice Activity Detection (VAD) register on the device.
    """
    TIMEOUT_MS = 3000

    def __init__(self, dev: "usb.core.Device"):
        self.dev = dev

    def read_vad(self) -> bool:
        """
        Query Voice Activity Detection (VAD) register on Seeed ReSpeaker XVF3800.
        Returns True when voice activity is detected by the onboard DSP.
        """
        if self.dev is None:
            return False
        try:
            # XVF3800 VAD control transfer packet: resource ID 20, command ID 0x80 | 18
            resid = 20
            cmdid = 0x80 | 18
            length = 5  # 4 bytes payload + 1 byte status

            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmdid, resid, length, self.TIMEOUT_MS
            )
            result = response.tolist()
            if len(result) > 3:
                return result[3] == 1
            return False
        except Exception:
            return False

    def close(self):
        """Release USB device resources cleanly."""
        if self.dev is not None and HAS_PYUSB:
            try:
                usb.util.dispose_resources(self.dev)
            except Exception:
                pass


def find_seeed_usb_device(vid: int = SEEED_VENDOR_ID, pid: int = SEEED_PRODUCT_ID) -> Optional[SeeedHardwareVAD]:
    """Finds and initializes the Seeed USB device using pyusb."""
    if not HAS_PYUSB:
        return None
    try:
        dev = usb.core.find(idVendor=vid, idProduct=pid)
        if dev is not None:
            return SeeedHardwareVAD(dev)
    except Exception as e:
        print(f"[Parent] Note on USB VAD interface: {e}")
    return None


def find_seeed_audio_device_index(vid: int = SEEED_VENDOR_ID, pid: int = SEEED_PRODUCT_ID) -> Optional[int]:
    """
    Locates the sounddevice / PortAudio device index corresponding to
    the Seeed USB microphone (VendorID 0x2886, ProductID 0x001A).
    Inspects Linux sysfs (/sys/class/sound/card*) and queries sounddevice.
    """
    # 1. Inspect Linux sysfs to match USB Vendor and Product IDs directly
    matched_card_id = None
    vid_hex = f"{vid:04x}".lower()
    pid_hex = f"{pid:04x}".lower()

    sound_cards_dir = Path("/sys/class/sound")
    if sound_cards_dir.exists():
        for card_path in sound_cards_dir.glob("card[0-9]*"):
            try:
                # Traverse parent sysfs directories looking for USB idVendor & idProduct
                curr = card_path.resolve()
                found_match = False
                for _ in range(6):
                    curr = curr.parent
                    v_file = curr / "idVendor"
                    p_file = curr / "idProduct"
                    if v_file.exists() and p_file.exists():
                        dev_vid = v_file.read_text().strip().lower()
                        dev_pid = p_file.read_text().strip().lower()
                        if dev_vid == vid_hex and dev_pid == pid_hex:
                            found_match = True
                            break

                if found_match:
                    id_file = card_path / "id"
                    if id_file.exists():
                        matched_card_id = id_file.read_text().strip()
                        break
            except Exception:
                continue

    # 2. Match with sounddevice / PortAudio host API devices
    try:
        devices = sd.query_devices()
        # Pass 1: exact ALSA card ID match (e.g. "ArrayUAC10", "ReSpeaker")
        if matched_card_id:
            for idx, dev_info in enumerate(devices):
                if dev_info.get("max_input_channels", 0) > 0 and matched_card_id in dev_info.get("name", ""):
                    print(f"[Parent] Bound to Seeed USB Mic via ALSA card '{matched_card_id}' [Index {idx}]: {dev_info['name']}")
                    return idx

        # Pass 2: Name heuristic search for Seeed ReSpeaker / ArrayUAC
        for idx, dev_info in enumerate(devices):
            if dev_info.get("max_input_channels", 0) > 0:
                name_lower = dev_info.get("name", "").lower()
                if any(k in name_lower for k in ("respeaker", "seeed", "arrayuac10", "xvf3800")):
                    print(f"[Parent] Bound to Seeed USB Mic [Index {idx}]: {dev_info['name']}")
                    return idx

        # Pass 3: Fallback to system default input device
        default_idx = sd.default.device[0]
        if default_idx is not None and default_idx >= 0:
            print(f"[Parent] Using default audio input device [Index {default_idx}]: {devices[default_idx]['name']}")
            return int(default_idx)

    except Exception as e:
        print(f"[Parent] Audio device query error: {e}")

    return None


# =============================================================================
# Child Process: Prompt Google Antigravity & Save Picture to AgyPics
# =============================================================================
class AntigravityPrompter:
    """
    Coordinates prompting Google Antigravity with audio recordings and
    saving picture responses into the 'AgyPics' directory.
    """

    def __init__(self, output_dir: str = OUTPUT_PICS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agy_cli_path = self._locate_agy_cli()

    @staticmethod
    def _locate_agy_cli() -> Optional[str]:
        """Finds the 'agy' CLI executable in PATH or standard user directories."""
        found = shutil.which("agy")
        if found:
            return found
        # Common installation paths on Linux and Jetson
        home = Path.home()
        candidates = [
            home / ".local" / "bin" / "agy",
            home / ".gemini" / "antigravity-cli" / "bin" / "agy",
            Path("/usr/local/bin/agy"),
            Path("/usr/bin/agy"),
            home / "AppData" / "Local" / "agy" / "bin" / "agy.exe",  # Windows dev support
        ]
        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                return str(c)
        return None

    def execute_prompt(self, audio_path: Path) -> Optional[Path]:
        """
        Executes the recording as a prompt with Google Antigravity.
        Only saves a picture into 'AgyPics' if Antigravity responds with one.
        Ignores all other responses.
        Returns the saved image Path if an image was saved, or None otherwise.
        """
        print(f"\n[Child] Processing recording: {audio_path.name}")

        # 1. Try Google Antigravity Python SDK if installed
        if HAS_AGY_SDK:
            try:
                saved_path = asyncio.run(self._prompt_via_sdk(audio_path))
                if saved_path:
                    return saved_path
            except Exception as e:
                print(f"[Child] SDK execution error: {e}; falling back to CLI interface.")

        # 2. Try Antigravity CLI ('agy')
        if self.agy_cli_path:
            return self._prompt_via_cli(audio_path)
        else:
            print("[Child] Error: Neither 'google.antigravity' SDK nor 'agy' CLI was found.")
            return None

    async def _prompt_via_sdk(self, audio_path: Path) -> Optional[Path]:
        """Invokes Google Antigravity via the official Python SDK."""
        print("[Child] Invoking Google Antigravity Python SDK...")
        config = LocalAgentConfig()
        audio_prompt = Audio.from_file(str(audio_path.resolve()))

        async with Agent(config) as agent:
            # Execute the recording as a prompt
            response = await agent.chat([
                audio_prompt,
                "Execute the user instructions contained in the audio prompt. "
                "If the prompt asks for an image, drawing, or picture, generate it."
            ])

            # Check if Antigravity responded with a picture
            saved_picture_path = self._extract_and_save_picture(response)
            if not saved_picture_path:
                print("[Child] Antigravity did not respond with a picture. Ignoring response.")
            return saved_picture_path

    def _prompt_via_cli(self, audio_path: Path) -> Optional[Path]:
        """Invokes Google Antigravity via the 'agy' CLI executable."""
        print(f"[Child] Invoking Antigravity CLI at: {self.agy_cli_path}")

        # Transcribe audio for the CLI prompt if speech recognition is available
        transcribed_text = None
        if HAS_SPEECH_RECOGNITION:
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(str(audio_path)) as source:
                    audio_data = recognizer.record(source)
                    transcribed_text = recognizer.recognize_google(audio_data)
                    print(f"[Child] Transcribed Audio Prompt: \"{transcribed_text}\"")
            except Exception:
                transcribed_text = None

        if transcribed_text:
            prompt_str = (
                f"{transcribed_text}. "
                f"If this instruction requests a drawing, illustration, or picture, "
                f"generate the image and return it."
            )
        else:
            # Multimodal file reference prompt for agy
            prompt_str = (
                f"Listen to the recording at @{audio_path.resolve()} and execute it as a prompt. "
                f"If the prompt requests an image, illustration, or picture, return the picture."
            )

        cmd = [
            self.agy_cli_path,
            "-p", prompt_str,
            "--output-format", "json"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"[Child] Antigravity CLI warning: {result.stderr.strip()[:200]}")

            output_text = result.stdout.strip()
            # Parse JSON or raw text output to detect picture
            saved_path = self._parse_cli_output_for_picture(output_text)
            if not saved_path:
                print("[Child] Antigravity response was not a picture. Ignoring response.")
            return saved_path

        except subprocess.TimeoutExpired:
            print("[Child] Timed out waiting for Antigravity response.")
        except Exception as e:
            print(f"[Child] Error invoking Antigravity CLI: {e}")
        return None

    def _extract_and_save_picture(self, response) -> Optional[Path]:
        """Inspects an SDK response object for picture content and saves it."""
        timestamp = get_timestamp()

        # 1. Check for response.images
        images = getattr(response, "images", None)
        if images and len(images) > 0:
            first_saved = None
            for idx, img in enumerate(images):
                target_file = self.output_dir / f"pic_{timestamp}_{idx}.png"
                if hasattr(img, "save"):
                    img.save(str(target_file))
                elif isinstance(img, (bytes, bytearray)):
                    target_file.write_bytes(img)
                print(f"[Child] Antigravity responded with a picture! Saved to: {target_file}")
                if first_saved is None:
                    first_saved = target_file
            return first_saved

        # 2. Check for response.artifacts with image MIME types
        artifacts = getattr(response, "artifacts", None)
        if artifacts:
            for art in artifacts:
                mime = getattr(art, "mime_type", "")
                if mime.startswith("image/") or getattr(art, "name", "").lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    data = getattr(art, "data", None)
                    if data:
                        target_file = self.output_dir / f"pic_{timestamp}.png"
                        target_file.write_bytes(data)
                        print(f"[Child] Antigravity responded with an image artifact! Saved to: {target_file}")
                        return target_file

        # 3. Check for base64 or file references in text response
        text = str(response)
        return self._save_picture_from_text(text, timestamp)

    def _parse_cli_output_for_picture(self, output: str) -> Optional[Path]:
        """Parses CLI output (JSON or text) for picture data and saves it."""
        timestamp = get_timestamp()
        if not output:
            return None

        # Try parsing structured JSON output
        try:
            data = json.loads(output)
            # Check for response field
            response_content = data.get("response", "")
            if isinstance(response_content, str):
                saved = self._save_picture_from_text(response_content, timestamp)
                if saved:
                    return saved

            # Check for artifacts or media lists in JSON
            for key in ("images", "media", "artifacts"):
                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                    for item in data[key]:
                        if isinstance(item, dict) and "data" in item:
                            b64_data = item["data"]
                            img_bytes = base64.b64decode(b64_data)
                            target_file = self.output_dir / f"pic_{timestamp}.png"
                            target_file.write_bytes(img_bytes)
                            print(f"[Child] Saved picture from JSON artifact to: {target_file}")
                            return target_file
        except json.JSONDecodeError:
            pass

        # Parse plain text output
        return self._save_picture_from_text(output, timestamp)

    def _save_picture_from_text(self, text: str, timestamp: str) -> Optional[Path]:
        """Detects base64 encoded images or generated image file paths in text."""
        # 1. Detect base64 data URI: data:image/(png|jpeg|webp);base64,<data>
        b64_match = re.search(r"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)", text)
        if b64_match:
            ext = b64_match.group(1).replace("jpeg", "jpg")
            raw_b64 = b64_match.group(2)
            try:
                img_bytes = base64.b64decode(raw_b64)
                target_file = self.output_dir / f"pic_{timestamp}.{ext}"
                target_file.write_bytes(img_bytes)
                print(f"[Child] Antigravity returned a base64 picture! Saved to: {target_file}")
                return target_file
            except Exception as e:
                print(f"[Child] Error decoding base64 image: {e}")

        # 2. Detect markdown image links: ![alt](path/to/image.png)
        md_match = re.search(r"!\[.*?\]\((.*?\.(png|jpg|jpeg|webp))\)", text, re.IGNORECASE)
        if md_match:
            img_path = Path(md_match.group(1).strip())
            if img_path.exists() and self._is_image_file(img_path):
                target_file = self.output_dir / f"pic_{timestamp}{img_path.suffix}"
                shutil.copy2(img_path, target_file)
                print(f"[Child] Antigravity generated an image file! Saved to: {target_file}")
                return target_file

        # 3. Detect standalone image file path mentioned in text
        path_matches = re.findall(r"([A-Za-z0-9_\-./\\]+\.(png|jpg|jpeg|webp))", text, re.IGNORECASE)
        for candidate_str, ext in path_matches:
            candidate_path = Path(candidate_str.strip())
            if candidate_path.exists() and self._is_image_file(candidate_path):
                target_file = self.output_dir / f"pic_{timestamp}.{ext}"
                shutil.copy2(candidate_path, target_file)
                print(f"[Child] Antigravity generated image at '{candidate_path}'. Saved to: {target_file}")
                return target_file

        return None

    @staticmethod
    def _is_image_file(file_path: Path) -> bool:
        """Verifies if a file is an actual image using header magic bytes."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)
            # PNG header: \x89PNG\r\n\x1a\n
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return True
            # JPEG header: \xff\xd8\xff
            if header.startswith(b"\xff\xd8\xff"):
                return True
            # WEBP header: RIFF....WEBP
            if header.startswith(b"RIFF") and b"WEBP" in header:
                return True
        except Exception:
            pass
        return False


def child_antigravity_worker(
    audio_queue: mp.queues.Queue,
    status_queue: mp.queues.Queue,
    stop_event: mp.synchronize.Event,
    output_dir: str = OUTPUT_PICS_DIR
):
    """
    Child Process Worker.
    Listens on audio_queue for completed recordings, executes them as
    prompts in Google Antigravity, and saves any picture response to 'AgyPics'.
    Notifies the parent process via status_queue so the parent process
    remains timed out until the image is saved.
    """
    print(f"[Child] Process started (PID: {os.getpid()}). Initializing Antigravity Prompter...")
    prompter = AntigravityPrompter(output_dir=output_dir)
    print(f"[Child] Ready. Awaiting audio recordings from parent process...")

    while not stop_event.is_set():
        try:
            audio_path_str = audio_queue.get(timeout=1.0)
        except Exception:
            continue

        if audio_path_str is None:
            # Sentinel token for clean shutdown
            break

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            status_queue.put({"status": "error", "error": f"Audio file not found: {audio_path_str}"})
            continue

        try:
            saved_picture_path = prompter.execute_prompt(audio_path)
            if saved_picture_path:
                # Notify parent that picture was successfully saved
                status_queue.put({"status": "image_saved", "path": str(saved_picture_path)})
            else:
                # Prompt finished with no picture (ignored per requirement)
                status_queue.put({"status": "no_image"})
        except Exception as e:
            print(f"[Child] Unexpected error handling prompt: {e}")
            traceback.print_exc()
            status_queue.put({"status": "error", "error": str(e)})
        finally:
            # Remove temporary audio file after processing
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except Exception:
                pass

    print("[Child] Antigravity worker process exiting cleanly.")


# =============================================================================
# Parent Process: Microphone Listener, Voice Activity Detector & Recorder
# =============================================================================
class ParentMicListener:
    """
    Manages Seeed USB Microphone listening, activity detection, utterance recording,
    and timeout synchronization until the child process saves the image in AgyPics.
    """

    def __init__(
        self,
        audio_queue: mp.queues.Queue,
        status_queue: mp.queues.Queue,
        stop_event: mp.synchronize.Event,
        temp_dir: str = TEMP_AUDIO_DIR
    ):
        self.audio_queue = audio_queue
        self.status_queue = status_queue
        self.stop_event = stop_event
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Hardware connection
        print("[Parent] Initializing Seeed USB Microphone (VID 0x2886, PID 0x001A)...")
        self.usb_vad = find_seeed_usb_device(SEEED_VENDOR_ID, SEEED_PRODUCT_ID)
        if self.usb_vad:
            print("[Parent] Connected to Seeed hardware VAD control register.")
        else:
            print("[Parent] Hardware VAD control transfer unavailable; using acoustic RMS energy detector.")

        self.audio_dev_idx = find_seeed_audio_device_index(SEEED_VENDOR_ID, SEEED_PRODUCT_ID)

    def run_listening_loop(self):
        """Main listening loop executed by the parent process."""
        print("\n" + "=" * 75)
        print("   SEEED MICROPHONE LISTENER & GOOGLE ANTIGRAVITY PIPELINE")
        print("   Platform: NVIDIA Jetson Orin Nano (Jetpack 7.2 / Tegra R39.2)")
        print("=" * 75)
        print(f"[Parent] Audio Config: {SAMPLE_RATE}Hz, Channels={CHANNELS}")
        print("[Parent] Listening for acoustic activity (Press Ctrl+C to stop)...\n")

        # Initial baseline noise calibration
        ambient_rms = self._calibrate_ambient_noise()
        rms_threshold = max(0.015, ambient_rms * 2.8)
        print(f"[Parent] Ambient RMS: {ambient_rms:.4f} | Dynamic Activity Threshold: {rms_threshold:.4f}")

        while not self.stop_event.is_set():
            # 1. Listen until activity is detected
            activity_detected = self._wait_for_activity(rms_threshold)
            if not activity_detected or self.stop_event.is_set():
                break

            print("\n[Parent] Acoustic activity detected! Recording audio prompt...")

            # 2. Record audio until user finishes speaking (silence detected)
            audio_data = self._record_utterance(rms_threshold)

            duration = len(audio_data) / SAMPLE_RATE
            if duration >= MIN_RECORD_DURATION:
                # Save recorded utterance to WAV
                timestamp = get_timestamp()
                wav_file = self.temp_dir / f"prompt_{timestamp}.wav"
                sf.write(str(wav_file), audio_data, SAMPLE_RATE, subtype=SUBTYPE)
                print(f"[Parent] Recording captured ({duration:.1f}s) -> Dispatched to Antigravity worker.")

                # Dispatch to child process
                try:
                    self.audio_queue.put(str(wav_file), timeout=2.0)

                    # 3. Timeout period: The parent microphone stops listening and
                    #    remains timed out until the child process saves the image!
                    self._wait_until_image_saved()
                except Exception as e:
                    print(f"[Parent] Queue put error: {e}")
            else:
                print(f"[Parent] Discarded brief noise trigger ({duration:.2f}s < {MIN_RECORD_DURATION}s).")
                time.sleep(1.0)

        self._cleanup()

    def _wait_until_image_saved(self):
        """
        Enforces the timeout period in the parent process.
        The parent microphone does NOT listen for audio and remains timed out
        until the child process saves the image in AgyPics.
        """
        print("[Parent] >>> Mic is now TIMED OUT (not listening) until child process saves the image... <<<")
        start_wait = time.time()
        while not self.stop_event.is_set():
            try:
                status_msg = self.status_queue.get(timeout=0.5)
            except Exception:
                continue

            status = status_msg.get("status")
            elapsed = time.time() - start_wait

            if status == "image_saved":
                img_path = status_msg.get("path", "AgyPics")
                print(f"[Parent] >>> Child process saved image to '{img_path}' (after {elapsed:.1f}s)! <<<")
                if COOLDOWN_TIMEOUT > 0:
                    print(f"[Parent] Enforcing {COOLDOWN_TIMEOUT:.0f}s post-save cooldown before listening...")
                    self._enforce_cooldown(COOLDOWN_TIMEOUT)
                print("[Parent] Timeout period ended. Resuming microphone listening...\n")
                return True
            elif status in ("no_image", "error"):
                reason = status_msg.get("error", "No picture in response (ignored)")
                print(f"[Parent] Child finished turn ({reason}) after {elapsed:.1f}s.")
                if COOLDOWN_TIMEOUT > 0:
                    self._enforce_cooldown(COOLDOWN_TIMEOUT)
                print("[Parent] Resuming microphone listening...\n")
                return False

        return False

    def _calibrate_ambient_noise(self, calibration_seconds: float = 1.0) -> float:
        """Records a short sample to establish ambient room noise floor."""
        try:
            frames = int(SAMPLE_RATE * calibration_seconds)
            data = sd.rec(
                frames=frames,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocking=True,
                device=self.audio_dev_idx,
                dtype="float32"
            )
            return compute_rms(data)
        except Exception as e:
            print(f"[Parent] Calibration note: {e}")
            return 0.005

    def _check_hardware_vad(self) -> bool:
        """Queries the hardware VAD register if pyusb device is available."""
        if self.usb_vad:
            return self.usb_vad.read_vad()
        return False

    def _wait_for_activity(self, rms_threshold: float) -> bool:
        """
        Continuously polls the Seeed microphone stream and hardware VAD
        until voice or sound activity exceeds the threshold.
        """
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=CHUNK_SIZE,
                device=self.audio_dev_idx,
                dtype="float32"
            ) as stream:
                while not self.stop_event.is_set():
                    chunk, overflowed = stream.read(CHUNK_SIZE)
                    rms = compute_rms(chunk)
                    hw_vad = self._check_hardware_vad()

                    if hw_vad or (rms > rms_threshold):
                        return True

                    time.sleep(0.01)
        except Exception as e:
            if not self.stop_event.is_set():
                print(f"[Parent] Audio input stream error: {e}")
                time.sleep(1.0)
        return False

    def _record_utterance(self, rms_threshold: float) -> np.ndarray:
        """
        Records the active utterance until silence is maintained for
        SILENCE_DURATION or MAX_RECORD_DURATION is reached.
        """
        recorded_chunks = []
        silence_chunks_count = 0
        silence_limit_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        max_chunks = int(MAX_RECORD_DURATION * SAMPLE_RATE / CHUNK_SIZE)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=CHUNK_SIZE,
                device=self.audio_dev_idx,
                dtype="float32"
            ) as stream:
                for _ in range(max_chunks):
                    if self.stop_event.is_set():
                        break

                    chunk, _ = stream.read(CHUNK_SIZE)
                    recorded_chunks.append(chunk)

                    rms = compute_rms(chunk)
                    hw_vad = self._check_hardware_vad()

                    if hw_vad or (rms > rms_threshold * 0.7):
                        silence_chunks_count = 0
                    else:
                        silence_chunks_count += 1

                    if silence_chunks_count >= silence_limit_chunks:
                        # User finished speaking
                        break
        except Exception as e:
            print(f"[Parent] Recording capture error: {e}")

        if recorded_chunks:
            return np.concatenate(recorded_chunks, axis=0)
        return np.zeros((0, CHANNELS), dtype=np.float32)

    def _enforce_cooldown(self, timeout_seconds: float):
        """
        Enforces a cooldown period where the microphone
        does not listen for audio, ensuring clean separation between prompts.
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds and not self.stop_event.is_set():
            time.sleep(0.2)

    def _cleanup(self):
        """Cleanly releases hardware interfaces."""
        if self.usb_vad:
            self.usb_vad.close()
        print("[Parent] Resources released.")


# =============================================================================
# Main Process Entry Point
# =============================================================================
def main():
    # Enforce safe multiprocessing start method on Linux/Ubuntu 24.04 (Jetpack 7.2)
    start_method = "forkserver" if sys.platform != "win32" else "spawn"
    try:
        mp.set_start_method(start_method)
    except RuntimeError:
        pass

    # IPC Synchronization & Communication
    stop_event = mp.Event()
    audio_queue = mp.Queue()
    status_queue = mp.Queue()

    # Graceful signal handling for Ctrl+C (SIGINT) and termination (SIGTERM)
    def signal_handler(signum, frame):
        print("\n[Shutdown] Termination signal received. Stopping all processes cleanly...")
        stop_event.set()
        try:
            audio_queue.put(None)
        except Exception:
            pass

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Spawn Child Process: Prompts Google Antigravity & Saves Pictures to AgyPics
    child_process = mp.Process(
        target=child_antigravity_worker,
        name="AntigravityChildWorker",
        args=(audio_queue, status_queue, stop_event, OUTPUT_PICS_DIR),
    )
    child_process.daemon = True
    child_process.start()

    # Run Parent Process: Listens on Seeed mic, records, and dispatches
    parent = ParentMicListener(
        audio_queue=audio_queue,
        status_queue=status_queue,
        stop_event=stop_event
    )
    try:
        parent.run_listening_loop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            audio_queue.put(None)
        except Exception:
            pass

        print("[Parent] Waiting for child process to terminate cleanly...")
        child_process.join(timeout=5.0)
        if child_process.is_alive():
            child_process.terminate()

        print("[Parent] Shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()
