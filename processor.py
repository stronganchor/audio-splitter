# processor.py
# Core audio processing: ffmpeg preprocessing, silence detection, speech segment
# inversion/merging, boundary snapping to local minima, and optional transcription.

import os
import re
import math
import wave
import atexit
import shutil
import tempfile
import subprocess
import contextlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# --- Optional deps (both are optional and the code degrades gracefully) ---
try:
    import assemblyai as aai  # pip install assemblyai
except Exception:  # pragma: no cover
    aai = None

try:
    from unidecode import unidecode  # pip install Unidecode
except Exception:  # pragma: no cover
    def unidecode(s: str) -> str:
        return s


# --------------------------- Defaults / Tunables ---------------------------

LESS_NOISE_REDUCTION_LEVEL = -50  # dB for afftdn noise floor
MORE_NOISE_REDUCTION_LEVEL = -30  # dB for afftdn noise floor
DEFAULT_TARGET_LUFS = -18.0       # loudnorm target

# Boundary snapping window (seconds) used when aligning to local RMS minima
SNAP_WINDOW = 0.12


# ------------------------------- Data Types -------------------------------

@dataclass
class TranscriptWord:
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class TranscriptData:
    words: List[TranscriptWord]


# ------------------------------- Utilities --------------------------------

def ffmpeg_exists() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except Exception:
        return False


def ffplay_available() -> bool:
    """Return True if ffplay (for playback) is available on PATH."""
    try:
        subprocess.run(
            ["ffplay", "-hide_banner", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except Exception:
        return False


# ------------------------------ Core Engine --------------------------------

class AudioProcessor:
    """
    Handles:
      - Preprocessing via ffmpeg: noise reduction (afftdn) + loudness normalization (loudnorm)
      - Wave loading and RMS envelope calculation
      - Silence detection (ffmpeg silencedetect OR RMS energy with hysteresis)
      - Converting silences -> speech segments and merging tiny segments
      - Snapping segment boundaries to local RMS minima
      - Optional transcription with AssemblyAI (if ASSEMBLYAI_API_KEY is set)
    """

    def __init__(self) -> None:
        self.input_path: Optional[str] = None
        self.temp_dir = tempfile.mkdtemp(prefix="audio_split_ui_")
        atexit.register(self.cleanup)

        # Results of processing
        self.processed_wav: Optional[str] = None  # mono, 48k, pcm_s16le
        self.sample_rate: Optional[int] = None
        self.samples: Optional[np.ndarray] = None  # mono float32 [-1..1]
        self.duration: float = 0.0

        # RMS envelope (used by energy detector + snapping)
        self.rms_db: Optional[np.ndarray] = None
        self.rms_hop_sec: float = 0.010  # hop size (s)

        # Optional transcript
        self.transcript: Optional[TranscriptData] = None

    # -- lifecycle --

    def cleanup(self) -> None:
        try:
            if os.path.isdir(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    # -- preprocessing --

    def load_and_process(self, path: str, use_more_noise: bool, target_lufs: float) -> None:
        """
        Run ffmpeg with noise reduction + loudnorm, convert to mono 48k WAV.
        Populates: processed_wav, sample_rate, samples, duration, rms_db.
        """
        if not ffmpeg_exists():
            raise RuntimeError("ffmpeg was not found in PATH.")

        self.input_path = path
        nf = MORE_NOISE_REDUCTION_LEVEL if use_more_noise else LESS_NOISE_REDUCTION_LEVEL
        out_wav = os.path.join(self.temp_dir, "processed.wav")

        # afftdn: frequency-domain denoiser; loudnorm: EBU R128 loudness normalization
        filter_chain = f"afftdn=nf={nf},loudnorm=I={target_lufs}:TP=-2:LRA=11"
        cmd = [
            "ffmpeg", "-y", "-i", path, "-af", filter_chain,
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", out_wav
        ]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode != 0:
            raise RuntimeError("ffmpeg processing failed.")

        self.processed_wav = out_wav
        self._load_wav(out_wav)
        self._compute_rms_db()

    def _load_wav(self, wav_path: str) -> None:
        with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            n = wf.getnframes()
            self.sample_rate = sr
            self.duration = n / float(sr)

            frames = wf.readframes(n)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if ch == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)  # downmix to mono
            self.samples = audio

    def _compute_rms_db(self) -> None:
        """Compute a smoothed RMS envelope in dBFS (~10 ms hop, ~50 ms smoothing)."""
        if self.samples is None or self.sample_rate is None:
            return

        x = self.samples
        sr = self.sample_rate

        win_sec = 0.020
        hop_sec = 0.010
        win = max(1, int(sr * win_sec))
        hop = max(1, int(sr * hop_sec))

        pad = (hop - (len(x) - win) % hop) % hop
        xpad = np.pad(x, (0, pad), mode='constant')
        n_frames = 1 + (len(xpad) - win) // hop

        frames = np.lib.stride_tricks.as_strided(
            xpad,
            shape=(n_frames, win),
            strides=(xpad.strides[0] * hop, xpad.strides[0])
        )
        rms = np.sqrt(np.maximum(1e-12, np.mean(frames * frames, axis=1)))
        dbfs = 20.0 * np.log10(rms + 1e-12)

        # smooth (moving average ~50 ms)
        k = max(1, int(round(0.050 / hop_sec)))
        if k > 1:
            kernel = np.ones(k) / k
            dbfs = np.convolve(dbfs, kernel, mode='same')

        self.rms_db = dbfs
        self.rms_hop_sec = hop_sec

    # -- silence detection --

    def detect_silences_ffmpeg(self, threshold_db: int, min_silence: float) -> List[Tuple[float, float]]:
        """
        Use ffmpeg's silencedetect to return a list of (start, end) silence pairs.
        """
        if not self.processed_wav:
            return []

        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", self.processed_wav,
            "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}",
            "-f", "null", "-"
        ]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        text = run.stderr

        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", text)]
        ends   = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", text)]

        pairs: List[Tuple[float, float]] = []
        ei = 0
        for s in starts:
            while ei < len(ends) and ends[ei] < s:
                ei += 1
            if ei < len(ends):
                pairs.append((s, ends[ei]))
                ei += 1
            else:
                # trailing open silence to EOF
                pairs.append((s, self.duration))

        return self._sanitize_pairs(pairs)

    def detect_silences_energy(
        self, threshold_db: int, min_silence: float, hysteresis_db: int
    ) -> List[Tuple[float, float]]:
        """
        Silence detector using the RMS envelope with hysteresis:
          - enter 'speech' when RMS >= (threshold + H/2)
          - exit  'speech' when RMS <  (threshold - H/2) for at least min_silence
        Returns (start, end) silence pairs.
        """
        if self.rms_db is None:
            return []

        db = self.rms_db
        hop = self.rms_hop_sec

        enter = threshold_db + hysteresis_db / 2.0
        exit_ = threshold_db - hysteresis_db / 2.0

        min_sil_frames = max(1, int(round(min_silence / hop)))
        speech = False
        sil_start_idx: Optional[int] = None
        pairs: List[Tuple[float, float]] = []

        i = 0
        while i < len(db):
            d = db[i]
            if not speech:
                if d >= enter:
                    # close a preceding silence (if any)
                    if sil_start_idx is not None:
                        s = sil_start_idx * hop
                        e = i * hop
                        if e > s:
                            pairs.append((s, min(self.duration, e)))
                    sil_start_idx = None
                    speech = True
                else:
                    if sil_start_idx is None:
                        sil_start_idx = i
            else:
                # we're in speech; look for sustained quiet
                if d < exit_:
                    j = i
                    while j < len(db) and db[j] < exit_ and (j - i) < min_sil_frames:
                        j += 1
                    if j - i >= min_sil_frames:
                        s = i * hop
                        e = j * hop
                        pairs.append((s, min(self.duration, e)))
                        speech = False
                        sil_start_idx = j
                        i = j
                        continue
            i += 1

        if not speech and sil_start_idx is not None:
            s = sil_start_idx * hop
            e = self.duration
            if e > s:
                pairs.append((s, e))

        return self._sanitize_pairs(pairs)

    # -- conversion helpers --

    def _sanitize_pairs(self, pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Clip to [0,duration], drop zero/negative spans, and merge overlaps."""
        if not pairs:
            return []
        pairs = [(max(0.0, s), min(self.duration, e)) for s, e in pairs if e > s]
        pairs.sort()
        merged: List[Tuple[float, float]] = []
        cs, ce = pairs[0]
        for s, e in pairs[1:]:
            if s <= ce + 1e-3:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        return merged

    def invert_to_speech_segments(
        self, silences: List[Tuple[float, float]], min_seg: float, min_gap: float
    ) -> List[Tuple[float, float]]:
        """
        Given (start,end) silence spans, return merged speech segments.
        - Silences shorter than min_gap are ignored (not considered real splits).
        - Speech segments shorter than min_seg are merged with the previous one.
        """
        if not silences:
            return [(0.0, self.duration)]

        silences = [(s, e) for s, e in silences if (e - s) >= min_gap]
        silences = self._sanitize_pairs(silences)

        segs: List[Tuple[float, float]] = []
        cur = 0.0
        for s, e in silences:
            if s > cur:
                segs.append((cur, s))
            cur = e
        if cur < self.duration:
            segs.append((cur, self.duration))

        merged: List[Tuple[float, float]] = []
        for a, b in segs:
            if not merged:
                if b - a >= min_seg:
                    merged.append((a, b))
                continue
            pa, pb = merged[-1]
            if (b - a) < min_seg:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))

        return merged if merged else [(0.0, self.duration)]

    def snap_times_to_minima(
        self, segs: List[Tuple[float, float]], window: float = SNAP_WINDOW
    ) -> List[Tuple[float, float]]:
        """
        For each boundary, snap to the nearest local minimum in the RMS curve
        within ±window seconds (helps borders hug actual silence).
        """
        if self.rms_db is None:
            return segs

        hop = self.rms_hop_sec
        db = self.rms_db

        def nearest_min(t: float) -> float:
            i = int(round(t / hop))
            w = int(round(window / hop))
            lo = max(0, i - w)
            hi = min(len(db) - 1, i + w)
            idx = lo + int(np.argmin(db[lo:hi + 1]))
            return idx * hop

        out: List[Tuple[float, float]] = []
        for a, b in segs:
            aa = max(0.0, min(nearest_min(a), self.duration))
            bb = max(0.0, min(nearest_min(b), self.duration))
            # If snapping collapses a very short span, keep originals.
            if bb - aa < 0.05:
                aa, bb = a, b
            out.append((aa, bb))
        return out

    # -- optional transcription --

    def transcribe(self, language_code: str = "en") -> Optional[TranscriptData]:
        """
        If AssemblyAI is available and ASSEMBLYAI_API_KEY is set, transcribe the
        processed WAV and populate self.transcript with word-level timings.
        Returns TranscriptData or None if unavailable.
        """
        if aai is None:
            return None
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            return None

        aai.settings.api_key = api_key
        cfg = aai.TranscriptionConfig(language_code=language_code)
        transcriber = aai.Transcriber(config=cfg)
        tr = transcriber.transcribe(self.processed_wav)

        words: List[TranscriptWord] = []
        try:
            for w in tr.words or []:
                words.append(
                    TranscriptWord(
                        text=unidecode(w.text or ""),
                        start=(w.start or 0) / 1000.0,
                        end=(w.end or 0) / 1000.0,
                    )
                )
        except Exception:
            pass

        self.transcript = TranscriptData(words=words)
        return self.transcript
