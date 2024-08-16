# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
from typing import Optional, Any
import os
import time
import numpy as np
import whisperx
from pydub import AudioSegment

from whisperx.transcribe import LANGUAGES, TO_LANGUAGE_CODE
from cog import BasePredictor, Input, Path, BaseModel

# Constants
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
MODEL_CACHE = "weights"
WHISPER_ARCH = f"./{MODEL_CACHE}/faster-whisper-large-v3"
BASE_URL = (
    f"https://weights.replicate.delivery/default/official-whisperx/{MODEL_CACHE}/"
)
DEFAULT_ASR_OPTIONS = {
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "suppress_numerals": False,
    "max_new_tokens": None,
    "clip_timestamps": None,
    "hallucination_silence_threshold": None,
}


class Output(BaseModel):
    detected_language: str
    transcription: str
    segments: Any
    translation: Optional[str]
    txt_file: Optional[Path]
    srt_file: Optional[Path]
    processing_time: float


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = [
            "faster-whisper-large-v3.tar",
            "vad.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        self.model = whisperx.load_model(
            WHISPER_ARCH, DEVICE, compute_type=COMPUTE_TYPE
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        model: str = Input(
            choices=[
                "large-v3",
            ],
            default="large-v3",
            description="Whisper model size (currently only large-v3 is supported).",
        ),
        transcription: str = Input(
            choices=["plain text", "srt", "vtt"],
            default="plain text",
            description="Choose the format for the transcription",
        ),
        translate: bool = Input(
            default=False,
            description="Translate the text to English when set to True",
        ),
        language: str = Input(
            choices=["auto"]
            + sorted(LANGUAGES.keys())
            + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
            default="auto",
            description="Language spoken in the audio, specify 'auto' for automatic language detection",
        ),
        temperature: float = Input(
            default=0,
            description="temperature to use for sampling",
        ),
        patience: float = Input(
            default=None,
            description="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
        ),
        suppress_tokens: str = Input(
            default="-1",
            description="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
        ),
        initial_prompt: str = Input(
            default=None,
            description="optional text to provide as a prompt for the first window.",
        ),
        condition_on_previous_text: bool = Input(
            default=True,
            description="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        ),
        temperature_increment_on_fallback: float = Input(
            default=0.2,
            description="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        ),
        compression_ratio_threshold: float = Input(
            default=2.4,
            description="if the gzip compression ratio is higher than this value, treat the decoding as failed",
        ),
        logprob_threshold: float = Input(
            default=-1.0,
            description="if the average log probability is lower than this value, treat the decoding as failed",
        ),
        no_speech_threshold: float = Input(
            default=0.6,
            description="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        ),
        batch_size: int = Input(
            default=32,
            description="Batch size for processing",
        ),
    ) -> Output:
        """Transcribe and optionally translate an audio file"""
        start_time = time.time()

        # Load audio
        audio = whisperx.load_audio(audio)

        # Handle temperature and temperature_increment_on_fallback
        if temperature_increment_on_fallback is not None:
            temperatures = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperatures = [temperature]

        # Update ASR options with user-provided values
        asr_options = DEFAULT_ASR_OPTIONS.copy()
        asr_options.update(
            {
                "temperatures": temperatures,
                "compression_ratio_threshold": compression_ratio_threshold,
                "log_prob_threshold": logprob_threshold,
                "no_speech_threshold": no_speech_threshold,
                "condition_on_previous_text": condition_on_previous_text,
                "initial_prompt": initial_prompt,
                "suppress_tokens": [int(t) for t in suppress_tokens.split(",") if t],
            }
        )

        # Add patience only if it's not None
        if patience is not None:
            asr_options["patience"] = patience

        # Update VAD options if needed
        vad_options = {
            "vad_onset": 0.5,
            "vad_offset": 0.363,
        }

        # Update language handling
        if language.lower() != "auto":
            language = self._normalize_language(language)

        # Reload the model with updated options
        self.model = whisperx.asr.load_model(
            WHISPER_ARCH,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            asr_options=asr_options,
            vad_options=vad_options,
            language=language if language != "auto" else None,
        )

        # Transcribe
        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=language if language != "auto" else None,
        )

        # Format transcription
        if transcription == "plain text":
            transcription_text = " ".join(
                [segment["text"] for segment in result["segments"]]
            )
        elif transcription == "srt":
            transcription_text = self.write_srt(result["segments"])
        else:  # vtt
            transcription_text = self.write_vtt(result["segments"])

        # Translate if requested
        translation = None
        if translate:
            translation_result = self.model.transcribe(
                audio,
                task="translate",
                batch_size=batch_size,
                language=language if language != "auto" else None,
            )
            translation = " ".join(
                [segment["text"] for segment in translation_result["segments"]]
            )

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Internal processing time: {processing_time:.2f} seconds")

        return Output(
            detected_language=result["language"],
            transcription=transcription_text,
            segments=result["segments"],
            translation=translation if translate else None,
            processing_time=processing_time,  # Add this line
        )

    def _normalize_language(self, language: str) -> str:
        """Normalize language input to ISO 639-1 code."""
        language = language.lower()
        if language in LANGUAGES:
            return language
        for code, name in LANGUAGES.items():
            if language == name.lower():
                return code
        for full_name, code in TO_LANGUAGE_CODE.items():
            if language == full_name.lower():
                return code
        raise ValueError(f"Unsupported language: {language}")

    @staticmethod
    def write_vtt(transcript):
        result = "WEBVTT\n\n"
        for segment in transcript:
            result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            result += f"{segment['text'].strip().replace('-->', '->')}\n\n"
        return result

    @staticmethod
    def write_srt(transcript):
        result = ""
        for i, segment in enumerate(transcript, start=1):
            result += f"{i}\n"
            result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            result += f"{segment['text'].strip().replace('-->', '->')}\n\n"
        return result


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def get_audio_duration(file_path):
    return len(AudioSegment.from_file(file_path)) / 1000.0
