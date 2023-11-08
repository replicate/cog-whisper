from typing import Optional, Any
import torch
import numpy as np

from cog import BasePredictor, Input, Path, BaseModel
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import format_timestamp


class ModelOutput(BaseModel):
    detected_language: str
    transcription: str
    segments: Any
    translation: Optional[str]
    txt_file: Optional[Path]
    srt_file: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        with open(f"./weights/large-v3.pt", "rb") as fp:
            checkpoint = torch.load(fp, map_location="cpu")
            dims = ModelDimensions(**checkpoint["dims"])
            self.model = Whisper(dims)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to("cuda")

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        model: str = Input(
            choices=["large-v3"],
            default="large-v3",
            description="This version only supports Whisper-large-v3.",
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
            choices=sorted(LANGUAGES.keys())
            + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
            default=None,
            description="language spoken in the audio, specify None to perform language detection",
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
    ) -> ModelOutput:
        """Transcribes and optionally translates a single audio file"""
        print(f"Transcribe with {model} model.")
        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        args = {
            "language": language,
            "patience": patience,
            "suppress_tokens": suppress_tokens,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "fp16": True,
            "verbose": False,
        }
        with torch.inference_mode():
            result = self.model.transcribe(str(audio), temperature=temperature, **args)

        if transcription == "plain text":
            transcription = result["text"]
        elif transcription == "srt":
            transcription = write_srt(result["segments"])
        else:
            transcription = write_vtt(result["segments"])

        if translate:
            translation = self.model.transcribe(
                str(audio), task="translate", temperature=temperature, **args
            )

        return ModelOutput(
            segments=result["segments"],
            detected_language=LANGUAGES[result["language"]],
            transcription=transcription,
            translation=translation["text"] if translate else None,
        )


def write_vtt(transcript):
    result = ""
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result


def write_srt(transcript):
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
        result += "\n"
    return result
