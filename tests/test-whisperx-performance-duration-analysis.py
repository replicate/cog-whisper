import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment
import tempfile
import logging
from whisperx_server import WhisperServer
import argparse
import requests
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
TEST_CASES_FILE = "tests/test-openai-whisper-testcases.json"
OUTPUT_DIR = "tests/test-whisperx-performance-duration-analysis-results"
TEMP_AUDIO_DIR = "tests/temp_audio"
TEMP_DIR = tempfile.mkdtemp()
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")


def upload_file_to_replicate(file_path):
    with open(file_path, "rb") as file:
        files = {"file": file}
        headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
        response = requests.post(
            "https://api.replicate.com/v1/uploads", files=files, headers=headers
        )
        response.raise_for_status()
        return response.json()["serving_url"]


def download_audio(url: str, output_path: str):
    response = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(response.content)


def trim_audio(file_path: str, duration: int) -> str:
    audio = AudioSegment.from_file(file_path)
    trimmed_audio = audio[: duration * 1000]  # pydub works in milliseconds
    trimmed_path = os.path.join(TEMP_DIR, f"trimmed_{duration}s.wav")
    trimmed_audio.export(trimmed_path, format="wav")
    return trimmed_path


def process_test_case(server: WhisperServer, case: dict, num_segments: int):
    local_audio_path = os.path.join(TEMP_AUDIO_DIR, f"{case['name']}.mp3")

    # Download audio if it doesn't exist
    if not os.path.exists(local_audio_path):
        download_audio(case["audio"], local_audio_path)

    full_duration = int(AudioSegment.from_file(local_audio_path).duration_seconds)
    duration_steps = np.linspace(1, full_duration, num=num_segments, dtype=int)

    results = []
    for duration in duration_steps:
        trimmed_path = trim_audio(local_audio_path, duration)

        # Upload trimmed audio to Replicate
        trimmed_url = upload_file_to_replicate(trimmed_path)

        output = server.make_prediction(
            audio_url=trimmed_url, translate=case["translate"], name=case["name"]
        )
        results.append(
            {
                "test_case": case["name"],
                "duration": int(duration),
                "inference_time": output.processing_time,
            }
        )
        os.remove(trimmed_path)
        logging.info(
            f"Processed {duration}s clip: Inference time = {output.processing_time:.2f}s"
        )

    return results


def plot_results(results_df):
    plt.figure(figsize=(12, 8))

    for test_case in results_df["test_case"].unique():
        case_data = results_df[results_df["test_case"] == test_case]
        plt.plot(
            case_data["duration"],
            case_data["inference_time"],
            label=test_case,
            alpha=0.7,
        )

    # Calculate and plot average
    avg_data = results_df.groupby("duration")["inference_time"].mean().reset_index()
    plt.plot(
        avg_data["duration"],
        avg_data["inference_time"],
        label="Average",
        linewidth=3,
        color="black",
    )

    plt.title("Inference Time vs Audio Duration for All Test Cases")
    plt.xlabel("Audio Duration (seconds)")
    plt.ylabel("Inference Time (seconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "duration_analysis_performance.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")


def main():
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

    parser = argparse.ArgumentParser(
        description="WhisperX performance duration analysis"
    )
    parser.add_argument(
        "--num-segments", type=int, default=10, help="Number of segments to test."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

    with open(TEST_CASES_FILE, "r") as f:
        test_cases = json.load(f)

    server = WhisperServer()
    server.warm_up()

    all_results = []
    for case in test_cases:
        logging.info(f"Processing test case: {case['name']}")
        results = process_test_case(server, case, args.num_segments)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "duration_analysis_results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to CSV: {csv_path}")

    # Plot results
    plot_results(results_df)

    logging.info("WhisperX performance duration analysis completed.")


if __name__ == "__main__":
    main()
