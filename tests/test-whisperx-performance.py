import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from typing import Dict, List
from datetime import datetime
from whisperx_server import WhisperServer
from pydub import AudioSegment

# Constants
NUM_RUNS = 3
OUTPUT_DIR = "tests/test-whisperx-performance-results"
TEST_CASES_FILE = "tests/test-openai-whisper-testcases.json"
TEMP_AUDIO_DIR = "tests/temp_audio"

def download_audio(url: str, output_path: str):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def get_audio_length(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds

def run_performance_test(
    server: WhisperServer, test_cases: List[Dict], num_runs: int = NUM_RUNS
) -> pd.DataFrame:
    results = []
    for case in test_cases:
        print(f"Processing test case: {case['name']}")
        local_audio_path = os.path.join(TEMP_AUDIO_DIR, f"{case['name']}.mp3")
        
        # Download audio if it doesn't exist
        if not os.path.exists(local_audio_path):
            download_audio(case['audio'], local_audio_path)
        
        audio_length = get_audio_length(local_audio_path)
        
        for run in range(num_runs):
            start_time = time.time()
            output = server.make_prediction(
                audio_url=case["audio"],
                translate=case["translate"],
                name=case["name"],
            )
            end_time = time.time()
            processing_time = end_time - start_time

            results.append(
                {
                    "test_case": case["name"],
                    "run": run + 1,
                    "processing_time": processing_time,
                    "detected_language": output.detected_language,
                    "transcription_length": len(output.transcription),
                    "audio_length": audio_length,
                }
            )

            print(
                f"Run {run + 1}/{num_runs}: Processing time: {processing_time:.2f} seconds"
            )

    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame):
    # Original box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="test_case", y="processing_time", data=df)
    plt.title("WhisperX Performance: Processing Time by Test Case")
    plt.xlabel("Test Case")
    plt.ylabel("Processing Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "whisperx-performance-boxplot.png"))
    plt.close()

    # New scatter plot: Audio Length vs Processing Time
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="audio_length", y="processing_time", hue="test_case", data=df)
    plt.title("WhisperX Performance: Audio Length vs Processing Time")
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Processing Time (seconds)")
    plt.legend(title="Test Case", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "whisperx-performance-scatter.png"))
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(TEMP_AUDIO_DIR):
        os.makedirs(TEMP_AUDIO_DIR)

    print("Setting up WhisperX server...")
    server = WhisperServer()
    server.warm_up()

    print("Loading test cases...")
    with open(TEST_CASES_FILE, "r") as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    print("Running performance tests...")
    results_df = run_performance_test(server, test_cases, NUM_RUNS)

    print("Generating visualizations...")
    plot_results(results_df)

    print("Saving results...")
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "whisperx-performance-results.csv"), index=False
    )

    print("Performance test completed successfully!")

if __name__ == "__main__":
    main()