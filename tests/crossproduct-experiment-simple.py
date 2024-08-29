import json
import random
import subprocess
import time
import os
import csv
import signal

# Constants
TEST_CASES_FILE = "tests/test-openai-whisper-testcases.json"
RESULTS_DIR = "tests/crossproduct-experiment-results"
JSON_FILE = os.path.join(RESULTS_DIR, "all_responses.jsonl")
CSV_FILE = os.path.join(RESULTS_DIR, "all_responses.csv")
SERVER_URL = "http://localhost:5000/predictions"

# Add this near the top of the file, after the other constants
AUDIO_FILES_TO_SKIP = [
    # "4th-dimension-explained-by-a-high-school-student.mp3"
]

param_ranges = {
    "language": [
        "auto",
        "sv",
        "zh",
        "fr",
        "no",
        "en",
        "ja",
        "ko",
        "de",
        "es",
        "it",
        "nl",
        "pl",
        "pt",
        "ru",
        "ar",
        "hi",
    ],
    "translate": [True, False],
    "transcription": [
        "plain text",
        "srt",
        "vtt",
    ],
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    "patience": [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "suppress_tokens": ["-1", "-1,0", "", "0,1,2,3,4,5,6,7,8,9", "264"],
    "initial_prompt": [
        None,
        "This is a test prompt",
        "Please transcribe the following audio accurately:",
        "Ignore any background noise and focus on the main speaker.",
    ],
    "condition_on_previous_text": [True, False],
    "temperature_increment_on_fallback": [0.1, 0.2, 0.3, 0.4, 0.5],
    "compression_ratio_threshold": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "logprob_threshold": [-2.0, -1.5, -1.0, -0.5, 0.0],
    "no_speech_threshold": [0.2, 0.4, 0.6, 0.8, 1.0],
    "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "model": ["large-v3"],
}


def load_test_cases():
    with open(TEST_CASES_FILE, "r") as f:
        return json.load(f)


def run_curl_command(payload, timeout=30):
    print("\nInput payload:")
    print(json.dumps(payload, indent=2))

    curl_command = [
        "curl",
        "-X",
        "POST",
        SERVER_URL,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
        "--max-time",
        str(timeout),
    ]
    try:
        result = subprocess.run(
            curl_command, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Error: Request timed out after {timeout} seconds")
        return None


def load_existing_responses():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            return [json.loads(line) for line in f]
    return []


def save_response(response):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(JSON_FILE, "a") as f:
        f.write(json.dumps(response) + "\n")


def signal_handler(sig, frame):
    print("\nTest run interrupted. Progress saved.")
    exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    test_cases = load_test_cases()
    existing_responses = load_existing_responses()
    test_counter = len(existing_responses)

    print(f"Continuing from test #{test_counter + 1}")

    max_retries = 3
    while True:
        test_case = random.choice(test_cases)
        params = {key: random.choice(values) for key, values in param_ranges.items()}
        payload = {"input": {"audio": test_case["audio"], **params}}

        # Check if the audio file should be skipped
        if AUDIO_FILES_TO_SKIP and any(
            file_to_skip in payload["input"]["audio"]
            for file_to_skip in AUDIO_FILES_TO_SKIP
        ):
            print(
                f"Skipping test with audio file in skip list: {payload['input']['audio']}"
            )
            continue

        print(f"\nRunning test #{test_counter + 1}")

        for attempt in range(max_retries):
            response = run_curl_command(payload)
            if response is None:
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
                continue

            try:
                response_json = json.loads(response)
                response_json["test_number"] = test_counter + 1
                save_response(response_json)
                test_counter += 1
                break
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON response for test #{test_counter + 1}")
                print("\nResponse:")
                print(response)
                if attempt == max_retries - 1:
                    print(
                        "\nStopping the test run due to repeated invalid JSON responses."
                    )
                    return
                print(f"Retrying... (Attempt {attempt + 2} of {max_retries})")
                time.sleep(2.0)

        time.sleep(random.uniform(4.0, 10.0))


if __name__ == "__main__":
    main()
