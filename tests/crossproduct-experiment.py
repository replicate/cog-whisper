import shutil
import sys
import json
import csv
import time
import os
import argparse
import random
import requests
import itertools
import pandas as pd
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from whisperx_server import WhisperServer
import logging
from pydub import AudioSegment
import traceback
from dataclasses import dataclass

# Constants
TEST_CASES_FILE = "tests/test-openai-whisper-testcases.json"
TEMP_AUDIO_DIR = "tests/temp_audio"
RESULTS_DIR = "tests/crossproduct-experiment-results"
RESULTS_CSV = f"{RESULTS_DIR}/whisperx_performance_test_results.csv"

# Define the columns for the CSV
PREDICT_METHOD_INPUTS = [
    "audio",
    "model",
    "transcription",
    "translate",
    "language",
    "temperature",
    "patience",
    "suppress_tokens",
    "initial_prompt",
    "condition_on_previous_text",
    "temperature_increment_on_fallback",
    "compression_ratio_threshold",
    "logprob_threshold",
    "no_speech_threshold",
    "batch_size",
]

CSV_COLUMNS = [
    "file_name",
    "length_of_input_audio",
    "url",
    *PREDICT_METHOD_INPUTS,
    "processing_time",
    "detected_language",
    "status",
    "expected_language",
    "expected_transcription",
    "expected_translation",
    "succeeded",
    "logs",
    "output_transcription",
    "output_translation",
    "output_segments",
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
    "transcription": ["plain text", "srt", "vtt"],
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


def safely_delete_results_folder():
    """Safely delete the results folder after user confirmation."""
    if os.path.exists(RESULTS_DIR):
        print(
            f"WARNING: You are about to delete the entire results folder: {RESULTS_DIR}"
        )
        print("This action cannot be undone.")
        confirmation = input("Are you sure you want to proceed? (yes/no): ").lower()
        if confirmation in ["yes", "y"]:
            try:
                shutil.rmtree(RESULTS_DIR)
                print(f"Results folder {RESULTS_DIR} has been deleted.")
            except Exception as e:
                print(
                    f"An error occurred while trying to delete the results folder: {e}"
                )
                sys.exit(1)
        else:
            print("Deletion cancelled. Exiting the program.")
            sys.exit(0)
    else:
        print(f"Results folder {RESULTS_DIR} does not exist. No action taken.")


@dataclass
class ServerInput:
    audio: str
    model: str
    translate: bool
    language: str
    transcription: str = "plain text"
    temperature: float = 0.0
    patience: Optional[float] = None
    suppress_tokens: str = "-1"
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    temperature_increment_on_fallback: float = 0.2
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    batch_size: int = 32


class ExtendedWhisperServer(WhisperServer):
    def make_prediction(self, audio_url, **kwargs) -> Dict[str, Any]:
        payload = {
            "input": {
                "audio": audio_url,
                "model": "large-v3",
                "translate": kwargs.get("translate", False),
                "language": kwargs.get("language", "auto"),
            }
        }
        response = requests.post(self.SERVER_URL, json=payload)
        json_response = response.json()

        # Print the raw response for debugging
        print("Raw server response:")
        print(json.dumps(json_response, indent=2))

        # Return the raw response
        return json_response


def load_test_cases() -> List[Dict[str, Any]]:
    """Load all test cases from the JSON file."""
    print("Loading test cases from JSON file...")
    with open(TEST_CASES_FILE, "r") as f:
        all_cases = json.load(f)
    print(f"Loaded {len(all_cases)} test cases.")
    return all_cases


def download_audio_files(test_cases: List[Dict[str, Any]]) -> None:
    """Download audio files for all test cases."""
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    for test_case in test_cases:
        audio_url = test_case["audio"]
        file_name = os.path.basename(audio_url)
        file_path = os.path.join(TEMP_AUDIO_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"Downloading {file_name} from {audio_url}...")
            response = requests.get(audio_url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded and saved to {file_path}")
            else:
                print(
                    f"Failed to download {file_name}. Status code: {response.status_code}"
                )
        else:
            print(f"File {file_name} already exists. Skipping download.")


def get_audio_length(file_path: str) -> float:
    """Get the length of an audio file in seconds."""
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0


def run_test_case(
    server: ExtendedWhisperServer,
    test_case: Dict[str, Any],
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    print(f"\nRunning test case: {test_case['name']}")
    logging.info(f"\nRunning test case: {test_case['name']}")
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"  audio: {test_case['audio']}")

    try:
        file_name = os.path.basename(test_case["audio"])
        file_path = os.path.join(TEMP_AUDIO_DIR, file_name)

        prediction_params = {
            "audio_url": file_path,
            "translate": params.get("translate", False),
            "language": params.get("language", "auto"),
            "name": test_case["name"],
        }

        print("Making prediction...")
        response = server.make_prediction(**prediction_params)
        print("Prediction made successfully")

        result_dict = {
            "file_name": file_name,
            "length_of_input_audio": get_audio_length(file_path),
            "url": test_case["audio"],
            "audio": test_case["audio"],
            "model": params.get("model", "large-v3"),
            "transcription": params.get("transcription", "plain text"),
            "translate": params.get("translate", False),
            "language": params.get("language", "auto"),
            "temperature": params.get("temperature", 0.0),
            "patience": params.get("patience", None),
            "suppress_tokens": params.get("suppress_tokens", "-1"),
            "initial_prompt": params.get("initial_prompt", None),
            "condition_on_previous_text": params.get(
                "condition_on_previous_text", True
            ),
            "temperature_increment_on_fallback": params.get(
                "temperature_increment_on_fallback", 0.2
            ),
            "compression_ratio_threshold": params.get(
                "compression_ratio_threshold", 2.4
            ),
            "logprob_threshold": params.get("logprob_threshold", -1.0),
            "no_speech_threshold": params.get("no_speech_threshold", 0.6),
            "batch_size": params.get("batch_size", 32),
            "processing_time": response.get("processing_time", "N/A"),
            "detected_language": response.get("detected_language", "N/A"),
            "status": response.get("status", "N/A"),
            "expected_language": test_case.get("expected_language"),
            "expected_transcription": test_case.get("expected_transcription"),
            "expected_translation": test_case.get("expected_translation"),
            "succeeded": response.get("status") == "succeeded",
            "logs": json.dumps(response.get("logs", {})),
            "output_transcription": response.get("transcription", "N/A"),
            "output_translation": response.get("translation", "N/A"),
            "output_segments": json.dumps(response.get("segments", [])),
        }

        print(f"Test case {test_case['name']} completed successfully")
        logging.info(f"Test case {test_case['name']} completed successfully")
        return result_dict

    except requests.RequestException as e:
        error_message = f"Network error in test case {test_case['name']}: {str(e)}"
        logging.error(error_message)
        return create_error_result(test_case, error_message)

    except Exception as e:
        error_message = f"Unexpected error in test case {test_case['name']}: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        return create_error_result(test_case, error_message)


def create_error_result(
    test_case: Dict[str, Any], error_message: str
) -> Dict[str, Any]:
    file_name = os.path.basename(test_case["audio"])
    file_path = os.path.join(TEMP_AUDIO_DIR, file_name)
    return {
        "file_name": file_name,
        "length_of_input_audio": get_audio_length(file_path),
        "url": test_case["audio"],
        "audio": test_case["audio"],
        "model": "large-v3",  # Default model
        "transcription": None,
        "translate": None,
        "language": None,
        "temperature": None,
        "patience": None,
        "suppress_tokens": None,
        "initial_prompt": None,
        "condition_on_previous_text": None,
        "temperature_increment_on_fallback": None,
        "compression_ratio_threshold": None,
        "logprob_threshold": None,
        "no_speech_threshold": None,
        "batch_size": None,
        "processing_time": None,
        "detected_language": None,
        "status": "failed",
        "expected_language": test_case.get("expected_language"),
        "expected_transcription": test_case.get("expected_transcription"),
        "expected_translation": test_case.get("expected_translation"),
        "succeeded": False,
        "logs": json.dumps({"error": error_message}),
        "output_transcription": None,
        "output_translation": None,
        "output_segments": None,
    }


def write_result_to_csv(result: Dict[str, Any]):
    """Append a single result to the CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"Result written to {RESULTS_CSV}")


def run_random_samples(
    server: ExtendedWhisperServer,
    test_cases: List[Dict[str, Any]],
    num_samples: int = 5,
) -> pd.DataFrame:
    """Run random samples and return results as a DataFrame."""
    results = []

    for _ in range(num_samples):
        test_case = random.choice(test_cases)
        params = {k: random.choice(v) for k, v in param_ranges.items()}

        result = run_test_case(server, test_case, params)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


def load_completed_tests() -> Set[Tuple]:
    """Load the parameter combinations of completed tests from the CSV file."""
    completed_tests = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                test_params = tuple(row[param] for param in param_ranges.keys())
                completed_tests.add(test_params)
    return completed_tests


def main(continue_from_last: bool, warm_up: bool = True, num_tests: int = 100) -> None:
    try:
        print("Starting main function")
        logging.info("Setting up WhisperX server...")
        server = ExtendedWhisperServer()
        print("WhisperX server set up")
        logging.info("WhisperX server set up successfully.")

        if warm_up:
            logging.info("Starting warm-up...")
            server.warm_up()
            logging.info("Warm-up completed.")
        else:
            print("Skipping warm-up")

        print("Loading test cases...")
        test_cases = load_test_cases()
        print(f"Loaded {len(test_cases)} test cases")

        print("Downloading audio files...")
        download_audio_files(test_cases)

        completed_tests = set()
        if continue_from_last:
            print("Loading completed tests...")
            completed_tests = load_completed_tests()
            print(f"Loaded {len(completed_tests)} completed tests")
            logging.info(f"Loaded {len(completed_tests)} completed tests.")

        results = []
        tests_run = 0
        print(f"Starting test loop, will run up to {num_tests} tests")
        while tests_run < num_tests:
            test_case = random.choice(test_cases)
            params = {k: random.choice(v) for k, v in param_ranges.items()}

            if continue_from_last and tuple(params.values()) in completed_tests:
                continue

            result = run_test_case(server, test_case, params)
            if result is not None:
                results.append(result)
                tests_run += 1
                print(f"Completed test {tests_run}/{num_tests}")

                if continue_from_last:
                    write_result_to_csv(result)

        df = pd.DataFrame(results)
        print("Random samples completed. Results:")
        print(df)

        if not continue_from_last:
            print(f"Saving results to {RESULTS_CSV}")
            df.to_csv(
                RESULTS_CSV,
                index=False,
                mode="a",
                header=not os.path.exists(RESULTS_CSV),
            )
            print("Results saved successfully")

        print(f"Test loop completed. Total tests run: {len(df)}")
        logging.info(f"Completed {len(df)} tests successfully.")
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        logging.exception(f"An error occurred in main: {str(e)}")
        raise

    print("Main function completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WhisperX performance tests")
    parser.add_argument(
        "--continue",
        dest="continue_from_last",
        action="store_true",
        help="Continue from last completed test",
    )
    parser.add_argument(
        "--no-warm-up   ",
        dest="warm_up",
        action="store_false",
        help="Skip warming up the server",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of tests to run before stopping",
    )
    parser.add_argument(
        "--delete-results",
        action="store_true",
        help="Delete the results folder and start fresh (use with caution)",
    )
    args = parser.parse_args()

    if args.delete_results:
        safely_delete_results_folder()

    try:
        main(args.continue_from_last, args.warm_up, args.num_tests)
    except KeyboardInterrupt:
        logging.info("\nTest execution stopped by user.")
    except Exception as e:
        logging.exception(f"\nAn unexpected error occurred: {e}")
    finally:
        logging.info("Exiting the program.")
