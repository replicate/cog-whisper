import replicate
import unittest
import os
import json
from typing import Dict, List


class TestOpenAIWhisperAPI(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join("tests", "test-openai-whisper-outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.errors = []
        self.whisper_model = "openai/whisper:be69de6b9dc57b3361dff4122ef4d6876ad4234bf5c879287b48d35c20ce3e83"

    def test_audio_transcriptions(self):
        test_cases = self.get_test_cases()
        for i, case in enumerate(test_cases):
            if case["name"] == "4th_dimension_explained":
                continue
            with self.subTest(case=case["name"]):
                self.run_test_case(i, case)

    @staticmethod
    def get_test_cases() -> List[Dict]:
        with open(
            os.path.join("tests", "test-openai-whisper-testcases.json"), "r"
        ) as f:
            return json.load(f)

    def run_test_case(self, index: int, case: Dict):
        print(f"\nProcessing test case {index+1}: {case['name']}...")
        try:
            output = self.get_whisper_output(case)
            self.perform_assertions(case, output)
            print(f"Test case {index+1}: {case['name']} completed successfully.")
        except Exception as e:
            self.handle_error(index, case, str(e))

    def get_whisper_output(self, case: Dict) -> Dict:
        filename = f"{case['name']}.json"
        file_path = os.path.join(self.output_dir, filename)

        if os.path.exists(file_path):
            print(f"Reading cached output for {case['name']}...")
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            print(f"Calling Whisper API for {case['name']}...")
            input_data = {
                "audio": case["audio"],
                "model": "large-v3",
                "translate": case["translate"],
            }
            output = replicate.run(self.whisper_model, input=input_data)
            self.log_output(case["name"], output)
            return output

    def log_output(self, case_name: str, output: Dict):
        filename = f"{case_name}.json"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            json.dump(output, f, indent=2)

    def perform_assertions(self, case: Dict, output: Dict):
        errors = []

        if output["detected_language"] != case["expected_language"]:
            errors.append(
                f"Language mismatch: Expected '{case['expected_language']}', "
                f"but got '{output['detected_language']}'"
            )

        if case["translate"]:
            output_text = output.get("translation", "").strip()
            expected_text = case["expected_translation"].strip()
            text_type = "translation"
        else:
            output_text = output.get("transcription", "").strip()
            expected_text = case["expected_transcription"].strip()
            text_type = "transcription"

        if output_text.lower() != expected_text.lower():
            errors.append(
                f"{text_type.capitalize()} mismatch:\n"
                f"Expected: '{expected_text}'\n"
                f"Actual: '{output_text}'"
            )

        if errors:
            print(f"Detected language: {output['detected_language']}")
            print(f"Expected language: {case['expected_language']}")
            print(f"Comparing {text_type}s:")
            print(f"Output text: {output_text}")
            print(f"Expected text: {expected_text}")
            raise AssertionError("\n".join(errors))

    def handle_error(self, index: int, case: Dict, error_msg: str):
        full_error_msg = f"Error in test case {index+1}: {case['name']} - {error_msg}"
        print(full_error_msg)
        self.errors.append(full_error_msg)

    def tearDown(self):
        if self.errors:
            print("\nErrors occurred during testing:")
            for error in self.errors:
                print(error)
            self.fail(f"{len(self.errors)} test cases failed. See above for details.")


if __name__ == "__main__":
    unittest.main()
