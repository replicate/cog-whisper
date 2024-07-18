import unittest
import os
import json
import subprocess
from typing import Dict, List

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]


class TestCogPredict(unittest.TestCase):
    def setUp(self):
        print("Setting up TestCogPredict...")
        self.output_dir = os.path.join("tests", "test-cog-predict-outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.errors = []
        print(f"Output directory created: {self.output_dir}")

    def test_audio_transcriptions(self):
        print("Starting audio transcription tests...")
        test_cases = self.get_test_cases()
        print(f"Found {len(test_cases)} test cases")
        for i, case in enumerate(test_cases):
            print(f"\nStarting test case {i+1}: {case['name']}")
            with self.subTest(case=case["name"]):
                self.run_test_case(i, case)

    @staticmethod
    def get_test_cases() -> List[Dict]:
        print("Loading test cases from JSON file...")
        with open(
            os.path.join("tests", "test-openai-whisper-testcases.json"), "r"
        ) as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases")
        return test_cases

    def run_test_case(self, index: int, case: Dict):
        print(f"\nProcessing test case {index+1}: {case['name']}...")
        try:
            output = self.get_cog_predict_output(case)
            print("Performing assertions...")
            self.perform_assertions(case, output)
            print(f"Test case {index+1}: {case['name']} completed successfully.")
        except Exception as e:
            print(f"Error occurred in test case {index+1}: {case['name']}")
            self.handle_error(index, case, str(e))

    def get_cog_predict_output(self, case: Dict) -> Dict:
        filename = f"{case['name']}_cog.json"
        file_path = os.path.join(self.output_dir, filename)

        if os.path.exists(file_path):
            print(f"Reading cached output for {case['name']}...")
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            print(f"Running cog predict for {case['name']}...")
            cmd = [
                "sudo",
                "cog",
                "predict",
                "-i",
                f"audio={case['audio']}",
                "-i",
                "model=large-v3",
                "-i",
                f"translate={str(case['translate']).lower()}",
                "-i",
                "language=auto",
                "--use-cog-base-image",
            ]
            print(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print("Parsing cog output...")
            output = self.parse_cog_output(result.stdout)
            print("Logging output...")
            self.log_output(case["name"], output)
            return output

    def parse_cog_output(self, output: str) -> Dict:
        print("Parsing cog output...")
        # Find the JSON part of the output
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON found in the output")

        json_str = output[json_start:json_end]
        result = json.loads(json_str)
        print(f"Parsed JSON output with {len(result)} keys")
        return result

    def log_output(self, case_name: str, output: Dict):
        filename = f"{case_name}_cog.json"
        file_path = os.path.join(self.output_dir, filename)
        print(f"Logging output to {file_path}")
        with open(file_path, "w") as f:
            json.dump(output, f, indent=2)
        print("Output logged successfully")

    def get_language_code(self, language_name: str) -> str:
        return TO_LANGUAGE_CODE.get(language_name.lower(), language_name)

    def perform_assertions(self, case: Dict, output: Dict):
        print("Performing assertions...")
        errors = []

        if "detected_language" in output:
            print(f"Checking detected language: {output['detected_language']}")
            expected_code = self.get_language_code(case["expected_language"])
            if output["detected_language"] != expected_code:
                errors.append(
                    f"Language mismatch: Expected '{expected_code}', "
                    f"but got '{output['detected_language']}'"
                )

        if case["translate"]:
            print("Checking translation...")
            output_text = output.get("translation", "").strip().lower()
            expected_text = case["expected_translation"].strip().lower()
            text_type = "translation"
        else:
            print("Checking transcription...")
            output_text = output.get("transcription", "").strip().lower()
            expected_text = case["expected_transcription"].strip().lower()
            text_type = "transcription"

        # Check for partial match (first 100 characters)
        match_length = min(100, len(output_text), len(expected_text))
        if output_text[:match_length] == expected_text[:match_length]:
            print(f"First {match_length} characters match (case-insensitive).")
            if output_text == expected_text:
                print("Full text matches exactly (case-insensitive).")
            else:
                print("Full text differs, but partial match is acceptable.")
        else:
            errors.append(
                f"{text_type.capitalize()} mismatch:\n"
                f"Expected (first {match_length} chars): '{expected_text[:match_length]}'\n"
                f"Actual (first {match_length} chars): '{output_text[:match_length]}'"
            )

        if errors:
            print("Errors detected during assertions:")
            print(f"Detected language: {output.get('detected_language', 'N/A')}")
            print(
                f"Expected language code: {self.get_language_code(case['expected_language'])}"
            )
            print(f"Comparing {text_type}s:")
            print(f"Output text: {output_text}")
            print(f"Expected text: {expected_text}")
            raise AssertionError("\n".join(errors))
        else:
            print("All assertions passed successfully")

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
        else:
            print("All tests completed successfully")


if __name__ == "__main__":
    print("Starting TestCogPredict...")
    unittest.main()
