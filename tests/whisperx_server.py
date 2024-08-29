import json
import requests
from typing import List, Optional
from cog import BaseModel


class Output(BaseModel):
    detected_language: str
    transcription: str
    segments: List[dict]
    translation: Optional[str]
    txt_file: Optional[str]
    srt_file: Optional[str]
    processing_time: float


class WhisperServer:
    SERVER_URL = "http://localhost:5000/predictions"
    TEST_CASES_FILE = "tests/test-openai-whisper-testcases.json"
    WARM_UP_ITERATIONS = 3

    def make_prediction(self, audio_url, translate, language="auto", name="") -> Output:
        payload = {
            "input": {
                "audio": audio_url,
                "model": "large-v3",
                "translate": translate,
                "language": language,
            }
        }
        response = requests.post(self.SERVER_URL, json=payload)  # IO Blocking
        json_response = response.json()
        output_data = Output(**json_response["output"])

        print(f"\nWARMUP {name}")
        print(f"Detected Language: {output_data.detected_language}")
        print(f"Transcription: {output_data.transcription}")
        print(f"Translation: {output_data.translation}")
        print(f"Processing Time: {output_data.processing_time} seconds")

        return output_data

    def warm_up(self):
        print("Warming up the server...")
        with open(self.TEST_CASES_FILE, "r") as f:
            test_cases = json.load(f)
        for case in test_cases:
            for i in range(self.WARM_UP_ITERATIONS):
                result = self.make_prediction(
                    audio_url=case["audio"],
                    translate=case["translate"],
                    name=case["name"],
                )
            print(f"\nWarm-up iteration {i + 1} complete for test case.")
        print("\nWarm-up complete.")


def main():
    server = WhisperServer()
    try:
        server.warm_up()
        print("\nServer setup completed successfully.")
        print("The server is ready for predictions.")
    except requests.RequestException as e:
        print(f"\nError during warm-up: {e}")
        print("Please check if the server is running.")


if __name__ == "__main__":
    main()
