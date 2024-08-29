# Whisper API and WhisperX Performance Tests

This project includes automated tests for the OpenAI Whisper API using Replicate and performance tests for WhisperX.

## Whisper API Tests

### Purpose of These Tests

- Evaluate audio transcription and translation across various languages.
- Check the accuracy of Whisper's language detection.
- Compare Whisper's output with expected transcriptions and translations.

### Steps to Run the Whisper API Tests

1. Install the necessary Python package:
   ```
   pip install replicate
   ```

2. Set your Replicate API token as an environment variable:
   ```
   export REPLICATE_API_TOKEN=your_api_token_here
   ```

3. Execute the tests:
   ```
   python -m unittest tests/test-openai-whisper.py
   ```
This will evaluate the openai/whisper model before the introduction of the official Whisper model. The code in `tests/test-openai-whisper.py` uses the version `openai/whisper:be69de6b9dc57b3361dff4122ef4d6876ad4234bf5c879287b48d35c20ce3e83`, which is the slower, non-WhisperX version. We are conducting these tests to implement and enhance the official version of the model. The tests will process multiple audio files and compare the outcomes with expected results. Successful tests will display messages indicating completion, and any errors will be reported at the end of the test run.

## WhisperX Performance Testing

### How It Works

1. Start the test server:
   Execute this command in your terminal:
   ```
   sudo cog run -p 5000 python -m cog.server.http
   ```

2. Warm up the server:
   We use `whisperx_server.py` to prepare the server. This step helps CUDA optimize the computations for enhanced performance.

3. Conduct the tests:
   The main script `test-whisperx-performance.py` performs the following actions:
   - Loads test cases from a JSON file.
   - Executes each test case `NUM_RUNS` times (default is 20).
   - Measures the processing time for each run.
   - Gathers data such as detected language and transcription length.

4. Results:
   After the tests, the script will:
   - Create a directory named `whisperx-performance-results` in the `tests` folder.
   - Save a CSV file with all the test results.
   - Produce a box plot image showing performance across different test cases.

### Steps to Run the WhisperX Performance Tests

1. Ensure the server is operational (refer to step 1 above).
2. Execute the performance test script:
   ```
   python tests/test-whisperx-performance.py
   ```

3. Review the results in the `tests/whisperx-performance-results` folder.

That concludes the process! The WhisperX performance tests will provide insights into how WhisperX handles various audio samples and languages.