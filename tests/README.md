# Whisper API Tests

This project contains automated tests for the OpenAI Whisper API using Replicate.

## What these tests do

- Test audio transcription and translation for various languages
- Verify the accuracy of Whisper's language detection
- Compare Whisper's output against expected transcriptions and translations

## How to run the tests

1. Install the required Python package:
   ```
   pip install replicate
   ```

2. Set your Replicate API token as an environment variable:
   ```
   export REPLICATE_API_TOKEN=your_api_token_here
   ```

3. Run the tests:
   ```
   python -m unittest tests/test-openai-whisper.py
   ```

The tests will process several audio files and compare the results to expected outputs. Successful tests will show "completed successfully" messages. Any errors will be reported at the end of the test run.