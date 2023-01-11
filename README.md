# Whisper Cog model

[![Replicate](https://replicate.com/openai/whisper/badge)](https://replicate.com/openai/whisper) 

This is an implementation of [Whisper](https://github.com/openai/whisper) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, run `get_weights.sh` from the project root to download pre-trained weights:

    ./scripts/get_weights.sh

You can then build a container and run predictions like so:

    cog predict -i audio="<path/to/your/audio/file>"
