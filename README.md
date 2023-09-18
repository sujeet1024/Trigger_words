# Trigger_words

1. This tool was built to detect multiple trigger words in a speech to follow corresponding command for respective trigger word

2. The dataset used for training the model can be downloaded from the tensorflow challenge through the link: 
[Tensorflow challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data)


3. For preprocessing the audio data pydub module has been used here, to use pydub you first need to set up ffmpeg package which aids in processing the audio data

    - For unix systems:
      ```
      sudo apt install ffmpeg
      ```

    - For windows systems:
      ```
      download the ffmpeg lib, extract, and add the ***\bin path to the environment path
      ```
