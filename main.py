import glob
import json
import os
import time
from pathlib import Path
import openai
import moviepy.editor
import whisper
import torch
from openai.error import RateLimitError

# In the future these variables will be passed as arguments
video_format = '.mkv'
video_dir = ".\\videos"

# Cuda allows for the GPU to be used which is more optimized than the cpu
if torch.cuda.is_available():
    print("Debug: CUDA is available.")
    torch.cuda.init()
    device = "cuda"
else:
    print("Debug: CUDA is not available: process will rely on CPU.")
    device = "cpu"

# Get a list of all the files with .mkv extension in pathdir folder.
mkv_filenames_list = glob.glob(os.path.join(video_dir, '*' + video_format))
wav_filenames_list = []
text_filenames_list = []

# Converting all videos to .wav
for filename in mkv_filenames_list:
    print('\t[' + filename + ']: Starting conversion to .wav.')

    # Replace .mkv with .wav
    wav_file_name = filename.replace(video_format, '.wav')
    path = Path(wav_file_name)

    if not (path.is_file()):
        video = moviepy.editor.VideoFileClip(filename)
        audio = video.audio

        # Some of my MKV files are without audio.
        if audio is not None:
            audio.write_audiofile(wav_file_name)
            wav_filenames_list.append(wav_file_name)
        else:
            print('\t[' + filename + ']: Audio is empty.')
    else:
        print('\t[' + filename + ']: Audio already exists in directory.')
        wav_filenames_list.append(wav_file_name)
else:
    print("Debug: Finished .mkv to .wav conversion")

# Load whisper model
model_size = "small"
print("loading model :", model_size)
model = whisper.load_model(model_size).to(device)
print("Debug: ", model_size, "model loaded")

# Chat-GPT api key request using key stored in JSON
with open("secrets.json") as f:
    secrets = json.load(f)
    api_key = secrets["api_key"]
    openai.api_key = api_key
url = 'https://api.openai.com/v1/engines/davinci-codex/completions'

# Transcribing audios
for filename in wav_filenames_list:
    print('\t[' + filename + ']: Starting transcription.')

    # Replace .wav with .txt
    txt_file_name = filename.replace('.wav', '.txt')
    path = Path(txt_file_name)

    if not (path.is_file()):
        result = model.transcribe(filename)

        # Some of my MKV files are without audio.
        if result is not None:
            with open(txt_file_name, "w", encoding="utf-8") as text_file:
                text_file.write(result["text"])
                text_file.close()
        else:
            print('\t[' + filename + ']: Transcription is empty.')
    else:
        with open(txt_file_name, "r") as text_file:
            result = dict([("text", text_file.read())])
            text_file.close()
        print('\t[' + filename + ']: Transcription already exists in directory.')

    # Requesting recap from ChatGPT
    if result["text"] is not None:
        print('\t[' + filename + ']: Starting recap.')

        # Define prompt
        fullprompt = "Potresti riassumere le seguenti lezioni universitarie trascritte? Testo: " + result["text"]

        # Split in parts of 8k characters to avoid RateLimitErrors
        n = 8000
        prompts = [(fullprompt[i:i + n]) for i in range(0, len(fullprompt), n)]
        print('Debug: Number of prompts:', len(prompts), '.')

        # Debug print
        print('Debug: Prompt = ', fullprompt[0:50], '...')

        messages = [
            # system message first, it helps set the behavior of the assistant
            {"role": "system",
             "content": "Sei un assistente di grande aiuto, mi dovrai aiutare a riassumere lezioni universitarie."}
        ]
        responses = []

        # First request prompt iteration (gpt-3.5 max 4096, gpt-4 max 8192 e gpt4-32k 32k)
        for prompt in prompts:
            try:
                messages = [{"role": "system",
                             "content": "Sei un assistente di grande aiuto, mi dovrai aiutare a riassumere lezioni universitarie."},
                            {"role": "user",
                             "content": prompt}]
                chat_completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=1.0
                )
                if chat_completion.choices[0].message is not None:
                    messages.append(chat_completion.choices[0].message)
                    responses.append(chat_completion.choices[0].message.content)

                print('Debug: one prompt finished.')
                # Add request time limit (10k tokens/min)
                time.sleep(60)
            except RateLimitError:
                print('Debug: RateLimitError.')

        # If the number of responses is not zero...
        if len(responses) > 0:
            print('\t[' + filename + ']: Writing long-form recap.')

            # Writing long-form recap on file
            text_file_name = filename.replace('.wav', '_recap.txt')
            with open(text_file_name, 'w') as f:
                for response in responses:
                    f.write(response + '\n')
                f.close()

            # Summed up abstract request
            print('\t[' + filename + ']: Abstract request.')

            prompt = "Ti passo varie trascrizioni di una lezione universitaria, potresti unirle in modo coeso e preciso? Testo: " + '\n'.join(
                responses)
            try:
                messages = [{"role": "system",
                             "content": "Sei un assistente di grande aiuto, mi dovrai aiutare a riassumere lezioni universitarie."},
                            {"role": "user",
                             "content": prompt}]
                chat_completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=1.0
                )

                text_file_name = filename.replace('.wav', '_abstract.txt')
                with open(text_file_name, 'w') as f:
                    f.write(chat_completion.choices[0].message.content)
                    f.close()
            except RateLimitError:
                print('Debug: RateLimitError.')
        else:
            print('\t[' + filename + ']: Recap is empty.')
    else:
        print('\t[' + filename + ']: Transcription is empty.')
else:
    print('\t[' + filename + ']: Recap finished.')
