from pathlib import Path
import whisper
import torch

def _torch_init():
    # Cuda allows for the GPU to be used which is more optimized than the cpu
    if torch.cuda.is_available():
        print("Debug: CUDA is available.")
        torch.cuda.init()
        device = "cuda"
    else:
        print("Debug: CUDA is not available: process will rely on CPU.")
        device = "cpu"
    return device


def _whisper_init(device):
    # Load whisper model
    model_size = "small"
    print("Debug: Loading [" + model_size + "] model.")
    model = whisper.load_model(model_size).to(device)
    print("Debug: [" + model_size + "]model loaded")
    return model


def _create_transcription(audio, model):
    print('\t[' + audio + ']: Starting transcription.')

    # Replace .wav with .txt
    transcription = audio.replace('.wav', '_transcription.txt')
    path = Path(transcription)

    # Check if transcription already exists
    if not (path.is_file()):
        result = model.transcribe(audio)

        # Some of my MKV files are without audio.
        if result is not None:
            with open(transcription, "w", encoding="utf-8") as text_file:
                text_file.write(result["text"])
                text_file.close()
        else:
            print('\t[' + audio + ']: Transcription is empty.')
            return None
    else:
        print('\t[' + audio + ']: Transcription already exists in directory.')
    return transcription


def audio_transcriptions(audio_list):
    transcription_list = []
    whisper_model = _whisper_init(_torch_init())

    for audio in audio_list:
        transcription_list.append(_create_transcription(audio, whisper_model))
    return transcription_list
