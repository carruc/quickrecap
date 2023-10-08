import glob
import os
import moviepy.editor
from pathlib import Path

# supported video formats
video_formats = ['wav', 'mp4', 'mkv', 'avi', 'mov']


# Get a list of all the files with .mkv extension in pathdir folder.
def _get_video_list(video_dir):
    video_list = []
    for video_format in video_formats:
        video_list.extend(glob.glob(os.path.join(video_dir, '*.' + video_format)))
    return video_list


def _create_wav(video_filename):
    # Replace video format with .wav
    for video_format in video_formats:
        audio_filename = video_filename.replace(video_format, '.wav')

    # Path variable to check if audio exists
    path = Path(audio_filename)

    # Skip conversion if file exists already
    if not (path.is_file()):
        video = moviepy.editor.VideoFileClip(video_filename)
        audio = video.audio

        # Some of my MKV files are without audio.
        if audio is not None:
            audio.write_audiofile(audio_filename)
        else:
            print('\t[' + video_filename + ']: Audio is empty.')
    else:
        print('\t[' + video_filename + ']: Audio already exists in directory.')
    return audio_filename


def video_converter(video_dir):
    audio_list = []

    # Converting all videos to .wav
    for video_filename in _get_video_list(video_dir):
        audio_list.append(_create_wav(video_filename))
        print('\t[' + video_filename + ']: Finished conversion to .wav.')
    return audio_list
