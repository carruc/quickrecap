import video_processing
import audio_processing
import recapper

audio_list = video_processing.video_converter(".\\videos")
transcription_list = audio_processing.audio_transcriptions(audio_list)
#recapper.recap(transcription_list, 'telecomunicazioni')