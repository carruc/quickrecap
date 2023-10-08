import videoconverter
import audiotranscription
import recapper

audio_list = videoconverter.video_converter("/videos/")
transcription_list = audiotranscription.audio_transcriptions(audio_list)
recapper.recap(transcription_list, 'telecomunicazioni')