from faster_whisper import WhisperModel
import speech_recognition as sr
import time
import os

num_cores = os.cpu_count()

whisper_size = 'small'
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores, num_workers=num_cores)
r = sr.Recognizer()
source = sr.Microphone()

def callback(recognizer, audio):
    command_audio_path = 'command.wav'
    with open(command_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())
    text = wav_to_text(command_audio_path)
    print(f"Recognized text: {text}")
    os.remove(command_audio_path)  

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('Listening...')
    
    stop_listening = r.listen_in_background(source, callback)
    return stop_listening

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

stop_listening = start_listening()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_listening(wait_for_stop=False)
    print("Stopped listening.")
