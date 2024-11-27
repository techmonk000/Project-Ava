import asyncio
import edge_tts
import pygame
import os
import time

VOICES = ['en-US-GuyNeural', 'en-US-JennyNeural']
VOICE = VOICES[0]
OUTPUT_FILE = "test.mp3"

async def amain(TEXT):
    communicate = edge_tts.Communicate(TEXT, VOICE,rate='-10%')
    await communicate.save(OUTPUT_FILE)

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.mixer.music.unload()

def speak(text):
    asyncio.run(amain(text))
    
    play_audio(OUTPUT_FILE)
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"{OUTPUT_FILE} has been deleted.")
    else:
        print(f"File {OUTPUT_FILE} not found.")

speak("welcome back from college boss, anything i can help you with?")