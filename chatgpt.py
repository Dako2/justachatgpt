# -*- coding: utf-8 -*-
#import subprocess
#subprocess.call('export LANG=en_US.UTF-8', shell=True)

import os
import sys
import openai 
import pyaudio
import wave
import numpy as np
import torch

from gtts import gTTS
import langdetect
from playsound import playsound

API_KEY = "sk-.." 

openai.api_key = API_KEY

class SileroVAD:
    def __init__(self, vad_theshold = 0.6):
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad"
        )
        self.vad_theshold = vad_theshold
        self.no_voice_counter = 0

    def detect(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = self.int2float(audio_int16)
        new_confidence = self.vad_model(torch.from_numpy(audio_float32), 16000).item()

        print(self.no_voice_counter, flush = True)
        if new_confidence < self.vad_theshold:
            self.no_voice_counter += 1
            #print(self.no_voice_counter, flush = True)
        else:
            self.no_voice_counter = 0
        return new_confidence

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()
        return sound

class ChatGpt:
  def __init__(self, Enable = True):
    self.enable_chatgpt = Enable

  def whisper(self,):
    if not self.enable_chatgpt:
      return ".."
    try:
      file = open("./openai.mp3", "rb")
      transcription = openai.Audio.transcribe("whisper-1", file)
      return transcription['text']
    except openai.error.RateLimitError:
        self.enable_chatgpt = False
        return "OpenAI: API limit exceeded. Please wait and try again later."
    
  def gpt(self,transcription):
    if not self.enable_chatgpt:
      return ".."
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=[{"role": "user", "content": transcription}]
    )
    return completion.choices[0].message.content

def tts(text):
  # Detect the language of the text
  try:
    lang = langdetect.detect(text)
    if lang:
        # Generate the audio file
        if lang =='zh-cn':
          lang = 'zh-tw'
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
    else:
        print(f"Language not supported: {lang}")
    # Replace "audio_file.mp3" with the path to your audio file
    playsound("output.mp3")
  except:
    pass

vad = SileroVAD(vad_theshold = 0.6)
chatgpt = ChatGpt(True)

while True:
  # set up audio recording
  audio = pyaudio.PyAudio()
  stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

  # record audio for 10 seconds
  print("recording...")
  frames = []
  for i in range(0, int(16000 / 1024 * 10)):
      data = stream.read(1024)
      if vad.detect(data) > vad.vad_theshold:
        frames.append(data)
      else:
        if vad.no_voice_counter > 30:
          vad.no_voice_counter = 0
          break

  #print("record finished")
  # stop audio recording
  stream.stop_stream()
  stream.close()
  audio.terminate()

  if not len(frames):
    continue
  # save audio to file

  try:
    waveFile = wave.open("openai.mp3", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(16000)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    input_text = chatgpt.whisper()
    output_text = chatgpt.gpt(input_text)
    print('Me:     ',  input_text)
    #print("", output_text, '\n')
    print("\033[91m" + "ChatGpt: " + output_text + "\033[0m \n\n")
    tts(output_text)

  except:
    pass







