import os

import speech_recognition as sr
from gtts import gTTS


# SPEECH TO TEXT
def load_mic(timeout=5):
    """
    Parameters
    timeout: int
        specifies the time that the audio will try to detect a phrase before it sends
        out an error message ("speech_recognition.WaitTimeoutError")
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something! Recording audio...")
        audio = r.listen(source, timeout=timeout)
        return r.recognize_google(audio)
        # ^ uses the Google Speech Recognition API to process and translate the audio
        # will provide more notes on this later


def load_audio(path: str, timeout=5):
    """
    Parameters
    path: str
        path for the audio file
    timeout: int
        specifies the time that the audio will try to detect a phrase before it sends
        out an error message ("speech_recognition.WaitTimeoutError")
    """
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        print("Processing audio...")
        audio = r.record(source)
        return r.recognize_google(audio)
        # ^ uses the Google Speech Recognition API to process and translate the audio
        # will provide more notes on this later


# ====================================================================================
# TEXT TO SPEECH
def text_to_speech(input_text, save_name: str):
    """Enter the text to convert to speech
    Enter the file name you want to save."""
    language = "en"
    myobject = gTTS(text=input_text, lang=language, slow=False)
    myobject.save(str(save_name) + str(".mp3"))
    os.system(str(save_name) + str(".mp3"))


# text_to_speech("Hi how are you doing today? I am a text to speech assistant.","helloworld")
