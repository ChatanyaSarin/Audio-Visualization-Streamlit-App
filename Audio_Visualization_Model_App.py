'''
Outline:
- Create animation: animate charts (potentially using streamlit)
'''
import librosa
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wave
import pickle

model_path = "model_simple.sav" #Defines the path to the model file

emotion_map = {
        'Disgust': 0,
        'Happiness': 1,
        'Saddness': 2,
        'Neutral': 3,
        'Fear': 4,
        'Anger': 5,
        'Surprise': 6
    } #Maps emotions to integers: taken from data preprocessing

reversed_emotion_map = {value:key for key, value in emotion_map.items()}
#Reverses emotion mapping such that integers can be mapped into emotions

#Uses librosa to load the inputted audio file as a list of frequency values
@st.cache_data
def process_audio(input_file):
    st.audio(input_file) #Creates an audio player within the streamlit app
    audio_signal, sample_rate = librosa.load(input_file)
    return audio_signal, sample_rate

#Creates a line chart displaying the audio frequency using librosa
@st.cache_data
def display_frequency(audio_signal, sample_rate):
    frequency_plot = librosa.display.waveshow(audio_signal, sr = sample_rate)
    st.pyplot(plt.gcf())

#Creates and displays a mel spectrogram using librosa
@st.cache_data
def display_mel_spectogram(audio_signal, sample_rate):
    mel_spectogram = librosa.feature.melspectrogram(y = audio_signal, sr = sample_rate)
    mel_plot = librosa.display.specshow(
        librosa.power_to_db(
            mel_spectogram, 
            ref = np.max
        ), 
        sr = sample_rate,
        hop_length = 512
    )
    plt.ylabel("Amplitude")
    plt.xlabel("Times")
    plt.colorbar(format = "%+2.0f dB")
    st.pyplot(plt.gcf())

#Creates the interface allowing users to select which plot they want displayed
def create_selections(audio_signal, sample_rate):
    chart_options = ["Frequency", "Mel-Spectogram"] #Graph titles go here
    functions = [display_frequency, display_mel_spectogram] #Graphing functions go here
    chart_selector = st.radio(
        label = "",
        options = chart_options,
        horizontal = True
    )
    selection_index = chart_options.index(chart_selector)
    functions[selection_index](audio_signal, sample_rate)

#Helper function to force the length of a given frequency array into a specific length
#Currently, this length is hard-coded at 66,150 though that may change in the future
@st.cache_data
def standardize_waveform_length(waveform):
    audio_length = 66150
    if len(waveform) > audio_length:
        waveform = waveform[:audio_length]
    else:
        waveform = np.pad(waveform, (0, max(0, audio_length - len(waveform))), "constant")
    return waveform

#Takes in a given audio signal and returns its mel-frequency cepstral coefficients
@st.cache_data
def preprocess_audio_for_prediction(audio_signal, sample_rate):
    waveform = standardize_waveform_length(waveform = audio_signal)
    mfcc = librosa.feature.mfcc(y = waveform, sr = sample_rate, n_mels = 128)
    mfcc = mfcc.reshape(-1)
    return mfcc

#Loads the model given in model_path and returns a Keras Sequential model
@st.cache_data
def load_model(model_path):
    model = pickle.load(open(model_path, "rb"))
    return model

#Uses the model to predict the speaker's emotion in the given audio clip
@st.cache_data
def get_emotion_prediction(mfcc):
    model = load_model(model_path)
    prediction = model.predict(mfcc[None])
    predicted_index = np.argmax(prediction)
    emotion = reversed_emotion_map[predicted_index]
    return emotion

#Combines all model functions and displays the model output as a subheader
@st.cache_data
def display_prediction(audio_signal, sample_rate):
    mfcc = preprocess_audio_for_prediction(audio_signal, sample_rate)
    prediction = get_emotion_prediction(mfcc)
    st.subheader("Predicted Emotion: " + prediction, divider = True)

#Defines the entire process of inputting audio, displaying the model's predictions, and displaying graphs
def run(input_file):
    audio_signal, sample_rate = process_audio(input_file)
    display_prediction(audio_signal, sample_rate)
    create_selections(audio_signal, sample_rate)

#Creates an input area to upload the file
def main():
    st.header("Upload your file here")
    file_uploader = st.file_uploader("", type = "wav")
    if file_uploader is not None:
        run(file_uploader)

if __name__ == "__main__":
    main()
