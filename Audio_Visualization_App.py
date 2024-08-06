import streamlit as st
import numpy as np
import pandas as pd
import wave
import matplotlib.pyplot as plt
import plotly.express as px
import librosa

#HELPER FUNCTIONS
#Displays a particular emotion
def displayEmotion (emotion):
    emotions = {
        "angry": ":angry:",
        "joy": ":smile:",
        "sadness": ":cry:"
    }
    st.header("Emotion Detected:" + emotions[emotion]) #displays specific emotion emoji

#Reads the audio file - returns audio time and array of signals (frequency)
@st.cache_data
def readAudio (inputFile):
    file = wave.open(inputFile, 'rb') #opens the file using wave
    frequency = file.getframerate()
    nsamples = file.getnframes()
    audioTime = nsamples/frequency
    signal = file.readframes(nsamples)
    signalArray = np.frombuffer(signal, dtype = np.int16)
    signalArray = signalArray.reshape(-1)
    times = np.linspace(0, audioTime, num = nsamples)

    return times, signalArray

#Plots the frequency using Plotly
@st.cache_data
def plotFrequency (times, signal):
    frequencyDf = pd.DataFrame({
        "Times": times,
        "Signal": signal
    })

    fig = px.line(
        frequencyDf, 
        x = "Times", 
        y = "Signal"
    )

    return fig

#Plots spectogram using plt.specgram
def plotSpectogram (signal):
    Pxx, frequencies, bins, im = plt.specgram(signal)
    st.pyplot(plt.gcf())

#Plots spectogram in 3D
@st.cache_data
def plot3dSpectogram (times, signal):
    amplitudeFFT = np.fft.fft(signal)
    spectogramDf = pd.DataFrame({
        "Times": times,
        "Amplitude": np.abs(amplitudeFFT),
        "Frequency": signal
    })
    fig = px.line_3d(
        spectogramDf, 
        x = "Times", 
        y = "Frequency", 
        z = "Amplitude"
    )
    return fig

#Plots amplitude using signal and plotly
@st.cache_data
def plotAmplitudeFFT (time, signal):
    amplitudeFFT = np.fft.fft(signal)
    fourierDf = pd.DataFrame({
        "Times": time,
        "Amplitude": np.abs(amplitudeFFT),
    })
    return(px.line(fourierDf, x = "Times", y = "Amplitude", log_x = True))

def cacheAllData(inputFile, times, signal):
    plotFrequency(times, signal)
    plot3dSpectogram(times, signal)
    plotAmplitudeFFT(times, signal)

#Creates Streamlit Home Page
inputFile = st.file_uploader("Drag and drop files here.", type = 'wav') #Takes file input
if inputFile is not None:
    st.audio(inputFile) #Creates audio player
    times, signalArray = readAudio(inputFile)
    cacheAllData(inputFile, times, signalArray) 
    displayEmotion("sadness")

    buttonOptions = ["Frequency", "Spectogram", "Amplitude", "3D"] #Creates radio buttons for different graphs

    plot = st.radio(
        "Choose a plot:",
        buttonOptions,
        horizontal = True
    )

    if plot == buttonOptions[0]:
        st.write(plotFrequency(times, signalArray))
    elif plot == buttonOptions[1]:
        plotSpectogram(signalArray)
    elif plot == buttonOptions[2]:
        st.write(plotAmplitudeFFT(times, signalArray))
    elif plot == buttonOptions[3]:
        st.write(plot3dSpectogram(times, signalArray))