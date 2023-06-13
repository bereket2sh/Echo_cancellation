# Import libraries
import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import pyroomacoustics as pra
import matplotlib.pyplot as plt

# Import adaptive filtering algorithms
from time_domain_adaptive_filters.lms import lms
from time_domain_adaptive_filters.nlms import nlms

def main():
    st.title("Adaptive Filtering Web App")

    file_x = st.file_uploader("Upload input audio file (x)", type=["wav"])
    file_d = st.file_uploader("Upload desired audio file (d)", type=["wav"])

    if file_x and file_d:
        st.write("Files uploaded successfully!")
        x, sr_x = librosa.load(file_x, sr=8000)
        d, sr_d = librosa.load(file_d, sr=8000)

        if sr_x != sr_d:
            st.write("Sampling rates of the two files don't match. Please upload files with the same sampling rate.")
            return

        # Ensure the same length for x and d
        min_len = min(len(x), len(d))
        x = x[:min_len]
        d = d[:min_len]

        # Filter the signals using LMS
        e_lms = lms(x, d, N=256, mu=0.1)
        e_lms = np.clip(e_lms, -1, 1)

        # Filter the signals using NLMS
        e_nlms = nlms(x, d, N=256, mu=0.1)
        e_nlms = np.clip(e_nlms, -1, 1)

        # Generate time array with matching length
        time = np.arange(min_len) / sr_x

        # Plotting the signals
        fig, ax = plt.subplots()

        ax.plot(time, x, label='Base Signal')
        ax.plot(time, e_lms[:min_len], label='LMS Signal')
        ax.plot(time, e_nlms[:min_len], label='NLMS Signal')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Base Signal vs. LMS Signal vs. NLMS Signal')
        ax.legend()

        # Display the plot in the Streamlit app
        st.pyplot(fig)

if __name__ == '__main__':
    main()
