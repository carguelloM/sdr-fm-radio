import numpy as np
import sounddevice as sd
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
from scipy.signal import butter, filtfilt, lfilter, bilinear
import threading
import queue
import time 

# Configuration
SAMPLE_RATE = 5e6
READ_SIZE = 131072
CHUNK_SIZE = int(5e6)
AUDIO_SAMPLE_RATE = 44e3
CENTER_FREQ = 99e6
RADIO_FREQ = 100.5e6
B_GLOBAL = 0
BZ_GLOBAL = 0
AZ_GLOBAL = 0
A_GLOBAL = 0

class Radio:
    def __init__(self, *args, **kwargs):
        self.sdr = SoapySDR.Device(*args, **kwargs)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
        self.sdr.setFrequency(SOAPY_SDR_CF32, 0, CENTER_FREQ)
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'LNA', 16)  
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'VGA', 20)  
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'AMP', 1)   

    ## THREAD: Contiously capture samples and puts them in a buffer
    def capture_samples(self, buffer):
        ## Method that captures samples and puts them in shared buffer
        while True:
            samples = np.empty(READ_SIZE, np.complex64)
            self.sdr.readStream(self.rx_stream, [samples], len(samples))
            buffer.put(samples)

    def stop(self):
        ## Clean-up function
        self.sdr.deactivateStream(self.rx_stream)
        self.sdr.closeStream(self.rx_stream)

## THREAD: Plays Audio
def play_audio(audio_buffer):
    time.sleep(5) ## Sleep this thread fo a while to let the buffer fillup initially 
    with sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            audio_signal = audio_buffer.get()
            stream.write(audio_signal)


## FM Demodulation Funvtion (Helper of Processing Thread)
def fm_demodulate(iq_samples, sample_rate, audio_sample_rate, offset_freq):
    
    ## Shift spectrum so is centered at desired frequency
    t = np.arange(len(iq_samples)) / sample_rate
    iq_shifted = iq_samples * np.exp(2j * np.pi * offset_freq * t)

    # Butterworth low pass filter 250 khz (FM bandwith), [b,a] coefficients precalculated in main
    iq_shifted = filtfilt(B_GLOBAL, A_GLOBAL, iq_shifted)

    # Actual FM demodulation
    angle = 0.5 * np.angle(iq_shifted[1:] * np.conj(iq_shifted[:-1]))
    angle = lfilter(BZ_GLOBAL, AZ_GLOBAL, angle)

    # Decimate to reduce the sample rate to the sample rate an audio card can take
    decimation_factor = int(sample_rate / audio_sample_rate)
    audio_signal = angle[::decimation_factor]

    # Normalize and convert to int16 for audio to be played in soundcard
    audio_signal = np.int16(audio_signal / np.max(np.abs(audio_signal)) * 32767)
    return audio_signal


## THREAD: 
def processing_thread(sample_buffer, audio_buffer):
    while True:
        samples = np.concatenate([sample_buffer.get() for _ in range(int(CHUNK_SIZE / READ_SIZE))])
        audio_signal = fm_demodulate(samples, SAMPLE_RATE, AUDIO_SAMPLE_RATE, CENTER_FREQ - RADIO_FREQ)
        audio_buffer.put(audio_signal)

def main():
    sample_buffer = queue.Queue(maxsize=10)
    audio_buffer = queue.Queue(maxsize=10)

    # Create Radio object
    myRad = Radio(dict(driver="hackrf"))

    highcut = 125e3 / (SAMPLE_RATE / 2)
    b, a = butter(N=4, Wn=highcut, btype='low')
    bz, az = bilinear(1, [75e-6, 1], fs=SAMPLE_RATE)
    global BZ_GLOBAL, AZ_GLOBAL
    BZ_GLOBAL = bz
    AZ_GLOBAL = az
    global B_GLOBAL, A_GLOBAL
    B_GLOBAL = b
    A_GLOBAL = a
    # Start threads
    capture_thread = threading.Thread(target=myRad.capture_samples, args=(sample_buffer,))
    process_thread = threading.Thread(target=processing_thread, args=(sample_buffer, audio_buffer))
    playback_thread = threading.Thread(target=play_audio, args=(audio_buffer,))

    capture_thread.start()
    process_thread.start()
    playback_thread.start()

    try:
        # Keep main thread alive
        capture_thread.join()
        process_thread.join()
        playback_thread.join()
    finally:
        myRad.stop()

if __name__ == "__main__":
    main()
