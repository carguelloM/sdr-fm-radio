import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt, lfilter, bilinear
import threading
import queue
import time 
import sys
from radio_class import Radio


# HARDWARE CONSTANTS   
SAMPLE_RATE = 5e6
READ_SIZE = 131072
CHUNK_SIZE = int(5e6)
AUDIO_SAMPLE_RATE = 44e3
CENTER_FREQ = 0
RADIO_FREQ = 0

## FILTER CONSTANTS 
B_GLOBAL = 0
BZ_GLOBAL = 0
AZ_GLOBAL = 0
A_GLOBAL = 0


STOP_THREADS = False



## THREAD: Plays Audio (takes out of 'audio buffer' and gives it to sound card)
def play_audio(audio_buffer):
    time.sleep(5) ## Sleep this thread fo a while to let the buffer fillup initially 
    with sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while not STOP_THREADS:
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

    ## FM De-emphasis filter, H(s) = 1/(RC*s + 1), implemented as IIR via bilinear transform
    ## Credits: https://pysdr.org/content/rds.html
    angle = lfilter(BZ_GLOBAL, AZ_GLOBAL, angle)

    # Decimate to reduce the sample rate to the sample rate an audio card can take
    decimation_factor = int(sample_rate / audio_sample_rate)
    audio_signal = angle[::decimation_factor]

    # Normalize and convert to int16 for audio to be played in soundcard
    audio_signal = np.int16(audio_signal / np.max(np.abs(audio_signal)) * 32767)
    return audio_signal


## THREAD: Takes samples out of 'sample buffer' in particular chunks, demodulates the chunk
## and puts chunk in audio buffer
def processing_thread(sample_buffer, audio_buffer):
    while not STOP_THREADS:
        samples = np.concatenate([sample_buffer.get() for _ in range(int(CHUNK_SIZE / READ_SIZE))])
        audio_signal = fm_demodulate(samples, SAMPLE_RATE, AUDIO_SAMPLE_RATE, CENTER_FREQ - RADIO_FREQ)
        audio_buffer.put(audio_signal)

def print_usage():
    print('Usage: fm-radio-sdr [STATION IN MHZ]')
    exit(1)


def main():

    ## check args
    if len(sys.argv) < 2:
        print_usage()
    
    ## get frequency
    try:
        fm_freq = float(sys.argv[1])
    except:
       print_usage()

    if fm_freq < 88 or fm_freq > 108:
        print('Valid FM Radio Freq are [88,108] MHz')
        exit(1)
    
    global CENTER_FREQ, RADIO_FREQ
    RADIO_FREQ = fm_freq * 1e6
    ## some sdrs have DC spikes, to avoid we set the 
    ## center freq a 1MHz lower
    CENTER_FREQ = RADIO_FREQ - 1e6 
    

    print(f'Tunning to station: {fm_freq} FM...wait a sec!')
    sample_buffer = queue.Queue(maxsize=10)
    audio_buffer = queue.Queue(maxsize=10)

    # Create Radio object
    myRad = Radio(dict(driver="hackrf"))
    myRad.config_radio(SAMPLE_RATE, CENTER_FREQ)
    myRad.setup_radio_stream()

    ## Butterworth LP filter parameters to get only FM radio bandwith
    highcut = 125e3 / (SAMPLE_RATE / 2)
    b, a = butter(N=4, Wn=highcut, btype='low')
    global B_GLOBAL, A_GLOBAL
    B_GLOBAL = b
    A_GLOBAL = a

    ## Bilinear trandor FM de-empahsis filter paramerters 
    bz, az = bilinear(1, [75e-6, 1], fs=SAMPLE_RATE)
    global BZ_GLOBAL, AZ_GLOBAL
    BZ_GLOBAL = bz
    AZ_GLOBAL = az
    
    # Start threads
    capture_thread = threading.Thread(target=myRad.capture_samples_streaming, args=(sample_buffer,))
    process_thread = threading.Thread(target=processing_thread, args=(sample_buffer, audio_buffer))
    playback_thread = threading.Thread(target=play_audio, args=(audio_buffer,))

    capture_thread.start()
    process_thread.start()
    playback_thread.start()

    global STOP_THREADS
    
    try:
       ## keep alive main thread until keyboard interrupt is recv
       ## maybe better wait to do this? effectively this will never
       ## happen
       capture_thread.join()
      
       
    except KeyboardInterrupt:
        ## Stop the threads
        STOP_THREADS = True
        ## stop radio streaming
        myRad.finish_capture_streaming()
        
        print('Cleaning up...Please wait! :)')
        ## wait for threads to finsih
        capture_thread.join()
        process_thread.join()
        playback_thread.join()

        ## stop th radio
        myRad.stop()

if __name__ == "__main__":
    main()
