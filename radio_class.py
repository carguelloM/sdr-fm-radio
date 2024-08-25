import SoapySDR
import numpy as np
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

READ_SIZE = 131072

class Radio:
    def __init__(self, *args, **kwargs):
        self.sdr = SoapySDR.Device(*args, **kwargs)
        self.capturing = False
    
    def config_radio(self, samp_rate, center_freq):
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, samp_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'LNA', 16)  
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'VGA', 20)  
        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'AMP', 1)   

    def setup_radio_stream(self):
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        self.capturing = True

    ## THREAD: Contiously capture samples and puts them in the 'samples buffer'
    def capture_samples_streaming(self, buffer):
        ## Method that captures samples and puts them in shared buffer
        while self.capturing:
            samples = np.empty(READ_SIZE, np.complex64)
            self.sdr.readStream(self.rx_stream, [samples], len(samples))
            buffer.put(samples)

    def finish_capture_streaming(self):
        self.capturing = False
        
    def stop(self):
        ## Clean-up function
        self.sdr.deactivateStream(self.rx_stream)
        self.sdr.closeStream(self.rx_stream)