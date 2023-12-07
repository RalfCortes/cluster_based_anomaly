import numpy as np

class DownSamplerSignal:
    def __init__(self, downsampling_rate):
        self.downsampling_rate = downsampling_rate

    def downsample_signal(self, signal):
        self.initial_shape = len(signal)
        return signal[::self.downsampling_rate]

    def upsample_signal(self, signal):
        return np.repeat(signal, self.downsampling_rate, axis=0)[:self.initial_shape]