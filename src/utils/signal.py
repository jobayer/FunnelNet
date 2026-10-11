import numpy as np
import scipy.signal as scipy_sig
from scipy.io import wavfile
from tensorflow import abs
from tensorflow.signal import stft  # type: ignore
from tqdm import tqdm
import pywt


def extract_audio_data(data, target_sr=None):
    signals = []
    with tqdm(total=len(data), desc="Extracting audio data") as pbar:
        for i in range(len(data)):
            audio = data[i]
            _, sig = wavfile.read(audio)
            sig = np.float64(sig)
            np.divide(sig, np.float64(32768), out=sig)

            if target_sr is not None:
                assert target_sr > 0, "Target sampling rate must be positive."
                sig = scipy_sig.resample(x=sig, num=target_sr)

            signals.append(sig)
            pbar.update(1)

    return signals


def remove_inconsistent_signals(data, label):
    f_shape = None

    new_data = []
    new_label = []

    for i in range(len(data)):
        s = data[i].shape
        if f_shape is None:
            f_shape = s
            new_data.append(data[i])
            new_label.append(label[i])
        elif s == f_shape:
            new_data.append(data[i])
            new_label.append(label[i])

    return new_data, new_label

def extract_signals(data, window=32):
    signals = []
    data_size = len(data)
    widths = np.geomspace(1, window, window)

    for i in tqdm(range(data_size), "Extracting signals"):
        sig = np.array([pywt.cwt(data[i], [w], 'morl')[0][0] for w in widths]).T
        sig = abs(sig)
        sig = sig.numpy()
        sig = sig.reshape(sig.shape[0], sig.shape[1], 1)

        signals.append(sig)

    return signals
