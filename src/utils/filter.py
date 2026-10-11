import numpy as np
from scipy.signal import butter, lfilter


def remove_outliers(audio, patience):
    print("Removing outliers...")
    threshold = np.mean(audio) + patience * np.std(audio)
    return np.where(audio > threshold, threshold, audio)


def bp_butter_filter(y, sr, order, highcut, lowcut):
    print("Filtering signals...")
    sample_rate = sr
    lowcut = lowcut / (sample_rate / 2)
    highcut = highcut / (sample_rate / 2)

    b, a = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        analog=True,
    )

    y = lfilter(b, a, y)

    return y
