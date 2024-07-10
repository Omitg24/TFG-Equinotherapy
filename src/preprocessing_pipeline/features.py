import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import kurtosis, skew


def aom(data: np.ndarray) -> float:
    # abs(max(data) - min(data))
    if len(data) <= 0:
        return 0
    return abs(max(data) - min(data))


def angle(data_x, data_y, data_z):
    # atan(z_median / sqrt(x_median^2 + y_median^2))
    if len(data_x) <= 0:
        return 0
    x_median = np.median(data_x)
    y_median = np.median(data_y)
    z_median = np.median(data_z)
    return np.degrees(np.arctan(z_median / ((x_median ** 2 + y_median ** 2) ** 0.5)))


def aom3_seconds(data):
    # AOM calculated for 3-second segments
    return [aom(segment) for segment in np.array_split(data, len(data) // 30)]


def angle3_seconds(data_x, data_y, data_z):
    #  Angle calculated for 3-second segments
    if len(data_x) != len(data_y) or len(data_y) != len(data_z):
        print("ERROR: Invalid data length")
        raise ValueError("Invalid data length")

    ids = np.ceil(np.arange(len(data_x)) / 30).astype(int)
    data_x_split = np.split(data_x, np.unique(ids))
    data_y_split = np.split(data_y, np.unique(ids))
    data_z_split = np.split(data_z, np.unique(ids))

    angle_vals = [angle(x, y, z) for x, y, z in zip(data_x_split, data_y_split, data_z_split)]
    return angle_vals


def feature_correlation(data_x, data_y):
    # sum((x - mean(x)) * (y - mean(y)) / (std(x) * std(y))) / (n - 1)
    mean_x, std_x = np.mean(data_x), np.std(data_x)
    mean_y, std_y = np.mean(data_y), np.std(data_y)

    corr_values = ((data_x - mean_x) * (data_y - mean_y)) / (std_x * std_y)
    return np.sum(corr_values) / (len(data_x) - 1) if len(data_x) > 1 else 0


def feature_kurtosis(data):
    # (E[(m-MEANm)^4] / (STDm)^2) - 3
    return kurtosis(data)


def feature_crest_factor(data):
    # max(data) / sqrt(sum(data^2) / (n - 1))
    return np.max(data) / np.sqrt(np.sum(np.square(data)) / (len(data) - 1)) if len(data) > 1 else 0


def feature_skewness(data):
    # E[(m-MEANm/STDm)^3]
    return skew(data)


def feature_zero_crossing(data):
    # |(2 ≤ n ≤ N ) ∧ (m(n) ∗ m(n − 1) < 0)|
    crosses, _ = find_peaks(data)
    return len(crosses) if len(crosses) > 0 else 0


def feature_entropy(data):
    # -sum(p * log2(p))
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))


def feature_band_energy(data, freq):
    # sum(sub_band) / sum(data)
    nyq = 0.5 * freq
    low = 0.25 / nyq
    high = 3.0 / nyq
    b, a = butter(2, [low, high], btype='band', fs=freq)
    sub_band = filtfilt(b, a, data)

    data_sum = np.sum(data)
    if data_sum == 0:
        data_sum += np.finfo(float).eps
    band_energy = np.sum(sub_band) / data_sum
    return band_energy


def feature_spectral_flux(data):
    # sum(diff(data)^2)
    if len(data) <= 1:
        return 0
    flux_values = np.square(np.diff(data))
    return np.sum(flux_values)
