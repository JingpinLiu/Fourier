import numpy as np
import scipy
import math

def check_frequency(signal: np.ndarray,n: int = -1) -> list:
    if n == -1:
        n = len(signal)
    N = len(signal)
    result = scipy.fft.rfft(signal)
    abs_result = np.abs(result)
    peak_indices = scipy.signal.find_peaks(abs_result)[0]
    if len(peak_indices) < n:
        n=len(peak_indices)
    ordered_bin = np.flip(peak_indices[np.argsort(abs_result[peak_indices])])
    freq = ordered_bin / N
    return freq[:n]

def check_phase(signal: np.ndarray, n: int = -1) -> list:
    if n == -1:
        n = len(signal)
    result = scipy.fft.rfft(signal)
    abs_result = np.abs(result)
    args = -np.angle(result)
    peak_indices = scipy.signal.find_peaks(abs_result)[0]
    if len(peak_indices) < n:
        n=len(peak_indices)

    ordered_bin = np.flip(peak_indices[np.argsort(abs_result[peak_indices])])
    freq = ordered_bin / len(signal)
    ordered_arg = np.flip(args[peak_indices][np.argsort(abs_result[peak_indices])])
    # ordered_arg = [angle if angle >= 0 else angle + 2*np.pi for angle in ordered_arg]
    ordered_arg = [angle % (2 * np.pi) for angle in ordered_arg]
    delay = ordered_arg/(2*math.pi*freq)
    delay = np.array(delay[:n],dtype=float)
    return delay


def check_wavelength(signal: np.ndarray, n: int = -1) -> list:
    if n == -1:
        n = len(signal)
    return [int(x) for x in 1/check_frequency(signal, n)]

def check_magnitude(signal: np.ndarray,n: int = -1) -> list:
    if n == -1:
        n = len(signal)
    N = len(signal)
    result = scipy.fft.rfft(signal)
    abs_result = np.abs(result)
    peak_indices = scipy.signal.find_peaks(abs_result)[0]
    if len(peak_indices) < n:
        n=len(peak_indices)
    return (np.flip(abs_result[peak_indices][np.argsort(abs_result[peak_indices])])*(2/N))[0:n]