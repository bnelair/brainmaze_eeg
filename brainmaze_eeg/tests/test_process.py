
import pytest

import numpy as np
import scipy.signal as signal

from brainmaze_eeg.preprocessing import (
    mask_segment_with_nans,
    filter_powerline_notch,
    detect_outlier_noise,
    detect_powerline,
    detect_flat_line,
)

def test_mask_signal_with_nans():
    # Test 1D input with drop rate below threshold
    x_1d = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    dr_threshold = 0.5
    result = mask_segment_with_nans(x_1d, dr_threshold)
    assert np.isnan(result).sum() == np.isnan(x_1d).sum(), "1D input below threshold should not be masked"

    # Test 1D input with drop rate above threshold
    x_1d_high_drop = np.array([np.nan, np.nan, 3.0, np.nan, np.nan])
    dr_threshold = 0.5
    result = mask_segment_with_nans(x_1d_high_drop, dr_threshold)
    assert np.all(np.isnan(result)), "1D input above threshold should be fully masked"

    # Test 2D input with drop rate below threshold
    x_2d = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    dr_threshold = 0.5
    result = mask_segment_with_nans(x_2d, dr_threshold)
    assert np.isnan(result).sum() == np.isnan(x_2d).sum(), "2D input below threshold should not be masked"

    # Test 2D input with drop rate above threshold
    x_2d_high_drop = np.array([[np.nan, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    dr_threshold = 0.5
    result = mask_segment_with_nans(x_2d_high_drop, dr_threshold)
    assert np.all(np.isnan(result[0, :])), "2D input row above threshold should be fully masked"
    assert np.all(np.isnan(result[1, :])), "2D input row above threshold should be fully masked"

    # Test 2D input with mixed drop rates
    x_2d_mixed = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]])
    dr_threshold = 0.5
    result = mask_segment_with_nans(x_2d_mixed, dr_threshold)
    assert np.isnan(result[0, :]).sum() == 1, "First row should have one NaN"
    assert np.all(np.isnan(result[1, :])), "Second row should be fully masked"
    assert np.isnan(result).sum() == 4, "Total NaNs should be 4"


def test_filter_powerline_notch():
    # Parameters
    fs = 1000  # Sampling frequency in Hz
    duration = 2  # Duration of the signal in seconds
    frequency_powerline = 60  # Powerline frequency in Hz

    # Generate a signal with 60 Hz sinusoidal component
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal_60hz = np.sin(2 * np.pi * frequency_powerline * t)

    # Add some noise to the signal
    noise = np.random.normal(0, 0.1, signal_60hz.shape)

    signal_with_noise = signal_60hz + noise

    signal_with_nan = signal_with_noise.copy()
    signal_with_nan[100:1000] = np.nan

    # Apply the notch filter
    filtered_signal = filter_powerline_notch(signal_with_noise, fs, frequency_powerline)
    filtered_signal_with_nan = filter_powerline_notch(signal_with_nan, fs, frequency_powerline)

    assert np.isnan(filtered_signal_with_nan).sum() == np.isnan(signal_with_nan).sum(), "Filtered signal with NaNs should have the same number of NaNs"
    assert np.nansum(np.abs(filtered_signal_with_nan)) < np.nansum(np.abs(signal_with_nan)), "Filtered signal with NaNs should have smaller power than original with 60 Hz"


    # Compute the power spectrum before filtering
    freqs, psd_before = signal.welch(signal_with_noise, fs, nperseg=fs)
    freqs, psd_before_nan = signal.welch(signal_with_nan, fs, nperseg=fs)

    # Compute the power spectrum after filtering
    freqs, psd_after = signal.welch(filtered_signal, fs, nperseg=fs)

    # Find the power at 60 Hz before and after filtering
    idx_60hz = np.argmin(np.abs(freqs - frequency_powerline))
    power_before = psd_before[idx_60hz]
    power_after = psd_after[idx_60hz]

    # Assert that the power at 60 Hz is reduced after filtering
    assert power_after < power_before, f"Power at 60 Hz was not reduced: before={power_before}, after={power_after}"

    print(f"Test passed: Power at 60 Hz reduced from {power_before:.2e} to {power_after:.2e}")


def test_detect_powerline():
    # Parameters
    fs = 250  # Sampling frequency in Hz
    duration = 100  # Duration of the signal in seconds
    frequency_powerline = 60  # Powerline frequency in Hz

    # Generate a signal with 60 Hz sinusoidal component
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal_60hz = np.sin(2 * np.pi * frequency_powerline * t)

    x = 0.1*np.random.randn(signal_60hz.shape[0])
    x[:int(x.shape[0]/2)] += signal_60hz[:int(x.shape[0]/2)]

    y = detect_powerline(x, fs, detection_window=1.0, powerline_freq=60)
    assert y.sum() == y.shape[0] / 2

    x_merged = np.stack([x, x[::-1]], 0)
    y_merged = detect_powerline(x_merged, fs, detection_window=1.0, powerline_freq=60)
    assert np.all(y_merged[0] == y)
    assert np.all(y_merged[1] == y[::-1])


def test_detect_outlier_noise():
    fs = 250  # Sampling frequency in Hz
    duration = 60  # Duration of the signal in seconds
    idx = np.arange(fs+10, fs + 20)

    x = 0.1 * np.random.randn(int(fs * duration))
    b, a = signal.butter(4, 50, btype='low', fs=fs)
    x = signal.filtfilt(b, a, x)

    x[idx] = 1e3

    x = np.stack([x, x[::-1]], 0)

    y = detect_outlier_noise(x, fs, detection_window=1)

    assert y.shape == (2, duration), "Output shape should be (2, duration)"
    assert y[0, 0] == False, "1st second segment should not be detected as not noise"
    assert y[0, 1] == True, "2nd second segment should be detected as noise"
    assert y[0, 2] == False, "3rd second segment should not be detected as noise"
    assert np.all(y[0] == y[1][::-1]), "Output should be the same for both channels"


def test_detect_flat_line():
    fs = 250
    duration = 60

    x = 1*np.random.randn(int(fs * duration))
    x[fs:2*fs] *= 1e-6

    x = np.stack([x, x[::-1]], 0)

    y = detect_flat_line(x, fs, detection_window=1)

    assert y.shape == (2, duration,), "Output shape should be (duration,)"
    assert y[0, 0] == False, "1st second segment should not be detected as flat line"
    assert y[0, 1] == True, "2nd second segment should be detected as flat line"
    assert y[0, 2] == False, "3rd second segment should not be detected as flat line"
    assert np.all(y[0] == y[1][::-1]), "Output should be the same for both channels"







