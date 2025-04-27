
import numpy as np
import scipy.signal as signal
from typing import Tuple

from brainmaze_utils.signal import PSD, buffer


def channel_data_rate_thresholding(x: np.typing.NDArray[np.float64], data_rate_threshold: float=0.1):
    """
    Masks the whole channel [nchans, nsamples] with nans if the channel data rate is below the threshold.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        dr_threshold (float, optional): Drop rate threshold for masking. Default is 0.1.

    Returns:
        np.ndarray: Signal with masked values replaced by NaNs.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """

    ndim = x.ndim

    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]  # Add a new axis to make it 2D

    ch_mask = 1 - (np.isnan(x).sum(axis=1) / x.shape[1]) <= data_rate_threshold
    x[ch_mask, :] = np.nan

    if ndim == 1:
        x = x[0]

    return x


def replace_nans_with_median(x: np.typing.NDArray[np.float64]):
    """
    Replaces NaN values in the input signal with the median of the non-NaN values along each channel.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - processed_signal (np.ndarray): Signal with NaN values replaced by the median.
            - data_rate (np.ndarray): Data rate for each channel, calculated as the proportion of non-NaN values.
            - mask (np.ndarray): Boolean mask indicating the positions of NaN values in the original signal.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """

    ndim = x.ndim

    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]  # Add a new axis to make it 2D

    mask = np.isnan(x)

    if not mask.any(): # if no nans, just return
        if ndim == 1:
            x = x[0]
            mask = mask[0]

        return x, mask

    med_vals = np.nanmedian(x, axis=1, keepdims=True)
    x = np.where(mask, med_vals, x)

    if ndim == 1:
        x = x[0]

    return x, mask


def filter_powerline(x: np.typing.NDArray[np.float64], fs: float, frequency_powerline: float=60):
    """
    Filters powerline noise from the input signal using a notch filter. The function replaces NaN values with the
    median and returns nan values after filtering. This can possibly cause ringing around artifacts and edges.

    """
    # substitute nans with median for 60Hz notch filter

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    mask = np.isnan(x)
    x = np.where(mask, np.nanmedian(x, axis=1, keepdims=True), x)

    b, a = signal.iirnotch(w0=frequency_powerline, Q=10, fs=fs)
    x = signal.filtfilt(b, a, x, axis=1)

    x[mask] = np.nan

    if ndim == 1:
        x = x[0]

    return x


def detect_powerline_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        detection_window: float = 0.5,
        powerline_freq:float = 60,
        threshold_ratio:float = 1000
):
    """
    Detects Powerline noise in the input signal using. Detection evaluates the power in the spectrum at
    powerline and its harmonics to the average power of the iEEG in 2 Hz - 40 Hz band. It
    drops the last segment if ndarray shape is not a multiple of whole seconds.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        detection_window (float): Length of the segment in seconds. Default is 0.5 seconds.
        powerline_freq (float): Frequency of the powerline noise. Default is 60 Hz.
        threshold_ratio (float): Threshold ratio for detection how many times the power of the powerline noise is higher than average power in 2 Hz - 40 Hz band. Default is 1000.

    Returns:
        np.ndarray: Boolean array indicating the presence of powerline noise for every 1 second segment.

    """


    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    xb =  np.array([
        buffer(x_, fs, segm_size=detection_window, drop=True) for x_ in x
    ])
    xb = xb - np.nanmean(xb, axis=2, keepdims=True)
    f, pxx = PSD(xb, fs)

    max_freq = f[-1]

    idx_lower_band = (f>=2) & (f <= 40)
    pow_40 = np.nanmean(pxx[:, :, idx_lower_band], axis=2, keepdims=True) # since we always buffer 1 second, we can use absolute indexes

    idx_pline = np.array([
        np.where((f >= f_det -2) & (f <= f_det + 2))[0] for f_det in np.arange(powerline_freq, max_freq, powerline_freq)
    ]).flatten()
    idx_pline = np.round(idx_pline).astype(np.int64)

    pow_pline = np.nanmax(pxx[:, :, idx_pline], axis=2, keepdims=True)

    pow_rat = pow_pline / pow_40

    pow_rat = pow_rat.squeeze(axis=2)
    detected_noise = pow_rat >= threshold_ratio

    if ndim == 1:
        detected_noise = detected_noise[0]

    return detected_noise


def detect_outlier_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        detection_window: float = 0.5,
        threshold: float = 10
):
    """
    Detects outlier noise in the input signal based on a threshold. The function evaluates the signal's deviation
    from the mean and identifies segments with excessive noise. It drops the last segment if ndarray shape
    is not a multiple of whole seconds.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        detection_window (float): Length of the segment in seconds. Default is 0.5 seconds.
        threshold (float): Threshold for detecting outliers. Default is 10.

    Returns:
        np.ndarray: Boolean array indicating the presence of outlier noise for each segment.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    x = x - np.nanmean(x, axis=1, keepdims=True)
    threshold_tukey = np.abs(np.nanpercentile(x, 90, axis=1) + \
         threshold * (np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)))

    b_idx = np.abs(x) > threshold_tukey[:, np.newaxis]

    detected_noise = np.array([
        buffer(b_ch, fs, segm_size=detection_window, drop=True).sum(1) > 1 for b_ch in b_idx
    ])

    if ndim == 1:
        detected_noise = detected_noise[0]

    return detected_noise

def detect_flat_line_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        detection_window:float = 0.5,
        threshold: float = 0.5e-6
):
    """
    Detects flat-line segments in the input signal. A flat-line segment is identified when the mean absolute
    difference of the signal within a detection window is below a specified threshold.  It
    drops the last segment if ndarray shape is not a multiple of whole seconds.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        detection_window (float): Length of the segment in seconds. Default is 0.5 seconds.
        threshold (float): Threshold for detecting flat-line segments. Default is 0.5e-6.

    Returns:
        np.ndarray: Boolean array indicating the presence of flat-line segments for each detection window.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    xb = np.array([
        buffer(x_, fs, segm_size=detection_window, drop=True) for x_ in x
    ])
    detected_flat_line = np.abs(np.diff(xb, axis=2).mean(axis=2)) < threshold

    if ndim == 1:
        detected_flat_line = detected_flat_line[0]

    return detected_flat_line


def detect_stim_segments(x: np.typing.NDArray[np.float64], fs: float, detection_window:float = 1,
                         detection_threshold:float = 2000, freq_band: Tuple[float, float] = (80, 110,)):
    """
    Detects stimulation artifacts in the input signal. Calculates differential signal of the input signal.
    Spectral power of the differential signal between the bands provided in frequency band is
    thresholded  for each detection window.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        detection_window (float): Length of the segment in seconds. Default is 1 second.
        detection_threshold (float): Threshold for detecting stimulation artifacts. Default is 2000.
        freq_band (tuple): Frequency band to consider for artifact detection (low, high). Default is (80, 110).

    Returns:
        tuple:
            - np.ndarray: Boolean array indicating the presence of stimulation artifacts for each detection window.
            - np.ndarray: Spectral power within the specified frequency band for each detection window.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    x_diff = np.diff(x, axis=-1)     # difference signal highlights artificial pulses
    x_diff = np.concat(
        (x_diff, x_diff[:, -1].reshape(-1, 1)), axis=1,
    )

    xb =  np.array([
        buffer(x_, fs, segm_size=detection_window, drop=True) for x_ in x_diff
    ])


    freq, psd = PSD(xb, fs=fs)
    psd_hf = psd[:, :, (freq > freq_band[0]) & (freq < freq_band[1])]
    psd_sum = np.sum(psd_hf, axis=-1)
    detected_stim = (psd_sum >= detection_threshold).astype(int)

    if ndim == 1:
        detected_stim = detected_stim[0]
        psd_sum = psd_sum[0]

    return detected_stim, psd_sum


def mask_segments_with_nans(x: np.typing.NDArray[np.float64], merged_noise: np.typing.NDArray[np.float64],
                            fs: float, n_sec: float):
    """
    Masks EEG signal segments with noise and stimulation artifacts by setting them to NaN.

    Parameters:
        x (np.ndarray): 1D or 2D array of EEG data with shape (n_channels, n_samples).
        fs (int): Sampling rate of the EEG signal in Hz.
        n_sec (int): Number of seconds in the EEG signal.
        merged_noise (np.ndarray): Binary matrix of shape (n_channels, n_sec) where 1 indicates
                                       the presence of a stimulation artifact in that second.

    Returns:
        np.ndarray: EEG signal with artifact segments replaced by NaN.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if merged_noise.ndim != ndim:
        raise ValueError("Input 'merged_noise' must have same dimension as input signal 'x'.")

    if x.ndim == 1:
        x = x[np.newaxis, :]
        merged_noise = merged_noise[np.newaxis, :]


    n_channels, n_samples = x.shape
    samples_per_segment = (fs * n_sec) // merged_noise.shape[1]

    #  # Create index offsets for each segment
    window_len = merged_noise.shape[1]
    segment_indices = np.arange(window_len) * samples_per_segment
    segment_range = np.arange(samples_per_segment)

    # Find all artifact locations
    segment_offsets = segment_range[None, :] + segment_indices[:, None]     #shape: (n_seconds, samples_per_segment)
    channel_idx, second_idx = np.where(merged_noise == 1)
    sample_indices = segment_offsets[second_idx]  # shape: (num_artifacts, samples_per_segment)

    # Filter out segments that would exceed signal bounds
    valid_mask = sample_indices[:, -1] < n_samples
    channel_idx = channel_idx[valid_mask]
    sample_indices = sample_indices[valid_mask]

    # Apply NaNs to the artifact regions
    x_sub = x.copy()
    x_sub[channel_idx[:, None], sample_indices] = np.nan

    if ndim == 1:
        x_sub = x_sub[0]
    return x_sub
