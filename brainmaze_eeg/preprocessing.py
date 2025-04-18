
import numpy as np
import scipy.signal as signal

from brainmaze_utils.signal import PSD, buffer


def mask_segment_with_nans(x: np.typing.NDArray[np.float64], data_rate_threshold: float=0.1):
    """
    Masks signals with NaNs based on a nan rate threshold. If data rate (non-nan values) is below the threshold,
    the entire row is masked with NaNs.

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


def filter_powerline_notch(x: np.typing.NDArray[np.float64], fs: float, frequency_powerline: float=60):
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


def detect_powerline(
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



def detect_outlier_noise(
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


def detect_flat_line(
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


