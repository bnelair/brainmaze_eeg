
import numpy as np
import scipy.signal as signal
from typing import Tuple
from scipy.ndimage import binary_dilation

from brainmaze_utils.signal import PSD, buffer


def channel_data_rate_thresholding(x: np.typing.NDArray[np.float64], threshold_data_rate: float=0.1):
    """
    Masks entire channels (sets to NaN) based on data availability.

    Assesses the proportion of non-NaN values (data rate) for each channel. Channels
    with a data rate at or below the specified `threshold_data_rate` are fully
    masked, resulting in the output signal having those channels entirely as NaN.
    This filters out channels with excessive missing data for quality control.

    Parameters:
        x (np.ndarray): Input signal array, expected to be [n_channels, n_samples] or [n_samples].
                        May contain NaN values.
        threshold_data_rate (float, optional): Minimum acceptable proportion of non-NaNs
            for a channel to be kept. Channels <= this rate are masked. Default is 0.1.

    Returns:
        np.ndarray: The input signal with channels below the data rate threshold
            set entirely to NaN. Shape is the same as the input (or the original
            1D shape if input was 1D).

    Raises:
        ValueError: If input is not 1D or 2D.
    """

    ndim = x.ndim

    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]  # Add a new axis to make it 2D

    ch_mask = 1 - (np.isnan(x).sum(axis=1) / x.shape[1]) <= threshold_data_rate
    x[ch_mask, :] = np.nan

    if ndim == 1:
        x = x[0]

    return x


def replace_nans_with_median(x: np.typing.NDArray[np.float64]):
    """
    Imputes NaN values per channel with the median of valid data.

    Replaces NaN values by computing the median of non-NaNs for each channel
    independently and filling the NaNs with this channel-specific median.
    This provides a robust way to fill missing data points. Returns the
    processed signal and a boolean mask indicating the original NaN locations.

    Parameters:
        x (np.ndarray): Input signal array [n_channels, n_samples] or [n_samples], can have NaNs.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - processed_signal (np.ndarray): Signal with NaNs replaced by channel medians.
            - mask (np.ndarray): Boolean mask where True indicates original NaN positions.

    Raises:
        ValueError: If input is not 1D or 2D.
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


def filter_powerline(x: np.typing.NDArray[np.float64], fs: float, powerline_freq: float=60):
    """
    Removes powerline noise using a notch filter and handles NaNs.

    Applies a notch filter at the specified powerline frequency. Handles NaNs
    by temporarily imputing with the median before filtering and then restoring
    the original NaN locations in the output. Note potential ringing artifacts
    near original NaN gaps or sharp signal transitions.

    Parameters:
        x (np.ndarray): Input signal [n_channels, n_samples] or [n_samples], can have NaNs.
        fs (float): Sampling frequency (Hz).
        powerline_freq (float, optional): Frequency of noise to remove. Default is 60 Hz.

    Returns:
        np.ndarray: Filtered signal with powerline noise attenuated. Original NaN
            locations are preserved. Same shape as input.

    Raises:
        ValueError: If input is not 1D or 2D.
    """
    # substitute nans with median for 60Hz notch filter

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    mask = np.isnan(x)
    x = np.where(mask, np.nanmedian(x, axis=1, keepdims=True), x)

    b, a = signal.iirnotch(w0=powerline_freq, Q=10, fs=fs)
    x = signal.filtfilt(b, a, x, axis=1)

    x[mask] = np.nan

    if ndim == 1:
        x = x[0]

    return x


def detect_powerline_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        window_s: float = 0.5,
        powerline_freq:float = 60,
        threshold:float = 1000
):
    """
    Detects time segments within each channel affected by powerline noise.

    Identifies segments by analyzing the spectral power ratio between the powerline
    frequency/harmonics and the 2-40Hz band within short time windows. Segments
    where this ratio exceeds a specified threshold are flagged. Operates on
    segments; drops partial segments at the end. Returns a boolean mask.

    Parameters:
        x (np.ndarray): Input signal [n_channels, n_samples] or [n_samples], can have NaNs.
        fs (float): Sampling frequency (Hz).
        window_s (float, optional): Analysis window duration in seconds. Default is 0.5.
        powerline_freq (float, optional): Fundamental powerline frequency. Harmonics also checked. Default is 60 Hz.
        threshold (float, optional): Power ratio threshold for flagging segments. Default is 1000.

    Returns:
        np.ndarray: Boolean mask [n_channels, n_segments] or [n_segments]. True indicates
            powerline noise detected in that specific segment and channel.

    Raises:
        ValueError: If input is not 1D or 2D.
    """


    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    xb =  np.array([
        buffer(x_, fs, segm_size=window_s, drop=True) for x_ in x
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
    detected_noise = pow_rat >= threshold

    if ndim == 1:
        detected_noise = detected_noise[0]

    return detected_noise


def detect_outlier_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        window_s: float = 0.5,
        threshold: float = 10
):
    """
    Detects time segments containing amplitude outliers.

    Identifies segments with sudden, large amplitude deflections by applying
    a robust percentile-based threshold to the signal within short time windows.
    Segments where samples exceed the threshold are flagged. Operates on segments.
    Returns a boolean mask indicating flagged segments.

    Parameters:
        x (np.ndarray): Input signal [n_channels, n_samples] or [n_samples], can have NaNs.
        fs (float): Sampling frequency (Hz).
        window_s (float, optional): Analysis window duration in seconds. Default is 0.5.
        threshold (float, optional): Multiplier for percentile range to set threshold. Default is 10.

    Returns:
        np.ndarray: Boolean mask [n_channels, n_segments] or [n_segments]. True indicates
            amplitude outliers detected in that specific segment and channel.

    Raises:
        ValueError: If input is not 1D or 2D.
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
        buffer(b_ch, fs, segm_size=window_s, drop=True).sum(1) > 1 for b_ch in b_idx
    ])

    if ndim == 1:
        detected_noise = detected_noise[0]

    return detected_noise

def detect_flat_line_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        window_s:float = 0.5,
        threshold: float = 0.5e-6
):
    """
    Detects flat-line segments in the signal based on low variability.

    Identifies periods where the signal is constant by checking if the mean
    absolute difference within short segments falls below a threshold. Operates
    on segments. Returns a boolean mask indicating flagged segments.

    Parameters:
        x (np.ndarray): Input signal [n_channels, n_samples] or [n_samples], can have NaNs.
        fs (float): Sampling frequency (Hz).
        window_s (float, optional): Analysis window duration in seconds. Default is 0.5.
        threshold (float, optional): Threshold for mean absolute difference to flag flat-line. Default is 0.5e-6.

    Returns:
        np.ndarray: Boolean mask [n_channels, n_segments] or [n_segments]. True indicates
            a flat-line detected in that specific segment and channel.

    Raises:
        ValueError: If input is not 1D or 2D.
    """

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    xb = np.array([
        buffer(x_, fs, segm_size=window_s, drop=True) for x_ in x
    ])
    detected_flat_line = np.abs(np.diff(xb, axis=2).mean(axis=2)) < threshold

    if ndim == 1:
        detected_flat_line = detected_flat_line[0]

    return detected_flat_line


def detect_stim_segments(x: np.typing.NDArray[np.float64], fs: float, window_s:float = 1,
                         threshold:float = 2000, freq_band: Tuple[float, float] = (80, 110,)):
    """
        Detects stimulation artifacts using spectral analysis of the difference signal.

        Identifies artifacts by checking for high spectral power in a specific
        high-frequency band (e.g., 80-110 Hz) within short time windows of the
        signal's derivative. Operates on segments, drops partial end segments.
        Returns a boolean mask of detected segments and the calculated power sums.

        Parameters:
            x (np.ndarray): Input signal [n_channels, n_samples] or [n_samples], can have NaNs.
            fs (float): Sampling frequency (Hz).
            window_s (float, optional): Analysis window duration in seconds. Default is 1.
            threshold (float, optional): Power sum threshold for flagging artifacts. Default is 2000.
            freq_band (tuple, optional): Frequency range (low, high in Hz) for artifact power check. Default is (80, 110).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - detected_stim (np.ndarray): Boolean mask [n_channels, n_segments] or [n_segments]. True indicates artifact detected.
                - psd_sum (np.ndarray): Sum of spectral power in `freq_band` per segment/channel.

        Raises:
            ValueError: If input is not 1D or 2D.
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
        buffer(x_, fs, segm_size=window_s, drop=True) for x_ in x_diff
    ])


    freq, psd = PSD(xb, fs=fs)
    psd_hf = psd[:, :, (freq > freq_band[0]) & (freq < freq_band[1])]
    psd_sum = np.sum(psd_hf, axis=-1)
    detected_stim = (psd_sum >= threshold).astype(int)

    if ndim == 1:
        detected_stim = detected_stim[0]
        psd_sum = psd_sum[0]

    return detected_stim, psd_sum


def mask_segments_with_nans(x: np.typing.NDArray[np.float64], segment_mask: np.typing.NDArray[np.float64],
                            fs: float, segment_len_s: float):
    """
        Masks (sets to NaN) signal segments specified by a boolean mask.

        Applies a pre-computed boolean/integer mask to set corresponding time
        segments in the signal to NaN. Converts segment indices from the mask
        to sample indices to apply masking. Returns a copy of the input signal
        with artifactual segments replaced by NaN.

        Parameters:
            x (np.ndarray): Input signal [n_channels, n_samples] or [n_samples].
            segment_mask (np.ndarray): Boolean/int mask [n_channels, n_segments] or [n_segments].
                                       True/1 flags segments to mask.
            fs (float): Sampling frequency (Hz).
            segment_len_s (float): Duration of each segment in seconds, matches mask resolution.

        Returns:
            np.ndarray: Copy of input signal with specified segments replaced by NaN. Same shape.

        Raises:
            ValueError: If input is not 1D/2D or mask dimension mismatch.
        """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if segment_mask.ndim != ndim:
        raise ValueError("Input 'merged_noise' must have same dimension as input signal 'x'.")

    if x.ndim == 1:
        x = x[np.newaxis, :]
        segment_mask = segment_mask[np.newaxis, :]


    n_channels, n_samples = x.shape
    samples_per_segment = int(np.round(fs * segment_len_s))

    #  # Create index offsets for each segment
    window_len = segment_mask.shape[1]
    segment_indices = np.arange(window_len) * samples_per_segment
    segment_range = np.arange(samples_per_segment)

    # Find all artifact locations
    segment_offsets = segment_range[None, :] + segment_indices[:, None]     #shape: (n_seconds, samples_per_segment)
    channel_idx, second_idx = np.where(segment_mask == 1)
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


def detection_dilatation(mask: np.ndarray, extend_left: int = 2, extend_right: int = 2):
    """
    Expands detected regions in a boolean/integer mask using binary dilation.

    Applies binary dilation to a mask, effectively widening the regions marked
    as True (or 1) by a specified number of positions to the left and right.
    This is useful post-detection to add a buffer around flagged segments,
    accounting for potential edge effects. Returns the expanded mask.

    Parameters:
        mask (np.ndarray): 1D or 2D boolean/int mask [n_channels, n_segments] or [n_segments]. True/1 indicates detection.
        extend_left (int, optional): Positions to extend the True region to the left. Default is 2.
        extend_right (int, optional): Positions to extend the True region to the right. Default is 2.

    Returns:
        np.ndarray: The expanded boolean mask (as int 0/1). Same shape as input.

    Raises:
        ValueError: If input mask is not 1D or 2D.
    """
    total_extend = extend_left + extend_right + 1
    structure = np.ones(total_extend, dtype=int)

    if mask.ndim == 1:
        return binary_dilation(mask, structure=structure, origin=0).astype(int)
    elif mask.ndim == 2:
        return np.array([
            binary_dilation(row, structure=structure, origin=0)
            for row in mask
        ]).astype(int)
    else:
        raise ValueError("Input 'mask' must be 1D or 2D.")
