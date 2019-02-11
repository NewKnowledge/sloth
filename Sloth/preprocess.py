from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from nk_logger import get_logger
from collections.abc import Iterable
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from sklearn.kernel_approximation import RBFSampler

def ScaleSeriesMeanVariance(self, series):
    return TimeSeriesScalerMeanVariance(mu = 0, std = 1).fit_transform(series)

def ScaleSeriesMinMax(self, series, min, max):
    return TimeSeriesScalerMinMax(min = min, max = max).fit_transform(series)

def events_to_rates(event_times, filter_bandwidth=3, num_bins=72, max_time=None):
    """ convert lists of event times into rate functions with a discrete time bin_size of 1/rates_per_unit.
    Uses a guassian filter over the empirical rate (histogram count / bin_size) """

    if len(event_times) == 0:  # if event times is an empty list or array
        logger.warning("empty event_times list/array")
        return np.zeros(num_bins), np.zeros(num_bins)

    # if its a nonempty list of floats, wrap in a list
    if not isinstance(event_times[0], Iterable):
        event_times = [event_times]

    if not max_time:
        max_time = max(max(times) for times in event_times)
    # NOTE: assumes min time is 0
    bins = np.linspace(0, max_time, num=num_bins + 1)
    rate_times = (bins[1:] + bins[:-1]) / 2

    counts = np.array([np.histogram(times, bins=bins)[0] for times in event_times])
    bin_size = max_time / num_bins
    sampled_rates = counts / bin_size
    rate_vals = gaussian_filter1d(sampled_rates, filter_bandwidth, mode="nearest")

    return rate_vals, rate_times

def rand_fourier_features(rate_vals, dim=1000, random_state=0):
    if rate_vals.ndim == 1:
        rate_vals = rate_vals[None, :]
    rand_fourier = RBFSampler(n_components=dim, random_state=random_state)
    return rand_fourier.fit_transform(rate_vals)