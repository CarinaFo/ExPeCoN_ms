import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd

from expecon_ms.utils import zero_pad_or_mirror_epochs


def create_signal_with_beta_drop(sfreq, tmin, tmax):
    """
    Create a signal with a beta power drop around stimulus onset.
    
    Args:
        sfreq: Sampling frequency.
        tmin: Minimum time.
        tmax: Maximum time.
        
    Returns:
        times: Time array.
        signal: Composite signal with alpha and beta components.
    """
    # Create MNE info structure
    sfreq = 100.0  # Lower sampling rate for simulation
    ch_names = ["SIM0001"]
    ch_types = ["eeg"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Simulation parameters
    n_times = 224  # Just over 2 second epochs
    n_epochs = 40
    seed = 42
    rng = np.random.RandomState(seed)
    data = rng.randn(len(ch_names), n_times * n_epochs + 200)  # Buffer for mirroring

    # Time vector for one epoch
    t = np.arange(n_times, dtype=np.float64) / sfreq

    # Simulate a 20 Hz (Beta) sinusoidal burst between 900ms and 1200ms
    signal_beta = np.sin(np.pi * 2.0 * 20.0 * t)  # 20 Hz sinusoid (beta band)
    signal_beta[np.logical_or(t < 0.9, t > 1.2)] = 0.0  # Hard windowing for beta
    on_time_beta = np.logical_and(t >= 0.9, t <= 1.2)
    signal_beta[on_time_beta] *= np.hanning(on_time_beta.sum())  # Ramping for beta

    # Simulate a 10 Hz (Alpha) sinusoidal burst between 1400ms and 1800ms
    signal_alpha = np.sin(np.pi * 2.0 * 10.0 * t)  # 10 Hz sinusoid (alpha band)
    signal_alpha[np.logical_or(t < 1.4, t > 1.8)] = 0.0  # Hard windowing for alpha
    on_time_alpha = np.logical_and(t >= 1.4, t <= 1.8)
    signal_alpha[on_time_alpha] *= np.hanning(on_time_alpha.sum())  # Ramping for alpha

    # Add both alpha and beta signals to the noise and repeat for each epoch
    composite_signal = signal_beta + signal_alpha
    data[:, 100:-100] += np.tile(composite_signal, n_epochs)  # Add to each epoch

    # Create RawArray and Epochs
    raw = mne.io.RawArray(data, info)
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(n_epochs) * n_times  # Event timing for epochs
    epochs = mne.Epochs(raw, events, event_id=dict(sin_signal=0), tmin=tmin, tmax=n_times / sfreq, baseline=None)

    # Plot averaged signal across epochs
    epochs.average().plot();

    epochs.load_data().crop(tmin=tmin, tmax=tmax)

    return epochs


def test_mirror_with_tfr_and_plot():

    # Create a signal with a beta power drop around stimulus onset
    epochs = create_signal_with_beta_drop(sfreq=100.0, tmin=-1.0, tmax=1.0)

    # Define the padding length
    pad_length = 50  # Add 50 time points of padding

    # crop the epochs in the pre-stimulus period
    epochs = epochs.crop(tmin=-1, tmax=0)

    ### Mirroring ###
    mirrored_epochs = zero_pad_or_mirror_epochs(epochs, zero_pad=False, pad_length=pad_length)
    
    times = epochs.times
    mirrored_times = mirrored_epochs.times

    # Plot original vs mirrored data
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot original data
    ax[0].plot(times, np.squeeze(epochs.average().data), label="Original", color='b')
    ax[0].set_title('Original Signal (Alpha + Beta with Increase)')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    
    # Plot mirrored data
    mirrored_times = mirrored_epochs.times
    ax[1].plot(mirrored_times, np.average(mirrored_epochs, axis=(0,1)), label="Mirrored", color='r')
    ax[1].set_title('Mirrored Signal')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Amplitude')
    
    # use original times tmin and tmax and plot a horizontal line to indicate the original data
    ax[1].axvline(x=min(times), color='k', linestyle='--',)
    ax[1].axvline(x=max(times), color='k', linestyle='--')

    plt.tight_layout()
    plt.legend()
    plt.show()

    # crop mirrored epochs to match the original epochs
    #mirrored_epochs = mirrored_epochs.crop(tmin=min(times), tmax=max(times))

    ### Apply TFR on Original and Mirrored Data ###
    freqs = np.arange(3, 36, 1)  # Frequencies from 3 to 35 Hz
    n_cycles = 2.  # As a rule of thumb, use n_cycles = freqs / 2
    power_original = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, 
    use_fft=True)
    power_mirrored = mne.time_frequency.tfr_multitaper(mirrored_epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, use_fft=True)

    # Plot TFR for Beta Band
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ### Normalization (Baseline Correction) ###
    baseline = (-0.8,-0.6) # Define baseline period (-500 ms to 0 ms)
    
    # Baseline correct both original and mirrored TFR using decibel (dB) change
    power_original.apply_baseline(baseline=baseline, mode='logratio')  # dB normalization
    power_mirrored.apply_baseline(baseline=baseline, mode='logratio')  # dB normalization

    # Original TFR (Beta Band)
    power_original.plot([0], axes=axes[0], show=False)
    axes[0].set_title('TFR - Original Data')

    # Mirrored TFR (Beta Band)
    power_mirrored.crop(min(times), max(times)).plot([0], axes=axes[1], show=False)
    axes[1].set_title('TFR - Zero Padding 4 cycles')

    plt.tight_layout()
    plt.show()

# Now run the test
test_mirror_with_tfr_and_plot()