# plot an oscillation

from neurodsp.sim import (sim_oscillation, sim_bursty_oscillation,
                          sim_variable_oscillation, sim_damped_oscillation)

from neurodsp.utils import set_random_seed
from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series

fs=1000

# Simulation settings
n_seconds = 0.25
times = create_times(n_seconds, fs)--

# Define oscillation frequency
osc_freq = 20

# Simulate a sinusoidal oscillation
osc_sine = sim_oscillation(n_seconds, fs, osc_freq, cycle='sine')

# Plot the simulated data, in the time domain
plot_time_series(times, osc_sine)

import neurokit2 as nk
import pandas as pd

ax = pd.DataFrame({
    "20Hz": nk.signal_simulate(duration=0.25, frequency=20, 
                               amplitude=1, noise=0.1),
    "20Hz_high": nk.signal_simulate(duration=0.25, frequency=20, 
                               amplitude=3, noise=0.1),
}).plot()
