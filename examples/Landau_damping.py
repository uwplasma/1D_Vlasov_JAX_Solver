import jax.numpy as jnp
import jax
import time
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import linregress
from src.solver import solver_ode
from src.equations import objective
from src.plot import plot_results

jax.config.update("jax_enable_x64", True)

start_time = time.time()

m_max = 40
t_span = jnp.linspace(0, 200, 2001)
k_values = jnp.linspace(0.3, 0.3, 1)
nu = 2
v_e = jnp.array([0,0])
q = jnp.array([1,0])

mathematica_data = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'damping_rate.csv')

slopes = []

for k in k_values:
    solution = jnp.log(jnp.abs(solver_ode(k, t_span, m_max, v_e, nu, q, objective)[:, 0]))

    # looking for local maxima
    local_maxima_indices, _ = find_peaks(solution)
    local_maxima_values = solution[local_maxima_indices]
    local_maxima_times = t_span[local_maxima_indices]

    # slopes
    if len(local_maxima_times) > 1:  # fit for graph have 2+ mix local max
        slope, intercept, r_value, p_value, std_err = linregress(local_maxima_times, local_maxima_values)
        slopes.append(slope)

plot_results(k_values, slopes, mathematica_data, m_max)