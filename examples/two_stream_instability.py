import jax.numpy as jnp
import jax
import time

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from vlax1d.equations import objective
from vlax1d.solver import solver_ode
from vlax1d.plot import plot_results

jax.config.update("jax_enable_x64", True)

start_time = time.time()

m_max = 40
t_span = jnp.linspace(0, 200, 2001)
k_values = jnp.linspace(0.01, 0.40, 40)
nu = 2
v_e = jnp.array([1,-1])
q = jnp.array([1,1])

mathematica_data = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'growth_rate.csv')

slopes = []

for k in k_values:
    solution = jnp.log(jnp.abs(solver_ode(k, t_span, m_max, v_e, nu, q, objective)[:, 0]))
    slope, intercept, r_value, p_value, std_err = linregress(t_span, solution)
    slopes.append(slope)

plot_results(k_values, slopes, mathematica_data, m_max)
