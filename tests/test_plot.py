import jax.numpy as jnp
import pandas as pd
from src.plot import plot_results

def test_plot_results_runs():
    """
    Test that plot_results works properly (does not throw errors) with simulated data.
    """
    k_values = jnp.linspace(0.1, 0.5, 5)
    slopes = jnp.linspace(-0.05, 0.05, 5)
    fake_data = pd.DataFrame({
        0: jnp.linspace(0.1, 0.5, 5),
        1: jnp.linspace(-0.05, 0.05, 5)
    })
    plot_results(k_values, slopes, fake_data, m_max=20)