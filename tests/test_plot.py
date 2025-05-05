import jax.numpy as jnp
import pandas as pd
from vlax1d.plot import plot_results

def test_plot_results_runs():
    """
    Tests the functionality of the `plot_results` function by providing test inputs.
    The function executes the `plot_results` function with predefined values for
    `k_values`, `slopes`, and `fake_data` to verify its performance.

    :param k_values: An array of evenly spaced values between 0.1 and 0.5,
        created using jnp.linspace.
    :param slopes: An array of evenly spaced slopes between -0.05 and 0.05,
        created using jnp.linspace.
    :param fake_data: A DataFrame containing simulated data with the first column
        containing values from 0.1 to 0.5 and the second column containing values
        from -0.05 to 0.05.
    :param m_max: An optional parameter for the `plot_results` function
        representing the maximum value of 'm'. The test provides the value 20.
    :type fake_data: pd.DataFrame
    :type m_max: int

    :return: None
    """
    k_values = jnp.linspace(0.1, 0.5, 5)
    slopes = jnp.linspace(-0.05, 0.05, 5)
    fake_data = pd.DataFrame({
        0: jnp.linspace(0.1, 0.5, 5),
        1: jnp.linspace(-0.05, 0.05, 5)
    })
    plot_results(k_values, slopes, fake_data, m_max=20)