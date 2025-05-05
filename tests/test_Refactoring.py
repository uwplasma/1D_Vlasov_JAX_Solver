# ../OneD_Vlasov_Poisson_JAX_Solver/tests/test_Refactoring.py
import jax.numpy as jnp
import pytest
from scipy.special import eval_hermite, factorial
from vlax1d.Refactoring import reconstruct_fv


def test_reconstruct_fv_output_shape():
    """
    Test if the output of reconstruct_fv has the correct shape based on inputs.
    """
    f_m_t = jnp.ones((10, 5))  # 10 time steps, 5 Hermite coefficients
    v_grid = jnp.linspace(-5, 5, 100)  # velocity grid with 100 points
    result = reconstruct_fv(f_m_t, v_grid)
    assert result.shape == (f_m_t.shape[0], len(v_grid)), "Output shape mismatch"


def test_reconstruct_fv_no_nan_inf():
    """
    Test if the output of reconstruct_fv contains no NaN or Inf values.
    """
    f_m_t = jnp.ones((10, 5))  # 10 time steps, 5 Hermite coefficients
    v_grid = jnp.linspace(-3, 3, 50)  # velocity grid with 50 points
    result = reconstruct_fv(f_m_t, v_grid)
    assert jnp.all(jnp.isfinite(result)), "Output contains NaN or Inf values"


def test_reconstruct_fv_hermite_polynomials():
    """
    Test if the Hermite polynomials are appropriately evaluated in reconstruct_fv.
    """
    f_m_t = jnp.zeros((1, 3))  # 3 Hermite coefficients (0, 1, and 2 only)
    f_m_t = f_m_t.at[0, 1].set(1)  # CoEfficient for 1st Hermite polynomial
    v_grid = jnp.array([-1, 0, 1])  # Small grid for clarity
    result = reconstruct_fv(f_m_t, v_grid)
    expected = eval_hermite(1, v_grid) * jnp.exp(-v_grid ** 2 / 2) / (
        jnp.sqrt(2 * factorial(1) * jnp.sqrt(jnp.pi))
    )
    assert jnp.allclose(result[0], expected), "Hermite polynomial evaluation mismatch"
