import jax.numpy as jnp
from vlax1d.equations import objective, compute_dfm_dt

def test_compute_dfm_dt_basic():
    """
    Test that the compute_dfm_dt function works correctly for the given input and that its output is finite.
    """
    m_max = 10
    f = jnp.ones(m_max, dtype=jnp.complex128)
    k = 0.5
    nu = 0.0
    v_e = jnp.array([1.0, 1.0])
    q = jnp.array([1.0, 1.0])
    index = 3
    Nn = m_max // 2

    val = compute_dfm_dt(f, k, nu, v_e, Nn, index, q)
    assert jnp.isfinite(val), "Output should be a finite number"

def test_objective_shape():
    """
    Test whether the shape of the output of the objective function is consistent with the input state f.
    """
    m_max = 10
    f = jnp.ones(m_max, dtype=jnp.complex128)
    k = 0.3
    nu = 0.1
    v_e = jnp.array([1.0, 1.0])
    q = jnp.array([1.0, 1.0])
    t = 0.0

    result = objective(t, f, (k, v_e, nu, m_max, q))
    assert result.shape == (m_max,)
    assert jnp.all(jnp.isfinite(result)), "Objective contains non-finite values"
