import jax.numpy as jnp
from vlax1d.solver import solver_ode
from vlax1d.equations import objective


def test_solver_ode_output():
    """
    Test the output of solver_ode: Is the result of the correct shape and without NaN/Inf returned
    """
    m_max = 40
    t_span = jnp.linspace(0, 200, 2001)
    k = 0.3
    nu = 2
    v_e = jnp.array([0, 0])
    q = jnp.array([1, 0])

    sol = solver_ode(k, t_span, m_max, v_e, nu, q, objective)
    assert sol.shape == (len(t_span), m_max), "Output shape mismatch"
    assert jnp.all(jnp.isfinite(sol)), "Solution contains NaN or Inf values"
