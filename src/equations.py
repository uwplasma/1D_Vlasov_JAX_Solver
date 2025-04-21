import jax.numpy as jnp
from jax import debug
import jax


@jax.jit
def compute_dfm_dt(f, k, nu, v_e, Nn, index, q):
    """
    Calculate the right-hand side of dfm/dt.

    :param f: Current state
    :param k: Wave number parameter
    :param nu: Collision parameter
    :param v_e: Velocity parameter
    :param Nn: The mode number for one species
    :param index: Current index
    :return: ODE relation
    """
    dn = jnp.where(index < Nn, 0, 1)
    n = index - Nn * dn
    dfm_dt = ((- 1j * k * jnp.sqrt(2) * (jnp.sqrt((n + 1) / 2) * f[index + 1] * jnp.sign(Nn - n - 1) +
                                         jnp.sqrt(n / 2) * f[index - 1] + v_e[dn] * f[index]) -
               (1j / k) * (q[0] ** 2 * f[0] + f[Nn] * q[1] ** 2) * jnp.where(n == 1, 1, 0))
              - nu * (n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) * f[index])
    return dfm_dt


def objective(t, f, args):
    """
    Target ODE calculation function.

    :param t: time
    :param f: state variable
    :param args: additional parameters
    :return: ODE ready for solver
    """
    k, v_e, nu, m_max, q = args
    dfm_dt = jax.vmap(compute_dfm_dt, in_axes=(None, None, None, None, None, 0, None))(
        f, k, nu, v_e, (m_max + 1) // 2, jnp.arange(m_max), q
    )
    return dfm_dt
