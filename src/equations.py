import jax.numpy as jnp
from jax import debug
import jax


@jax.jit
def compute_dfm_dt(f, k, nu, v_e, Nn, index, q):
    """
    Computes the time derivative of a function representing a system described in the
    Fourier and physical space, taking into account effects like wave interactions,
    damping, and external forces.

    :param f: jnp.ndarray
        The wave function represented as an array in the Fourier or physical space.
    :param k: float
        The wave number that determines the spatial frequency of the wave.
    :param nu: float
        The viscosity coefficient impacting energy dissipation.
    :param v_e: jnp.ndarray
        An array specifying velocity factors for the external forces.
    :param Nn: int
        The maximum number of wave modes in the system.
    :param index: int
        The current index in the array representing the mode being evaluated.
    :param q: jnp.ndarray
        An array of coefficients that modify the coupling between modes.

    :return: jnp.complex128
        The computed time derivative of the specified wave mode, taking into
        account interactions between different modes and external forces.
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
    Computes the time derivative of the Fourier mode dynamics based on the provided input
    parameters and a set of arguments. This function utilizes `jax.vmap` for vectorized map
    computation, applying the `compute_dfm_dt` function over a specific range of modes.

    The computation involves iterating over a sequence of values determined by the provided
    maximum number of modes (`m_max`) and processing elements using the arguments supplied
    in `args`. JAX's `vmap` is employed to enable efficient parallel application of the
    calculation across multiple data points or modes.

    The function encapsulates a physics-oriented calculation typically used in time-evolution
    simulations, spectrum modeling, or Fourier transform-based computational frameworks.

    :param t: Current time, treated as a placeholder in this function.
    :type t: float
    :param f: Array of Fourier modes at the current timestep.
               Represents the state or amplitude values of the modes.
    :type f: jax.numpy.ndarray
    :param args: Tuple of additional parameters passed to define the dynamics:
                 - k: Wavenumber or characteristic parameter affecting dynamics.
                 - v_e: Constant or coefficient influencing velocity evolution.
                 - nu: Diffusion or damping coefficient used in the computation.
                 - m_max: Maximum number of Fourier modes utilized in the summation.
                 - q: External or forcing parameter influencing mode amplitudes.
    :type args: tuple
    :return: Array containing the time derivatives of the Fourier modes computed over
             the range of indices specified by `m_max`.
    :rtype: jax.numpy.ndarray
    """
    k, v_e, nu, m_max, q = args
    dfm_dt = jax.vmap(compute_dfm_dt, in_axes=(None, None, None, None, None, 0, None))(
        f, k, nu, v_e, (m_max + 1) // 2, jnp.arange(m_max), q
    )
    return dfm_dt
