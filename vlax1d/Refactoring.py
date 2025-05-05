from scipy.special import eval_hermite
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
from vlax1d.equations import objective
from vlax1d.solver import solver_ode
from scipy.special import eval_hermite, factorial


jax.config.update("jax_enable_x64", True)

start_time = time.time()

m_max = 200
t_span = jnp.linspace(0, 200, 2001)
# k_values = jnp.linspace(0.01, 0.40, 40)
nu = 2
v_e = jnp.array([0,0])
q = jnp.array([1,0])

sol = jnp.abs(solver_ode(0.03, t_span, m_max, v_e, nu, q, objective))

def reconstruct_fv(f_m_t, v_grid):
    """
    Reconstructs the time-evolving function f(v, t) on a velocity grid, given its
    decomposition in terms of Hermite functions. The reconstruction is performed
    by computing the Hermite functions for all modes and velocities, normalizing
    them, and summing over the modes to obtain the desired function.

    :param f_m_t: Coefficients f_m(t) of the Hermite decomposition at each time step.
                  The shape is (T, M), where T is the number of time steps, and M
                  is the number of modes.
    :type f_m_t: ndarray or jnp.ndarray
    :param v_grid: The velocity grid over which the function is reconstructed.
                   The length of v_grid corresponds to the number of velocity points.
    :type v_grid: ndarray or jnp.ndarray
    :return: The reconstructed function f(v, t) on the velocity grid. The shape of
             the output is (T, V), where V is the number of velocity points.
    :rtype: jnp.ndarray
    """
    T, M = f_m_t.shape
    V = len(v_grid)
    phi_mv = jnp.zeros((M, V))

    # Precompute Hermite functions φ_m(v) for all m, v
    for m in range(M):
        Hm_v = eval_hermite(m, v_grid)
        norm = jnp.sqrt(2 ** m * factorial(m, exact=False) * jnp.sqrt(jnp.pi))
        phi_mv = phi_mv.at[m].set(Hm_v * jnp.exp(-v_grid ** 2 / 2) / norm)

    # Sum over m to reconstruct f(v, t)
    f_vt = jnp.dot(f_m_t, phi_mv)

    return f_vt

# Define velocity grid
v_grid = jnp.linspace(-6, 6, 300)

# Reconstruct particle distribution
f_vt = reconstruct_fv(sol, v_grid)  # shape (T, V)

plt.plot(v_grid, f_vt[10])
plt.xlabel("Velocity v")
plt.ylabel("f(v)")
plt.title("Particle distribution at t = {:.2f}".format(t_span[0]))
plt.show()
