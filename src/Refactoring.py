from scipy.special import eval_hermite
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
from src.equations import objective
from src.solver import solver_ode
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
    Reconstruct f(v, t) from Hermite coefficients.
    :param f_m_t: Hermite coefficients, shape (T, M)
    :param v_grid: 1D array of v values
    :return: f(v, t) array, shape (T, len(v_grid))
    """
    T, M = f_m_t.shape
    V = len(v_grid)
    f_vt = jnp.zeros((T, V))

    # Precompute Hermite functions φ_m(v) for all m, v
    phi_mv = jnp.zeros((M, V))
    for m in range(M):
        Hm_v = eval_hermite(m, v_grid)
        norm = jnp.sqrt(2**m * factorial(m, exact=False) * jnp.sqrt(jnp.pi))
        phi_mv = phi_mv.at[m, :].set(Hm_v * jnp.exp(-v_grid**2 / 2) / norm)

    # Sum over m to reconstruct f(v, t)
    for t in range(T):
        f_vt = f_vt.at[t, :].set(jnp.dot(f_m_t[t, :], phi_mv))

    return f_vt

# Define velocity grid
v_grid = jnp.linspace(-6, 6, 300)

# Reconstruct particle distribution
f_vt = reconstruct_fv(sol, v_grid)  # shape (T, V)

plt.plot(v_grid, f_vt[0])
plt.xlabel("Velocity v")
plt.ylabel("f(v)")
plt.title("Particle distribution at t = {:.2f}".format(t_span[0]))
plt.show()
