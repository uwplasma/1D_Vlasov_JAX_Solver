from scipy.signal import find_peaks
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
import pandas as pd
import time
from scipy.stats import linregress
from diffrax import diffeqsolve, Dopri5, Tsit5, Dopri8, ODETerm, SaveAt, ConstantStepSize, PIDController, PIDController

jax.config.update("jax_enable_x64", True)

start_time = time.time()

m_max = 100
t_span = jnp.linspace(0, 200, 2001)
k_values = (jnp.linspace(0.01, 0.35, 100))

nu = 0
v_e = jnp.array([1,-1])
q = jnp.array([0,1])

def compute_dfm_dt(f, k, nu, v_e, Nn, index):
    dn = jnp.where(index < Nn, 0, 1)
    n = index-Nn*dn
    dfm_dt = (- 1j * k * jnp.sqrt(2) * (jnp.sqrt((n + 1) / 2) * f[index + 1] * jnp.sign(Nn - n - 1) +
                                        jnp.sqrt(n / 2) * f[index - 1] + v_e[dn] * f[index]) -
              (1j / k) * (q[1]**2*f[0]+f[Nn]*q[1]**2) * jnp.where(n == 1, 1,0))
    return dfm_dt

def objective(t, f, args):
    k, v_e = args
    dfm_dt = (jax.vmap(compute_dfm_dt, in_axes=(None, None, None, None, None, 0))(f, k, nu, v_e, (m_max+1)//2, jnp.arange(m_max)))
    return dfm_dt

# indices = jnp.arange(len(f))
    # dfm_dt = -1j * jnp.sqrt(2) * k * (jnp.roll(f, -1) * jnp.sqrt((indices + 1) / 2) +
    #                                   jnp.sqrt(indices / 2) * jnp.roll(f, 1) + v_e * f)
    # dfm_dt = dfm_dt.at[1].add(-(1j / k) * f[0])
    # dfm_dt = dfm_dt.at[0].set(-1j * jnp.sqrt(2) * k * (f[1] * jnp.sqrt(1 / 2) + v_e * f[0]))
    # dfm_dt = dfm_dt.at[-1].set(-1j * jnp.sqrt(2) * k * (jnp.sqrt(indices[-1] / 2) * f[-2] + v_e * f[-1]))
    # dfm_dt = dfm_dt.at[3:].add( - nu * (indices[3:] * (indices[3:] - 1) * (indices[3:] - 2)) / ((m_max - 1) * (m_max - 2) * (m_max - 3)) * f[3:])



def solver_ode(k, t_eval, m_max, v_e):
    y0 = jnp.zeros(m_max, dtype=jnp.complex128)
    y0 = y0.at[0].set(1)

    term = ODETerm(objective)
    solver = Dopri5()
    solution = diffeqsolve(
        term,
        solver,
        t0=t_eval[0],  # start time
        t1=t_eval[-1],  # end time
        dt0=0.01,  # Initial step size
        args=(k, v_e),  # value to objective
        y0=y0,
        saveat=SaveAt(ts=t_eval),  # save at Specify time
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
        max_steps = 1_000_000  # for end
    )

    return jnp.log(jnp.abs(solution.ys))

# def compute_slope(k):
#     solution = solver_ode(k, t_span, m_max, v_e)[:, 0]
#     peaks,_ = find_peaks(solution)
#     values = solution[peaks]
#     times = t_span[peaks]
#     return jax.lax.cond(times.shape[0] > 1,
#                     lambda _: linregress(times, values)[0],
#                     lambda _: jnp.nan,
#                     operand=None)


# slopes = jax.vmap(compute_slope)(k_values)

mathematica_data = (pd.read_csv('growth_rate.csv'))
# pd.read_csv('damping_rate.csv'))


slopes = []


for k in k_values:
    solution = solver_ode(k, t_span, m_max, v_e)[:, 0]

    # looking for local maxima
    # local_maxima_indices, _ = find_peaks(solution)
    # local_maxima_values = solution[local_maxima_indices]
    # local_maxima_times = t_span[local_maxima_indices]
    slope, intercept, r_value, p_value, std_err = linregress(t_span, solution)
    slopes.append(slope)
    # fit_line = slope * t_span + intercept
    # plt.figure()
    # plt.plot(t_span, solution, label='solution')
    # plt.plot(t_span, fit_line, 'r--', label=f'fit line: slope={slope:.3f}')
    # plt.xlabel('time t')
    # plt.ylabel('value')
    # plt.title('liner_fit_line')
    # plt.legend()
    # plt.show()

    # slopes
    # if len(local_maxima_times) > 1:  # fit for graph have 2+ mix local max
    #     slope, intercept, r_value, p_value, std_err = linregress(local_maxima_times, local_maxima_values)
    #     slopes.append(slope)
    #
    #     # plot fit for each C
    #     fit_line = slope * local_maxima_times + intercept
    #     plt.plot(local_maxima_times, fit_line, 'r--', label=f'Fit Line: slope={slope:.2f}')
    #     plt.plot(t_span, solution, label=f'K = {k:.2f}')
    #     plt.legend()
    #     plt.show()


# plot k value and slope

plt.figure()
plt.plot(k_values, jnp.array(slopes), '-o', label='Slope vs. k')
plt.plot(mathematica_data.values[:, 0], mathematica_data.values[:, 1], '-', label='Mathematica Slope vs. k')
# plt.plot(mathematica_data_2.values[:, 0], mathematica_data_2.values[:, 1], '-r', label='Mathematica Slope vs. k')
plt.xlim(0.0, 0.35)
plt.ylim(-0.05, 0.05)
plt.xlabel('k')
plt.ylabel('Slope')
plt.title('Slope vs. k Value(m_max: {})'.format(m_max))
plt.legend()
plt.show()

end_time = time.time()
print("runtime: {:.3f} sec".format(end_time - start_time))