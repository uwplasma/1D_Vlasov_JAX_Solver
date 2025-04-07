import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController


def solver_ode(k, t_span, m_max, v_e, nu, q, objective):
    """
    Solving ODEs for Vlasov equations.

    :param k: Wave Number parameters
    :param t_span: Time step array
    :param m_max: Maximum number of all modes
    :param v_e: Speed Parameters
    :param objective: ODE Computes the Objective Function
    :return: Calculation results (after log processing)
    """
    y0 = jnp.zeros(m_max, dtype=jnp.complex128)
    y0 = y0.at[0].set(1)

    term = ODETerm(objective)
    solver = Dopri5()
    solution = diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[-1],
        dt0=0.01,
        args=(k, v_e, nu, m_max, q),
        y0=y0,
        saveat=SaveAt(ts=t_span),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
        max_steps=1_000_000
    )

    return solution.ys
