import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController


def solver_ode(k, t_span, m_max, v_e, nu, q, objective):
    """
    Solves a system of ordinary differential equations (ODEs) using the specified
    objective function and problem parameters. The solver integrates the system from
    a given initial state over a specified time span. It employs a Dopri5 solver with
    a PID-based step size controller to achieve precise numerical results.

    :param k: Parameter to be passed as an argument to the ODE system.
    :type k: Array-like
    :param t_span: Time span over which the ODEs are solved. It must be a sequence
        containing the initial and final times of integration.
    :type t_span: Sequence[float]
    :param m_max: The maximum length used to initialize the state vector for the ODE
        problem.
    :type m_max: int
    :param v_e: Additional argument to be passed into the objective function.
    :type v_e: float
    :param nu: Additional argument to be passed into the objective function.
    :type nu: float
    :param q: Additional argument to be passed into the objective function.
    :type q: Array-like
    :param objective: The objective function defining the ODE system. It represents
        the time derivative of the state vector.
    :type objective: Callable
    :return: The solution of the ODE system, which includes the time-evaluated state
        vector at specified save points.
    :rtype: Array-like
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
