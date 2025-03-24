# One D Vlasov-Poisson Equation JAX Solver

This repository provides a numerical solver for the one-dimensional Vlasov-Poisson equation using JAX for high-performance computing. The solver leverages GPU acceleration and JAX's automatic differentiation to efficiently simulate plasma wave dynamics, including Landau damping and instability phenomena.

## Project Overview
(add some explanation of calculation)
This solver numerically integrates the 1D linear Vlasov-Poisson equation to analyze wave propagation, damping rates, and stability in plasma physics. The solver utilizes JAX to achieve high-performance computations, supporting automatic differentiation and GPU/TPU acceleration.

## Features
- **High-performance JAX-based solver:** Leverages JAX's automatic differentiation and just-in-time (JIT) compilation.
- **Robust analysis tools:** Peak detection, linear fitting, and data export to CSV.
- **Customizable parameters:** Easy-to-adjust parameters such as wave number (`k`), collision frequency (`nu`), and the mode number for all species (`m_max`).

## Installation
```bash
pip install Oned-Vlasov-Poisson-jax-solver
```

## Usage
To run simulations and analyze damping rates:

```bash
python two_stream_instability.py
```

Adjust parameters within the script:
```python
# Set the maximum number of all modes (the dimension of the state variable)
# Which should be an even number because the subsequent code will divide it into two groups
m_max = 40

# Define the time range of the simulation
# From t = 0 to t = 200, a total of 2001 time points (i.e. Δt = 0.1)
t_span = jnp.linspace(0, 200, 2001)

# Set of wave numbers (k values) to loop over or analyze.
# These represent different spatial scales of perturbations in the system.
k_values = jnp.linspace(0.01, 0.40, 40)

# Collision frequency (nu), representing the strength of the damping term.
# Higher values introduce stronger diffusion (e.g., Landau damping, viscosity).
nu = 2

# Average velocities for two interacting populations (e.g., beam and background).
# Used to model counter-propagating or asymmetric systems.
v_e = jnp.array([1, -1])

# Charge weights or coefficients associated with the populations.
# For population one and two.
q = jnp.array([0, 1])
```

## Mathematical Background

We start with the 1D Vlasov equation:

∂f/∂t + v ∂f/∂x + (qE/m) ∂f/∂v = 0

After linearizing and Fourier-transforming:

∂(δf)/∂t + i·k·v·(δf) + (q·δE/m) ∂fₘ/∂v = 0

We expand δf(v, t) in Hermite polynomials:

δf(v, t) = Σₙ fₙ(t) Hₙ((v - vₑ)/α)

Then, projecting onto each Hermite mode:

∂fₙ/∂t = -i√[(n+1)/2] fₙ₊₁ - i√[n/2] fₙ₋₁ - i·vₑ·fₙ - C·f₀·δₙ₁

where C ∝ 1/k from Poisson's equation.

for more detail please check plasma_equation in docs 

---

## Project Structure
```
OneD_Vlasov_Poisson_JAX_Solver/
├── data/                           # Benchmark datasets
│   ├── damping_rate.csv            # Reference damping rates
│   └── growth_rate.csv             # Reference growth rates
│
├── docs/                           # Project documentation
│   ├── plasma_equation.pdf         # Full derivation of the model equations
│   └── README.md                   # Main project overview (this file)
│
├── examples/                       # Run-ready example scripts
│   ├── Landau_damping.py           # Landau damping simulation
│   └── two_stream_instability.py   # Two-stream instability simulation
│
├── src/                            # Core source code
│   ├── __init__.py                 # Makes src a Python package
│   ├── equations.py                # Hermite-mode RHS and equation definitions
│   ├── plot.py                     # Plotting data
│   └── solver.py                   # Time integrator using Diffrax
│
├── MANIFEST.in                     # Include non-code files in builds (e.g. .csv)
├── pyproject.toml                  # Modern build configuration file
├── setup.py                        # Traditional setup for pip install
└── README.md                       # Project documentation (top-level entry)
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
