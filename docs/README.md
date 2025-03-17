# 1D Vlasov-Poisson Equation JAX Solver

This repository provides a numerical solver for the one-dimensional Vlasov-Poisson equation using JAX for high-performance computing. The solver leverages GPU acceleration and JAX's automatic differentiation to efficiently simulate plasma wave dynamics, including Landau damping and instability phenomena.

## Project Overview
This solver numerically integrates the 1D linear Vlasov-Poisson equation to analyze wave propagation, damping rates, and stability in plasma physics. The solver utilizes JAX to achieve high-performance computations, supporting automatic differentiation and GPU/TPU acceleration.

## Features
- **High-performance JAX-based solver:** Leverages JAX's automatic differentiation and just-in-time (JIT) compilation.
- **Robust analysis tools:** Peak detection, linear fitting, and data export to CSV.
- **Customizable parameters:** Easy-to-adjust parameters such as wave number (`k`), collision frequency (`nu`), and system size (`m_max`).

## Installation
```bash
pip install -e
```

## Usage
To run simulations and analyze damping rates:

```bash
python run_simulation.py
```

Adjust parameters within the script:
```python
m_max = 40
t_span = jnp.linspace(0, 200, 2001)
k_values = jnp.linspace(0.01, 0.40, 40)
nu = 2
v_e = jnp.array([1,-1])
q = jnp.array([0,1])
```

## Project Structure
```
1D_Vlasov_Poisson_JAX_Solver/
├── src/
│   ├── equations.py       # Core Vlasov_Poisson solver equations
│   ├── solver.py          # Solver integration and ODE definitions
│   └── plot.py            # Visualization and data plotting
├── examples/
│   ├── run_simulation.py  # Script to execute simulations
│   └── analyze.py         # Analyze and visualize results
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
