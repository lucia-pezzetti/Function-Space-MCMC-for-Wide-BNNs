# Function-Space MCMC for Wide Bayesian Neural Networks
The code has been re-adapted by [github.com/google/wide_bnn_sampling]/(https://github.com/google/wide_bnn_sampling).

The contributions to the code regards the addition of the preconditioned Crank-Nicholson (pCN) and preconditioned Crank-Nicholson Langevin (pCNL) samplers, see `samplers.py` for the implementation.

## Setup
The dependencies are in the file `setup.py` and can be installed running
```python
git clone https://github.com/google/wide_bnn_sampling
cd wide_bnn_sampling
pip install -e .
```
Note that `jaxlib` is also required and needs a specific installation based on the hardware. Refer to: [JAX's repository](https://github.com/google/jax#installation).

## Contents
- `config.py`: contains the parameters and the specifications for the experiments
- `datasets.py`: manage the data (loading and preprocessing)
- `main.py`: core of the experimental procedure.
- `measurements.py`:
- `models.py`: define the neural networks architectures (FCN and ResNet).
- `reparametrisation.py`: implement the reparametrisation of the posterior of the weights (see [Hron et al.](https://arxiv.org/abs/2206.07673) for more details)
- `samplers.py`: contains the implementations of the used MCMC procedures, specifically the Hamiltonia Monte Carlo/Langevin Monte Carlo, pCN, pCNL, Metropolis-Hastings with a simple Gaussian proposal.
-  `utils.py`: auxiliary methods.
