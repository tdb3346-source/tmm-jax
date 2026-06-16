# TMM-JAX: Thin Film Inverse Modeling

## What this is
Given a reflectance spectrum, gradient descent can recover the film 
thickness — but the answer depends on the starting guess, and wrong 
starting points converge to false solutions.

## What's in the notebook
- Forward model: given thickness d, compute R(λ) across 400-700nm
- Inversion: given R(λ), recover d using jax.grad
- Identifiability experiment: same target, 5 different starting guesses

## Key result
| Starting guess | Final d | Converged? |
|---|---|---|
| 50nm | 67nm | No |
| 100nm | 150nm | Yes |
| 200nm | 150nm | Yes |
| 300nm | 365nm | No |
| 400nm | 365nm | No |

## How to run
1. conda activate tmmjax
2. jupyter lab
3. Open tmm_forward.ipynb
4. Kernel → Restart Kernel and Run All