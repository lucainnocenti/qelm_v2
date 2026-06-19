# QELM rank diagnostics

This folder contains reusable helpers for the low-rank probability-matrix
diagnostics that were previously defined directly in `tests.ipynb`.

## Structure

- `qelm/linalg.py`: basic matrix norms, inverses, validation, and log-log fits.
- `qelm/blocks.py`: deterministic SVD block decomposition for a probability matrix `P`.
- `qelm/noise.py`: scaled noise matrices `Xi` and noisy design matrices `P_hat`.
- `qelm/trials.py`: fixed-`P` block diagnostics using scaled shot-noise `Xi`.
- `qelm/plotting.py`: collapse plots, failure-rate plots, and sweep diagnostics.
- `qelm/quantum.py`: construction of probability matrices from POVM effects and states.
- `qelm/training.py`: structured QELM specs, contexts, leading-error formulas, and actual noisy least-squares training.
- `qelm/workflows.py`: notebook workflows that compose `trials.py` and `training.py`.
- `qelm/markov.py`: Markov-slack helper columns.

## Typical notebook usage

```python
from qelm import run_toy_low_rank_sweep, plot_metric_vs_predictors

trial_df, summary = run_toy_low_rank_sweep(
    q_values=(4, 8),
    p_values=(500, 1000, 2500, 5000),
    trials=30,
    noise_model="gaussian",
)

display(summary)
plot_metric_vs_predictors(summary, quantile="p90")
```

To inspect deterministic scaling for random quantum probability matrices:

```python
from qelm import (
    fit_random_quantum_scaling_laws,
    plot_random_quantum_scaling,
    run_random_quantum_scaling_sweep,
)

scaling_df = run_random_quantum_scaling_sweep(
    d_values=[2, 3],
    nout_values=[8, 12, 16, 24],
    ntr_values=[32, 64, 128],
    repetitions=3,
    progress=False,
)

display(scaling_df[["d", "nout", "ntr", "lambda_min_C22", "lambda_max_C22", "delta_shape", "c_p"]])
display(fit_random_quantum_scaling_laws(scaling_df))
plot_random_quantum_scaling(scaling_df)
```

For a real probability matrix, use:

```python
from qelm import (
    generate_haar_random_pure_dms,
    generate_random_rank1_povm,
    probability_matrix_from_povm_states,
    run_single_P_workflow,
)

effects = generate_random_rank1_povm(nout=8, dim=d, rng=rng)
states = generate_haar_random_pure_dms(num_states=ntr, dim=d, rng=rng)

P = probability_matrix_from_povm_states(
    povm_effects=effects,  # shape (nout, d, d)
    states=states,         # shape (ntr, d, d)
)

blocks, trial_df, summary = run_single_P_workflow(P, r=d**2)
```
