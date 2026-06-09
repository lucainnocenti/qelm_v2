# QELM rank diagnostics

This folder contains reusable helpers for the low-rank probability-matrix
diagnostics that were previously defined directly in `tests.ipynb`.

## Structure

- `qelm_rank/linalg.py`: basic matrix norms, inverses, validation, and log-log fits.
- `qelm_rank/blocks.py`: deterministic SVD block decomposition for a probability matrix `P`.
- `qelm_rank/noise.py`: multinomial and Gaussian shot-noise generation.
- `qelm_rank/trials.py`: one-trial diagnostics, repeated trials, and summary tables.
- `qelm_rank/plotting.py`: collapse plots, failure-rate plots, and sweep diagnostics.
- `qelm_rank/quantum.py`: construction of probability matrices from POVM effects and states.
- `qelm_rank/workflows.py`: high-level workflows for one `P`, toy matrices, and dimension sweeps.
- `qelm_rank/markov.py`: Markov-slack helper columns.

## Typical notebook usage

```python
from qelm_rank import run_toy_low_rank_sweep, plot_metric_vs_predictors

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
from qelm_rank import (
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
from qelm_rank import (
    generate_haar_random_pure_states,
    generate_random_rank1_povm,
    probability_matrix_from_povm_states,
    run_single_P_workflow,
)

effects = generate_random_rank1_povm(nout=8, dim=d, rng=rng)
states = generate_haar_random_pure_states(num_states=ntr, dim=d, rng=rng)

P = probability_matrix_from_povm_states(
    povm_effects=effects,  # shape (nout, d, d)
    states=states,         # shape (ntr, d, d)
)

blocks, trial_df, summary = run_single_P_workflow(P, r=d**2)
```
