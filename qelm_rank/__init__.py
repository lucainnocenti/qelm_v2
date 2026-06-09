"""Diagnostics for low-rank probability-matrix shot-noise experiments."""

from .blocks import PBlocks, block_report, deterministic_blocks_from_P
from .linalg import (
    empirical_quantiles,
    frobnorm,
    loglog_slope,
    opnorm,
    safe_inv,
    validate_probability_matrix,
)
from .markov import add_markov_slack_columns
from .noise import generate_gaussian_Xi, generate_multinomial_Xi, project_noise_blocks
from .plotting import (
    compare_q3_q5_collapse,
    deterministic_cp_scaling_fit,
    plot_Xi21_markov_check,
    plot_deterministic_block_scalings,
    plot_failure_rates,
    plot_metric_vs_kappa,
    plot_metric_vs_predictors,
    plot_random_quantum_scaling,
    plot_sweep_diagnostics,
    scatter_loglog,
)
from .quantum import (
    generate_haar_random_isometry,
    generate_haar_random_pure_states,
    generate_haar_random_state_vectors,
    generate_random_rank1_povm,
    probability_matrix_from_povm_states,
)
from .trials import one_trial_diagnostics, run_trials, summarize_trials, theoretical_predictors
from .workflows import (
    fit_random_quantum_scaling_laws,
    make_random_quantum_probability_matrix,
    make_toy_P_for_sweep,
    make_toy_low_rank_probability_matrix,
    run_dimension_sweep,
    run_random_quantum_scaling_sweep,
    run_single_P_workflow,
    run_toy_low_rank_sweep,
)

__all__ = [
    "PBlocks",
    "add_markov_slack_columns",
    "block_report",
    "compare_q3_q5_collapse",
    "deterministic_blocks_from_P",
    "deterministic_cp_scaling_fit",
    "empirical_quantiles",
    "frobnorm",
    "generate_gaussian_Xi",
    "generate_haar_random_isometry",
    "generate_haar_random_pure_states",
    "generate_haar_random_state_vectors",
    "generate_multinomial_Xi",
    "generate_random_rank1_povm",
    "loglog_slope",
    "fit_random_quantum_scaling_laws",
    "make_random_quantum_probability_matrix",
    "make_toy_P_for_sweep",
    "make_toy_low_rank_probability_matrix",
    "one_trial_diagnostics",
    "opnorm",
    "plot_Xi21_markov_check",
    "plot_deterministic_block_scalings",
    "plot_failure_rates",
    "plot_metric_vs_kappa",
    "plot_metric_vs_predictors",
    "plot_random_quantum_scaling",
    "plot_sweep_diagnostics",
    "probability_matrix_from_povm_states",
    "project_noise_blocks",
    "run_dimension_sweep",
    "run_random_quantum_scaling_sweep",
    "run_single_P_workflow",
    "run_toy_low_rank_sweep",
    "run_trials",
    "safe_inv",
    "scatter_loglog",
    "summarize_trials",
    "theoretical_predictors",
    "validate_probability_matrix",
]
