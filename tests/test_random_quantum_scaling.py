import numpy as np
import pytest

from qelm_rank import (
    POVM,
    QuantumStateBatch,
    fit_random_quantum_scaling_laws,
    run_random_quantum_scaling_sweep,
    validate_probability_matrix,
)


def test_random_rank1_povm_and_state_batch_probability_matrix_shape_and_columns():
    rng = np.random.default_rng(123)

    povm = POVM.random_rank1(nout=6, dim=2, rng=rng)
    states = QuantumStateBatch.haar_pure(num_states=9, dim=2, rng=rng)
    P = povm.probability_matrix(states)

    assert P.shape == (6, 9)
    validate_probability_matrix(P, atol=1e-10)


def test_run_random_quantum_scaling_sweep_records_block_quantities():
    df = run_random_quantum_scaling_sweep(
        d_values=[2],
        nout_values=[5, 6],
        ntr_values=[8, 10],
        repetitions=2,
        seed=123,
        progress=False,
    )

    assert len(df) == 8

    expected_columns = {
        "d",
        "d2",
        "nout",
        "ntr",
        "r",
        "q",
        "p",
        "lambda_min_C22",
        "lambda_max_C22",
        "delta_shape",
        "c_p",
        "nout_over_d2",
        "ntr_over_d2",
        "ntr_over_nout",
        "rank_estimate",
        "sigma_r",
    }
    assert expected_columns <= set(df.columns)

    assert np.all(df["d"] == 2)
    assert np.all(df["d2"] == 4)
    assert np.all(df["nout"] > df["d2"])
    assert np.all(df["ntr"] > df["d2"])
    assert np.all(df["ntr"] > df["nout"])
    assert np.all(df["lambda_min_C22"] > 0)
    assert np.all(df["lambda_max_C22"] >= df["lambda_min_C22"])
    assert np.all(df["delta_shape"] > 0)
    assert np.all(df["c_p"] >= 0)
    assert np.all(df["sigma_r"] > 0)


def test_fit_random_quantum_scaling_laws_returns_exponents():
    df = run_random_quantum_scaling_sweep(
        d_values=[2],
        nout_values=[5, 6],
        ntr_values=[8, 10],
        repetitions=2,
        seed=123,
        progress=False,
    )

    fit = fit_random_quantum_scaling_laws(df)

    assert set(fit["quantity"]) == {
        "lambda_min_C22",
        "lambda_max_C22",
        "delta_shape",
        "c_p",
    }
    assert np.all(fit["num_rows"] == len(df))
    assert np.all(np.isfinite(fit["nout_power"]))
    assert np.all(np.isfinite(fit["ntr_power"]))


def test_random_quantum_scaling_sweep_rejects_nout_le_d_squared():
    with pytest.raises(ValueError, match="nout > d\\^2"):
        run_random_quantum_scaling_sweep(
            d_values=[2],
            nout_values=[4],
            ntr_values=[8],
            progress=False,
        )
