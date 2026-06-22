import numpy as np
import pytest

from qelm import (
    POVM,
    QELMQuantumDataset,
    QuantumStateBatch,
    clear_default_rng,
    generate_haar_random_isometry,
    generate_haar_random_kets,
    generate_haar_random_pure_dms,
    generate_qubit_mub_povm,
    generate_random_rank1_povm,
    haar_probability_moments_from_isometry,
    probability_matrix_from_povm_states,
    probability_vector_from_povm_state,
    ket2dm,
    set_default_rng,
    validate_probability_matrix,
)


def ket(vector):
    vector = np.asarray(vector, dtype=complex)
    return vector / np.linalg.norm(vector)


def density(vector):
    vector = ket(vector)
    return np.outer(vector, vector.conj())


def test_projective_measurement_probability_matrix():
    zero = density([1, 0])
    one = density([0, 1])
    plus = density([1, 1])
    mixed = np.eye(2) / 2

    povm = np.stack([zero, one])
    states = np.stack([zero, one, plus, mixed])

    P = probability_matrix_from_povm_states(povm, states)

    expected = np.array(
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
        ]
    )
    assert P.shape == (2, 4)
    np.testing.assert_allclose(P, expected, atol=1e-12)
    validate_probability_matrix(P)


def test_probability_vector_from_povm_state_accepts_vector_and_density_matrix():
    zero = density([1, 0])
    one = density([0, 1])
    plus = ket([1, 1])

    povm = np.stack([zero, one])

    from_vector = probability_vector_from_povm_state(povm, plus)
    from_density = probability_vector_from_povm_state(povm, density(plus))

    np.testing.assert_allclose(from_vector, np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_allclose(from_density, from_vector, atol=1e-12)


def test_pure_state_density_matrix_normalizes_vector():
    rho = ket2dm([2.0, 2.0j])

    expected = density([1.0, 1.0j])
    np.testing.assert_allclose(rho, expected, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)


def test_povm_effects_probability_vector_matches_probability_matrix_column():
    rng = np.random.default_rng(123)
    povm = POVM.random_isometry(nout=5, dim=2, rng=rng)
    state = density([1.0, 1.0j])

    p = povm.probability_vector(state)
    P = povm.probability_matrix(QuantumStateBatch.from_state_like(state, dim=2))

    np.testing.assert_allclose(p, P[:, 0], atol=1e-12)
    np.testing.assert_allclose(np.sum(p), 1.0, atol=1e-12)


def test_four_outcome_qubit_povm_probability_matrix():
    zero = density([1, 0])
    one = density([0, 1])
    plus = density([1, 1])
    minus = density([1, -1])

    # Half-weighted Z-basis and X-basis projectors form a valid POVM:
    # 0.5(|0><0| + |1><1| + |+><+| + |-><-|) = I.
    povm = 0.5 * np.stack([zero, one, plus, minus])
    states = np.stack([zero, one, plus, minus])

    P = probability_matrix_from_povm_states(povm, states)

    expected = np.array(
        [
            [0.5, 0.0, 0.25, 0.25],
            [0.0, 0.5, 0.25, 0.25],
            [0.25, 0.25, 0.5, 0.0],
            [0.25, 0.25, 0.0, 0.5],
        ]
    )
    assert P.shape == (4, 4)
    np.testing.assert_allclose(P, expected, atol=1e-12)
    validate_probability_matrix(P)


def test_generate_qubit_mub_povm_is_complete_six_outcome_povm():
    povm = generate_qubit_mub_povm()

    assert povm.shape == (6, 2, 2)
    np.testing.assert_allclose(povm, povm.conj().transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(np.sum(povm, axis=0), np.eye(2), atol=1e-12)

    for effect in povm:
        evals = np.linalg.eigvalsh(effect)
        np.testing.assert_allclose(evals, np.array([0.0, 1.0 / 3.0]), atol=1e-12)


def test_generate_qubit_mub_povm_probability_matrix_for_own_basis_states():
    povm = generate_qubit_mub_povm()
    states = np.stack(
        [
            density([1, 0]),
            density([0, 1]),
            density([1, 1]),
            density([1, -1]),
            density([1, 1j]),
            density([1, -1j]),
        ]
    )

    P = probability_matrix_from_povm_states(povm, states)

    expected = np.array(
        [
            [1 / 3, 0, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            [0, 1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 1 / 3, 0, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 0, 1 / 3, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3, 0],
            [1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 1 / 3],
        ]
    )
    np.testing.assert_allclose(P, expected, atol=1e-12)
    validate_probability_matrix(P)


def test_complex_state_probabilities_are_real():
    zero = density([1, 0])
    one = density([0, 1])
    plus_i = density([1, 1j])

    P = probability_matrix_from_povm_states(
        povm_effects=np.stack([zero, one]),
        states=np.stack([plus_i]),
    )

    np.testing.assert_allclose(P, np.array([[0.5], [0.5]]), atol=1e-12)
    assert np.issubdtype(P.dtype, np.floating)


def test_rejects_effects_that_do_not_sum_to_identity():
    zero = density([1, 0])
    one = density([0, 1])

    with pytest.raises(ValueError, match="sum to identity"):
        probability_matrix_from_povm_states(
            povm_effects=np.stack([zero, 0.5 * one]),
            states=np.stack([zero]),
        )


def test_rejects_state_with_wrong_trace():
    zero = density([1, 0])
    one = density([0, 1])

    with pytest.raises(ValueError, match="trace 1"):
        probability_matrix_from_povm_states(
            povm_effects=np.stack([zero, one]),
            states=np.stack([0.5 * zero]),
        )


def test_generate_haar_random_kets_are_normalized():
    rng = np.random.default_rng(123)

    vectors = generate_haar_random_kets(num_states=7, dim=3, rng=rng)

    assert vectors.shape == (3, 7)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), np.ones(7), atol=1e-12)
    assert np.iscomplexobj(vectors)


def test_default_rng_reproducibly_feeds_calls_without_explicit_rng():
    try:
        set_default_rng(123)
        first = generate_haar_random_kets(num_states=3, dim=2)
        second = generate_haar_random_kets(num_states=3, dim=2)

        set_default_rng(123)
        first_again = generate_haar_random_kets(num_states=3, dim=2)
        second_again = generate_haar_random_kets(num_states=3, dim=2)
    finally:
        clear_default_rng()

    np.testing.assert_allclose(first, first_again, atol=1e-12)
    np.testing.assert_allclose(second, second_again, atol=1e-12)
    assert not np.allclose(first, second)


def test_generate_haar_random_pure_dms_are_density_matrices():
    rng = np.random.default_rng(123)

    states = generate_haar_random_pure_dms(num_states=5, dim=4, rng=rng)

    assert states.shape == (5, 4, 4)
    np.testing.assert_allclose(states, states.conj().transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(np.trace(states, axis1=1, axis2=2), np.ones(5), atol=1e-12)

    for rho in states:
        evals = np.linalg.eigvalsh(rho)
        np.testing.assert_allclose(evals[:-1], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(evals[-1], 1.0, atol=1e-12)


def test_generate_haar_random_isometry_has_orthonormal_columns():
    rng = np.random.default_rng(123)

    isometry = generate_haar_random_isometry(nout=6, dim=3, rng=rng)

    assert isometry.shape == (6, 3)
    np.testing.assert_allclose(isometry.conj().T @ isometry, np.eye(3), atol=1e-12)


def test_generate_random_rank1_povm_is_complete_and_rank_one():
    rng = np.random.default_rng(123)

    effects = generate_random_rank1_povm(nout=6, dim=3, rng=rng)

    assert effects.shape == (6, 3, 3)
    np.testing.assert_allclose(effects, effects.conj().transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(np.sum(effects, axis=0), np.eye(3), atol=1e-12)

    for effect in effects:
        evals = np.linalg.eigvalsh(effect)
        assert np.count_nonzero(evals > 1e-12) <= 1
        assert np.min(evals) >= -1e-12


def test_haar_probability_moments_for_projective_qubit_measurement():
    isometry = np.eye(2, dtype=complex)

    mean, second = haar_probability_moments_from_isometry(isometry)

    np.testing.assert_allclose(mean, np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_allclose(
        second,
        np.array(
            [
                [1.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 3.0],
            ]
        ),
        atol=1e-12,
    )


def test_random_rank1_povm_and_haar_states_make_probability_matrix():
    rng = np.random.default_rng(123)
    effects = generate_random_rank1_povm(nout=5, dim=2, rng=rng)
    states = generate_haar_random_pure_dms(num_states=11, dim=2, rng=rng)

    P = probability_matrix_from_povm_states(effects, states)

    assert P.shape == (5, 11)
    validate_probability_matrix(P, atol=1e-10)


def test_povm_effects_reject_invalid_completion():
    zero = density([1, 0])
    one = density([0, 1])

    with pytest.raises(ValueError, match="sum to identity"):
        POVM.from_effects(np.stack([zero, 0.5 * one]))


def test_povm_effects_and_state_batch_make_probability_matrix():
    rng = np.random.default_rng(123)

    povm = POVM.random_rank1(nout=5, dim=2, rng=rng)
    states = QuantumStateBatch.haar_pure(num_states=11, dim=2, rng=rng)
    P = povm.probability_matrix(states)
    expected = probability_matrix_from_povm_states(
        povm.effects,
        states.states,
        validate_inputs=False,
    )

    assert povm.nout == 5
    assert povm.dim == 2
    np.testing.assert_allclose(P, expected, atol=1e-12)
    assert povm._effect_rows_cache is None
    assert states._state_rows_cache is None
    validate_probability_matrix(P, atol=1e-10)


def test_povm_effects_haar_moments_match_isometry_formula():
    rng = np.random.default_rng(123)

    povm = POVM.random_isometry(nout=6, dim=3, rng=rng)
    mean, second = povm.haar_probability_moments()
    expected_mean, expected_second = haar_probability_moments_from_isometry(povm.isometry)

    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(second, expected_second, atol=1e-12)
    assert povm._effect_rows_cache is None


def test_povm_haar_moments_use_effects_not_optional_isometry():
    rng = np.random.default_rng(123)
    effect_isometry = generate_haar_random_isometry(nout=6, dim=3, rng=rng)
    stale_isometry = generate_haar_random_isometry(nout=6, dim=3, rng=rng)
    effects = np.einsum("ai,aj->aij", effect_isometry.conj(), effect_isometry)
    povm = POVM(effects, isometry=stale_isometry)

    mean, second = povm.haar_probability_moments()
    expected_mean, expected_second = haar_probability_moments_from_isometry(effect_isometry)
    stale_mean, stale_second = haar_probability_moments_from_isometry(stale_isometry)

    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(second, expected_second, atol=1e-12)
    assert not np.allclose(mean, stale_mean, atol=1e-12)
    assert not np.allclose(second, stale_second, atol=1e-12)
    assert povm._effect_rows_cache is None


def test_qelm_quantum_dataset_training_dual_rows_match_composed_frame_formula():
    rng = np.random.default_rng(123)
    dataset = QELMQuantumDataset.random_isometry(
        nout=6,
        ntr=8,
        ntest=3,
        dim=2,
        rng=rng,
        rcond=1e-12,
    )

    frame = dataset._train_state_rows.T @ dataset._train_state_rows.conj()
    frame_pinv = np.linalg.pinv(frame, rcond=1e-12)
    effect_frame = dataset._effect_rows.T @ dataset._effect_rows.conj()
    povm_dual_rows = (np.linalg.pinv(effect_frame, rcond=1e-12) @ dataset._effect_rows.T).T
    expected = (frame_pinv @ povm_dual_rows.T).T

    np.testing.assert_allclose(dataset.training_dual_effect_rows, expected, atol=1e-12)
    np.testing.assert_allclose(dataset.povm_dual_effect_rows, povm_dual_rows, atol=1e-12)
    assert dataset.P_train.shape == (6, 8)
    assert dataset.P_test.shape == (6, 3)
    assert dataset.dual_P_train.shape == (6, 8)
    assert dataset.dual_P_test.shape == (6, 3)


def test_random_isometry_rejects_too_few_outputs():
    with pytest.raises(ValueError, match="nout must be at least dim"):
        generate_haar_random_isometry(nout=2, dim=3, rng=np.random.default_rng(123))
