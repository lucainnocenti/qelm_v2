import numpy as np
import pytest

from qelm_rank import (
    generate_haar_random_isometry,
    generate_haar_random_pure_states,
    generate_haar_random_state_vectors,
    generate_random_rank1_povm,
    probability_matrix_from_povm_states,
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


def test_generate_haar_random_state_vectors_are_normalized():
    rng = np.random.default_rng(123)

    vectors = generate_haar_random_state_vectors(num_states=7, dim=3, rng=rng)

    assert vectors.shape == (7, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(7), atol=1e-12)
    assert np.iscomplexobj(vectors)


def test_generate_haar_random_pure_states_are_density_matrices():
    rng = np.random.default_rng(123)

    states = generate_haar_random_pure_states(num_states=5, dim=4, rng=rng)

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


def test_random_rank1_povm_and_haar_states_make_probability_matrix():
    rng = np.random.default_rng(123)
    effects = generate_random_rank1_povm(nout=5, dim=2, rng=rng)
    states = generate_haar_random_pure_states(num_states=11, dim=2, rng=rng)

    P = probability_matrix_from_povm_states(effects, states)

    assert P.shape == (5, 11)
    validate_probability_matrix(P, atol=1e-10)


def test_random_isometry_rejects_too_few_outputs():
    with pytest.raises(ValueError, match="nout must be at least dim"):
        generate_haar_random_isometry(nout=2, dim=3, rng=np.random.default_rng(123))
