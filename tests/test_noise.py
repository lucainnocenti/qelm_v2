import numpy as np

from qelm_rank import (
    generate_gaussian_Xi,
    generate_multinomial_Xi,
    noisy_probability_matrix,
    sample_finite_shot_probability_matrix,
    sample_shot_noise,
    shot_noise_matrix,
)


def test_sample_finite_shot_probability_matrix_matches_manual_multinomial_columns():
    P = np.array(
        [
            [0.2, 0.1, 0.4],
            [0.3, 0.6, 0.4],
            [0.5, 0.3, 0.2],
        ]
    )
    N = 20

    sampled = sample_finite_shot_probability_matrix(
        P,
        N=N,
        rng=np.random.default_rng(123),
    )
    manual_rng = np.random.default_rng(123)
    expected = np.column_stack(
        [
            manual_rng.multinomial(N, P[:, i]) / N
            for i in range(P.shape[1])
        ]
    )

    np.testing.assert_allclose(sampled, expected, atol=0.0)
    np.testing.assert_allclose(sampled.sum(axis=0), 1.0, atol=1e-12)
    assert np.all((N * sampled).round() == N * sampled)


def test_noisy_probability_matrix_multinomial_returns_empirical_probabilities():
    P = np.array(
        [
            [0.2, 0.1, 0.4],
            [0.3, 0.6, 0.4],
            [0.5, 0.3, 0.2],
        ]
    )
    N = 20

    sampled = noisy_probability_matrix(
        P,
        rng=np.random.default_rng(123),
        Nshots=N,
        noise="multinomial",
    )
    expected = sample_finite_shot_probability_matrix(
        P,
        N=N,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(sampled, expected, atol=0.0)
    np.testing.assert_allclose(sampled.sum(axis=0), 1.0, atol=1e-12)


def test_multinomial_xi_is_derived_from_same_empirical_probability_draw():
    P = np.array(
        [
            [0.2, 0.1, 0.4],
            [0.3, 0.6, 0.4],
            [0.5, 0.3, 0.2],
        ]
    )
    N = 20

    Xi = generate_multinomial_Xi(P, N=N, rng=np.random.default_rng(123))
    P_hat = sample_finite_shot_probability_matrix(
        P,
        N=N,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(Xi, np.sqrt(N) * (P_hat - P), atol=1e-12)


def test_noisy_probability_matrix_gaussian_adds_scaled_xi():
    P = np.array(
        [
            [0.2, 0.1, 0.4],
            [0.3, 0.6, 0.4],
            [0.5, 0.3, 0.2],
        ]
    )
    N = 25

    sampled = noisy_probability_matrix(
        P,
        rng=np.random.default_rng(123),
        Nshots=N,
        noise="gaussian",
    )
    Xi = generate_gaussian_Xi(P, rng=np.random.default_rng(123))

    np.testing.assert_allclose(sampled, P + Xi / np.sqrt(N), atol=1e-12)


def test_sample_shot_noise_routes_xi_and_probability_outputs():
    P = np.array(
        [
            [0.2, 0.1, 0.4],
            [0.3, 0.6, 0.4],
            [0.5, 0.3, 0.2],
        ]
    )
    N = 25

    Xi = sample_shot_noise(
        P,
        rng=np.random.default_rng(123),
        Nshots=N,
        noise="gaussian",
        output="xi",
    )
    P_hat = sample_shot_noise(
        P,
        rng=np.random.default_rng(123),
        Nshots=N,
        noise="gaussian",
        output="probability",
    )

    np.testing.assert_allclose(Xi, shot_noise_matrix(P, rng=np.random.default_rng(123), Nshots=N))
    np.testing.assert_allclose(P_hat, P + Xi / np.sqrt(N), atol=1e-12)
