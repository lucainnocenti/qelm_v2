import numpy as np

from qelm_rank.blocks import (
    deterministic_blocks_from_P,
    schur_covariance_blocks,
    svd_probability_blocks,
)
from qelm_rank.quantum import POVMEffects, QuantumStateBatch


def make_toy_rank_r_probability_matrix(nout, ntr, r, rng):
    basis = rng.dirichlet(np.ones(nout), size=r).T
    coeffs = rng.dirichlet(np.ones(r), size=ntr).T
    return basis @ coeffs


def _explicit_weighted_covariance_block(P, Ua, Ub, pi2_diag):
    weights = np.asarray(pi2_diag, dtype=float)
    weights = weights / np.sum(weights)

    out = np.zeros((Ua.shape[1], Ub.shape[1]))
    for i, weight in enumerate(weights):
        p = P[:, i]
        sigma_i = np.diag(p) - np.outer(p, p)
        out += weight * (Ua.T @ sigma_i @ Ub)
    return out


def _assert_svd_block_decomposition(P, r, atol=1e-11):
    blocks = svd_probability_blocks(P, rank=r)
    U1 = blocks["U1"]
    U2 = blocks["U2"]
    V1 = blocks["V1"]
    V2 = blocks["V2"]

    np.testing.assert_allclose(U1.T @ U1, np.eye(r), atol=atol)
    np.testing.assert_allclose(U2.T @ U2, np.eye(P.shape[0] - r), atol=atol)
    np.testing.assert_allclose(V1.T @ V1, np.eye(r), atol=atol)
    np.testing.assert_allclose(V2.T @ V2, np.eye(P.shape[1] - r), atol=atol)
    np.testing.assert_allclose(U1.T @ U2, 0.0, atol=atol)
    np.testing.assert_allclose(V1.T @ V2, 0.0, atol=atol)

    singular_values = np.linalg.svd(P, compute_uv=False)
    P11 = U1.T @ P @ V1
    P12 = U1.T @ P @ V2
    P21 = U2.T @ P @ V1
    P22 = U2.T @ P @ V2

    np.testing.assert_allclose(P11, np.diag(singular_values[:r]), atol=atol)
    np.testing.assert_allclose(
        np.linalg.svd(P11, compute_uv=False),
        singular_values[:r],
        atol=atol,
    )
    np.testing.assert_allclose(P12, 0.0, atol=atol)
    np.testing.assert_allclose(P21, 0.0, atol=atol)
    np.testing.assert_allclose(P22, 0.0, atol=atol)

    np.testing.assert_allclose(blocks["Pi2_diag"], np.sum(V2 * V2, axis=1), atol=atol)
    np.testing.assert_allclose(np.sum(blocks["Pi2_diag"]), P.shape[1] - r, atol=atol)


def _assert_covariance_blocks_match_explicit_sum(P, r, atol=1e-11):
    blocks = svd_probability_blocks(P, rank=r)
    C12, C22 = schur_covariance_blocks(
        P,
        blocks["U1"],
        blocks["U2"],
        blocks["Pi2_diag"],
    )

    expected_C12 = _explicit_weighted_covariance_block(
        P,
        blocks["U1"],
        blocks["U2"],
        blocks["Pi2_diag"],
    )
    expected_C22 = _explicit_weighted_covariance_block(
        P,
        blocks["U2"],
        blocks["U2"],
        blocks["Pi2_diag"],
    )
    np.testing.assert_allclose(C12, expected_C12, atol=atol)
    np.testing.assert_allclose(C22, expected_C22, atol=atol)

    deterministic = deterministic_blocks_from_P(P, r=r)
    expected_C11 = _explicit_weighted_covariance_block(
        P,
        deterministic.U1,
        deterministic.U1,
        deterministic.w,
    )
    expected_C12 = _explicit_weighted_covariance_block(
        P,
        deterministic.U1,
        deterministic.U2,
        deterministic.w,
    )
    expected_C22 = _explicit_weighted_covariance_block(
        P,
        deterministic.U2,
        deterministic.U2,
        deterministic.w,
    )
    np.testing.assert_allclose(deterministic.C11, expected_C11, atol=atol)
    np.testing.assert_allclose(deterministic.C12, expected_C12, atol=atol)
    np.testing.assert_allclose(deterministic.C22, expected_C22, atol=atol)


def test_svd_probability_blocks_decompose_toy_rank_r_probability_matrix():
    P = make_toy_rank_r_probability_matrix(
        nout=32,
        ntr=16,
        r=4,
        rng=np.random.default_rng(42),
    )

    _assert_svd_block_decomposition(P, r=4)
    _assert_covariance_blocks_match_explicit_sum(P, r=4)


def test_svd_probability_blocks_decompose_state_povm_probability_matrix():
    rng = np.random.default_rng(123)
    povm = POVMEffects.random_isometry(nout=32, dim=2, rng=rng)
    states = QuantumStateBatch.haar_pure_from_columns(num_states=16, dim=2, rng=rng)
    P = povm.probability_matrix(states)

    _assert_svd_block_decomposition(P, r=4)
    _assert_covariance_blocks_match_explicit_sum(P, r=4)
