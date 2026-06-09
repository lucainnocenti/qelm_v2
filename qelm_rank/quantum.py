from typing import Sequence

import numpy as np

from .linalg import validate_probability_matrix


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _as_operator_stack(operators: Sequence[np.ndarray] | np.ndarray, name: str) -> np.ndarray:
    stack = np.asarray(operators, dtype=complex)

    if stack.ndim != 3:
        raise ValueError(f"{name} must be an array-like stack with shape (n, d, d).")
    if stack.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one operator.")
    if stack.shape[1] != stack.shape[2]:
        raise ValueError(f"{name} operators must be square matrices.")
    if not np.all(np.isfinite(stack)):
        raise ValueError(f"{name} contains non-finite entries.")

    return stack


def _check_hermitian_psd(stack: np.ndarray, name: str, atol: float) -> None:
    for index, operator in enumerate(stack):
        if not np.allclose(operator, operator.conj().T, atol=atol):
            raise ValueError(f"{name}[{index}] is not Hermitian within tolerance.")

        hermitian_part = (operator + operator.conj().T) / 2
        evals = np.linalg.eigvalsh(hermitian_part)
        if np.min(evals) < -atol:
            raise ValueError(f"{name}[{index}] is not positive semidefinite within tolerance.")


def generate_haar_random_state_vectors(
    num_states: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Haar random pure state vectors in C^dim.

    Returns an array with shape (num_states, dim). Each row is normalized to
    unit Euclidean norm.
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    rng = _rng_or_default(rng)
    vectors = rng.standard_normal((num_states, dim)) + 1j * rng.standard_normal((num_states, dim))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def generate_haar_random_pure_states(
    num_states: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Haar random pure states rho_i = |psi_i><psi_i|.

    Returns an array with shape (num_states, dim, dim).
    """
    vectors = generate_haar_random_state_vectors(num_states=num_states, dim=dim, rng=rng)
    return np.einsum("ij,ik->ijk", vectors, vectors.conj())


def generate_haar_random_isometry(
    nout: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a Haar random isometry V with shape (nout, dim).

    The columns of V are orthonormal, so V.conj().T @ V = I_dim.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if nout < dim:
        raise ValueError("nout must be at least dim to generate an nout x dim isometry.")

    rng = _rng_or_default(rng)
    gaussian = rng.standard_normal((nout, dim)) + 1j * rng.standard_normal((nout, dim))
    q, r = np.linalg.qr(gaussian, mode="reduced")

    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q * phases


def generate_random_rank1_povm(
    nout: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a random rank-1 POVM from a Haar random nout x dim isometry.

    If v_a is row a of the isometry V, the effect is
        mu_a = |v_a><v_a|.
    These effects are generally unnormalized rank-1 positive semidefinite
    matrices and satisfy sum_a mu_a = I_dim.
    """
    isometry = generate_haar_random_isometry(nout=nout, dim=dim, rng=rng)
    return np.einsum("ai,aj->aij", isometry.conj(), isometry)


def probability_matrix_from_povm_states(
    povm_effects: Sequence[np.ndarray] | np.ndarray,
    states: Sequence[np.ndarray] | np.ndarray,
    *,
    validate_inputs: bool = True,
    validate_output: bool = True,
    atol: float = 1e-10,
) -> np.ndarray:
    """
    Build the probability matrix P[a, i] = Tr(mu_a rho_i).

    Parameters
    ----------
    povm_effects:
        Stack or sequence of POVM effects with shape (nout, d, d).
    states:
        Stack or sequence of density matrices with shape (ntr, d, d).
    validate_inputs:
        If True, check Hermiticity, positive semidefiniteness, unit-trace states,
        and that the POVM effects sum to identity.
    validate_output:
        If True, check that the resulting matrix is column-stochastic.
    atol:
        Numerical tolerance for validation and tiny imaginary/negative parts.

    Returns
    -------
    np.ndarray
        Real array with shape (nout, ntr), where nout is the number of POVM
        effects and ntr is the number of states.
    """
    effects = _as_operator_stack(povm_effects, "povm_effects")
    rho = _as_operator_stack(states, "states")

    if effects.shape[1] != rho.shape[1]:
        raise ValueError(
            "POVM effects and states must have the same Hilbert-space dimension."
        )

    dim = effects.shape[1]

    if validate_inputs:
        _check_hermitian_psd(effects, "povm_effects", atol=atol)
        _check_hermitian_psd(rho, "states", atol=atol)

        povm_sum = np.sum(effects, axis=0)
        if not np.allclose(povm_sum, np.eye(dim), atol=atol):
            max_err = np.max(np.abs(povm_sum - np.eye(dim)))
            raise ValueError(f"POVM effects do not sum to identity. Max error: {max_err:.3e}")

        state_traces = np.trace(rho, axis1=1, axis2=2)
        if not np.allclose(state_traces, 1.0, atol=atol):
            max_err = np.max(np.abs(state_traces - 1.0))
            raise ValueError(f"States do not all have trace 1. Max error: {max_err:.3e}")

    probabilities = np.einsum("ajk,ikj->ai", effects, rho)

    max_imag = np.max(np.abs(np.imag(probabilities)))
    if max_imag > atol:
        raise ValueError(f"Computed probabilities have non-negligible imaginary part: {max_imag:.3e}")

    P = np.real(probabilities)
    P = np.where((P < 0.0) & (P >= -atol), 0.0, P)

    if validate_output:
        validate_probability_matrix(P, atol=atol)

    return P
