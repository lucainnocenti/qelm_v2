from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .linalg import validate_probability_matrix


_DEFAULT_RNG: np.random.Generator | None = None


def set_default_rng(seed_or_rng: int | np.random.Generator | None = None) -> np.random.Generator:
    """
    Set the package-level default RNG used when functions receive rng=None.

    Pass an integer seed for reproducible notebook cells, an existing Generator
    to share its stream, or None to create a fresh unpredictable stream.
    """
    global _DEFAULT_RNG
    if isinstance(seed_or_rng, np.random.Generator):
        _DEFAULT_RNG = seed_or_rng
    else:
        _DEFAULT_RNG = np.random.default_rng(seed_or_rng)
    return _DEFAULT_RNG


def clear_default_rng() -> None:
    """Clear the package-level default RNG so rng=None uses fresh entropy."""
    global _DEFAULT_RNG
    _DEFAULT_RNG = None


def get_rng(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    """
    Resolve an RNG argument.

    Explicit Generators and integer seeds are honored. If rng is None and a
    package default has been set with set_default_rng, that shared stream is
    used. Otherwise a fresh unpredictable Generator is returned.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is not None:
        return np.random.default_rng(rng)
    if _DEFAULT_RNG is not None:
        return _DEFAULT_RNG
    return np.random.default_rng()


def _rng_or_default(rng: np.random.Generator | int | None) -> np.random.Generator:
    return get_rng(rng)


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


def _state_like_to_density(state, *, dim: int, name: str = "state") -> np.ndarray:
    state = np.asarray(state, dtype=complex)
    if state.ndim == 1:
        return _pure_state_vector_to_density(state, dim, name=name)
    if state.ndim == 2:
        if state.shape != (dim, dim):
            raise ValueError(
                f"{name} must be a state vector with shape ({dim},) "
                f"or a density matrix with shape ({dim}, {dim})."
            )
        return state
    raise ValueError(f"{name} must be a state vector or density matrix.")


def _clean_probability_matrix_columns(P: np.ndarray, *, atol: float) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError("P must be a 2D array.")
    if not np.all(np.isfinite(P)):
        raise ValueError("P contains non-finite entries.")
    if np.any(P < -atol):
        raise ValueError("P has negative entries beyond tolerance.")

    P = np.where((P < 0.0) & (P >= -atol), 0.0, P)
    col_sums = P.sum(axis=0, keepdims=True)
    if np.any(col_sums <= 0):
        raise ValueError("Each column of P must have positive total probability mass.")
    return P / col_sums


def probability_matrix_from_operator_rows(
    effect_rows: np.ndarray,
    state_rows: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    # this computes the probability matrix P[a, i] = Tr(mu_a rho_i) from the effect rows and state rows
    # by "rows" we mean that each effect mu_a is flattened into a row vector, and each state rho_i is flattened into a row vector, so the trace is just a dot product of the effect row with the state row.
    # This allows us to compute the entire probability matrix as a single matrix multiplication.
    probabilities = np.asarray(effect_rows, dtype=complex).conj() @ np.asarray(state_rows, dtype=complex).T
    probabilities = probabilities.real
    if clip:
        probabilities = np.maximum(probabilities, 0.0)
    return probabilities


def probability_vector_from_povm_state(
    povm_effects: Sequence[np.ndarray] | np.ndarray,
    state: np.ndarray,
    *,
    validate_inputs: bool = True,
    validate_output: bool = True,
    atol: float = 1e-10,
) -> np.ndarray:
    """
    Compute p[a] = Tr(mu_a rho) directly from POVM effects and one state.

    state can be either a pure state vector with shape (d,) or a density
    matrix with shape (d, d).
    """
    effects = _as_operator_stack(povm_effects, "povm_effects")
    dim = effects.shape[1]
    rho = _state_like_to_density(state, dim=dim)

    if validate_inputs:
        _check_hermitian_psd(effects, "povm_effects", atol=atol)
        _check_hermitian_psd(rho[None, :, :], "state", atol=atol)

        povm_sum = np.sum(effects, axis=0)
        if not np.allclose(povm_sum, np.eye(dim), atol=atol):
            max_err = np.max(np.abs(povm_sum - np.eye(dim)))
            raise ValueError(f"POVM effects do not sum to identity. Max error: {max_err:.3e}")

        state_trace = np.trace(rho)
        if not np.allclose(state_trace, 1.0, atol=atol):
            max_err = np.abs(state_trace - 1.0)
            raise ValueError(f"State does not have trace 1. Error: {max_err:.3e}")

    probabilities = np.einsum("aij,ji->a", effects, rho)
    max_imag = np.max(np.abs(np.imag(probabilities)))
    if max_imag > atol:
        raise ValueError(f"Computed probabilities have non-negligible imaginary part: {max_imag:.3e}")

    p = np.real(probabilities)
    p = np.where((p < 0.0) & (p >= -atol), 0.0, p)

    if validate_output:
        validate_probability_matrix(p[:, None], atol=atol)

    return p


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
    # P = np.where((P < 0.0) & (P >= -atol), 0.0, P)

    if validate_output:
        validate_probability_matrix(P, atol=atol)

    return P


def sample_finite_shot_probability_matrix(
    P: np.ndarray,
    N: int,
    rng: np.random.Generator | None = None,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Sample empirical probabilities from exact column probabilities.

    Returns P_hat[:, i] = Multinomial(N, P[:, i]) / N for each column i.
    """
    if int(N) != N or N <= 0:
        raise ValueError("N must be a positive integer.")

    rng = _rng_or_default(rng)
    P = _clean_probability_matrix_columns(P, atol=atol)
    nout, ntr = P.shape
    P_hat = np.empty((nout, ntr), dtype=float)

    for i in range(ntr):
        P_hat[:, i] = rng.multinomial(int(N), P[:, i]) / int(N)

    return P_hat


def haar_moments_from_operator_rows(
    operator_rows: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    operators = np.asarray(operator_rows, dtype=complex).reshape(operator_rows.shape[0], dim, dim)
    traces = np.trace(operators, axis1=1, axis2=2)
    trace_products = np.einsum("aij,bji->ab", operators, operators, optimize=True)

    mean = traces / dim
    second = (np.outer(traces, traces) + trace_products) / (dim * (dim + 1))
    return mean.real, second.real


def dual_rows_for_operator_frame(
    operator_rows: np.ndarray,
    rcond: float,
) -> np.ndarray:
    rows = np.asarray(operator_rows, dtype=complex)
    frame = rows.T @ rows.conj()
    frame_pinv = np.linalg.pinv(frame, rcond=rcond)
    return (frame_pinv @ rows.T).T


def _pure_state_vector_to_density(vector: np.ndarray, dim: int, name: str = "state") -> np.ndarray:
    vector = np.asarray(vector, dtype=complex)
    if vector.ndim != 1 or vector.shape[0] != dim:
        raise ValueError(f"{name} vector must have shape ({dim},).")
    norm = np.linalg.norm(vector)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError(f"{name} vector must have finite nonzero norm.")
    vector = vector / norm
    return np.outer(vector, vector.conj())


def generate_haar_random_state_vectors(
    num_states: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Haar random pure state vectors as rows.

    This compatibility wrapper preserves the original public API. New code that
    naturally works with column vectors can use
    generate_haar_random_state_vector_columns instead.
    """
    return generate_haar_random_state_vector_columns(
        num_states=num_states,
        dim=dim,
        rng=rng,
    ).T


def generate_haar_random_state_vector_columns(
    num_states: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Haar random pure state vectors as columns.

    Returns an array with shape (dim, num_states).
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    rng = _rng_or_default(rng)
    vectors = (rng.normal(size=(dim, num_states)) + 1j * rng.normal(size=(dim, num_states))) / np.sqrt(2.0)
    vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
    return vectors


def generate_haar_random_pure_states(
    num_states: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Haar random pure density matrices rho_i = |psi_i><psi_i|.

    Returns an array with shape (num_states, dim, dim).
    """
    vectors = generate_haar_random_state_vector_columns(num_states=num_states, dim=dim, rng=rng)
    return np.einsum("ji,ki->ijk", vectors, vectors.conj())


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


def generate_haar_random_isometry_conjugate_phase(
    nout: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a Haar random isometry using the conjugate QR phase convention.

    This convention is used by the Schur-correction scaling notebook. It has
    the same distribution as generate_haar_random_isometry, but preserves that
    notebook's seeded numerical output.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if nout < dim:
        raise ValueError("nout must be at least dim to generate an nout x dim isometry.")

    rng = _rng_or_default(rng)
    gaussian = (rng.normal(size=(nout, dim)) + 1j * rng.normal(size=(nout, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(gaussian)

    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return (q * phases.conj()[None, :])[:, :dim]


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


@dataclass(frozen=True)
class QuantumStateBatch:
    states: np.ndarray

    def __post_init__(self) -> None:
        states = _as_operator_stack(self.states, "states")
        if not np.allclose(states, states.conj().transpose(0, 2, 1), atol=1e-10):
            raise ValueError("States must be Hermitian.")
        traces = np.trace(states, axis1=1, axis2=2)
        if not np.allclose(traces, 1.0, atol=1e-10):
            max_err = np.max(np.abs(traces - 1.0))
            raise ValueError(f"States must have trace 1. Max error: {max_err:.3e}")
        object.__setattr__(self, "states", states)

    @classmethod
    def from_state_like(cls, state, *, dim: int, name: str = "state") -> "QuantumStateBatch":
        state = np.asarray(state, dtype=complex)
        if state.ndim in {1, 2}:
            return cls(_state_like_to_density(state, dim=dim, name=name)[None, :, :])
        if state.ndim == 3:
            if state.shape[1:] != (dim, dim):
                raise ValueError(f"{name} batch must have shape (n, {dim}, {dim}).")
            return cls(state)
        raise ValueError(
            f"{name} must be a state vector, density matrix, or density-matrix batch."
        )

    @classmethod
    def from_state_vectors(
        cls,
        state_vectors,
        *,
        dim: int,
        axis: str = "auto",
        name: str = "state_vectors",
    ) -> "QuantumStateBatch":
        vectors = np.asarray(state_vectors, dtype=complex)
        axis = axis.lower()
        if axis not in {"auto", "rows", "columns"}:
            raise ValueError("axis must be 'auto', 'rows', or 'columns'.")

        if vectors.ndim == 1:
            return cls(_pure_state_vector_to_density(vectors, dim=dim, name=name)[None, :, :])

        if vectors.ndim != 2:
            raise ValueError(f"{name} must be a vector or a 2D array of vectors.")

        if axis == "auto":
            rows_match = vectors.shape[1] == dim
            cols_match = vectors.shape[0] == dim
            if rows_match == cols_match:
                raise ValueError(
                    f"{name} has ambiguous shape {vectors.shape}; pass axis='rows' "
                    "or axis='columns'."
                )
            axis = "rows" if rows_match else "columns"

        if axis == "rows":
            if vectors.shape[1] != dim:
                raise ValueError(f"{name} rows must have shape (n, {dim}).")
            states = np.stack(
                [_pure_state_vector_to_density(vector, dim=dim, name=f"{name}[{i}]")
                 for i, vector in enumerate(vectors)],
                axis=0,
            )
            return cls(states)

        if vectors.shape[0] != dim:
            raise ValueError(f"{name} columns must have shape ({dim}, n).")
        states = np.stack(
            [_pure_state_vector_to_density(vectors[:, i], dim=dim, name=f"{name}[:, {i}]")
             for i in range(vectors.shape[1])],
            axis=0,
        )
        return cls(states)

    @classmethod
    def haar_pure(
        cls,
        num_states: int,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> "QuantumStateBatch":
        return cls(generate_haar_random_pure_states(num_states=num_states, dim=dim, rng=rng))

    @classmethod
    def haar_pure_from_columns(
        cls,
        num_states: int,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> "QuantumStateBatch":
        vectors = generate_haar_random_state_vector_columns(
            num_states=num_states,
            dim=dim,
            rng=rng,
        )
        states = np.einsum("ik,jk->kij", vectors, vectors.conj())
        return cls(states)

    @property
    def num_states(self) -> int:
        return self.states.shape[0]

    @property
    def dim(self) -> int:
        return self.states.shape[1]

    # this seems to be vectorizing the states by flattening them, is this correct?
    @property
    def state_rows(self) -> np.ndarray:
        return self.states.reshape(self.num_states, -1)


@dataclass(frozen=True)
class POVMEffects:
    effects: np.ndarray
    label: str | None = None
    isometry: np.ndarray | None = None
    atol: float = 1e-10

    def __post_init__(self) -> None:
        effects = np.asarray(self.effects, dtype=complex)

        if effects.ndim != 3:
            raise ValueError("Explicit POVM effects must have shape (nout, d, d).")
        if effects.shape[1] != effects.shape[2]:
            raise ValueError("Explicit POVM effects must be square matrices.")
        if effects.shape[0] <= 0 or effects.shape[1] <= 0:
            raise ValueError("Explicit POVM effects must have positive shape.")
        if not np.all(np.isfinite(effects)):
            raise ValueError("Explicit POVM effects contain non-finite entries.")
        if not np.allclose(effects, effects.conj().transpose(0, 2, 1), atol=self.atol):
            raise ValueError("Explicit POVM effects must be Hermitian.")

        eigvals = np.linalg.eigvalsh(effects)
        min_eig = float(np.min(eigvals))
        if min_eig < -self.atol:
            raise ValueError(
                "Explicit POVM effects must be positive semidefinite. "
                f"Min eigenvalue: {min_eig:.3e}"
            )

        povm_sum = np.sum(effects, axis=0)
        if not np.allclose(povm_sum, np.eye(effects.shape[1]), atol=self.atol):
            max_err = np.max(np.abs(povm_sum - np.eye(effects.shape[1])))
            raise ValueError(
                "Explicit POVM effects do not sum to identity. "
                f"Max error: {max_err:.3e}"
            )

        if self.isometry is not None:
            isometry = np.asarray(self.isometry, dtype=complex)
            if isometry.shape != (effects.shape[0], effects.shape[1]):
                raise ValueError("POVM isometry has shape incompatible with effects.")
            object.__setattr__(self, "isometry", isometry)
        object.__setattr__(self, "effects", effects)

    @classmethod
    def from_effects(
        cls,
        effects: Sequence[np.ndarray] | np.ndarray,
        *,
        dim: int | None = None,
        nout: int | None = None,
        label: str | None = None,
        atol: float = 1e-10,
    ) -> "POVMEffects":
        povm = cls(np.asarray(effects, dtype=complex), label=label, atol=atol)
        if nout is not None and povm.nout != int(nout):
            raise ValueError(
                f"Explicit POVM has nout={povm.nout}, but the config has nout={nout}."
            )
        if dim is not None and povm.dim != int(dim):
            raise ValueError(
                f"Explicit POVM has dimension d={povm.dim}, but the config has d={dim}."
            )
        return povm

    @classmethod
    def random_rank1(
        cls,
        nout: int,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> "POVMEffects":
        effects = generate_random_rank1_povm(nout=nout, dim=dim, rng=rng)
        return cls(effects, label="random_rank1")

    @classmethod
    def random_isometry(
        cls,
        nout: int,
        dim: int,
        rng: np.random.Generator | None = None,
        *,
        conjugate_phase: bool = True,
    ) -> "POVMEffects":
        generator = (
            generate_haar_random_isometry_conjugate_phase
            if conjugate_phase
            else generate_haar_random_isometry
        )
        isometry = generator(nout=nout, dim=dim, rng=rng)
        effects = np.einsum("ai,aj->aij", isometry.conj(), isometry)
        return cls(effects, label="random_isometry", isometry=isometry)

    @property
    def nout(self) -> int:
        return self.effects.shape[0]

    @property
    def dim(self) -> int:
        return self.effects.shape[1]

    @property
    def effect_rows(self) -> np.ndarray:
        return self.effects.reshape(self.nout, -1)

    def probability_matrix(
        self,
        states: QuantumStateBatch | Sequence[np.ndarray] | np.ndarray,
        *,
        clip: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        batch = states if isinstance(states, QuantumStateBatch) else QuantumStateBatch(states)
        if batch.dim != self.dim:
            raise ValueError("POVM effects and states must have the same dimension.")
        P = probability_matrix_from_operator_rows(self.effect_rows, batch.state_rows, clip=clip)
        if normalize:
            P /= P.sum(axis=0, keepdims=True)
        return P

    def probability_vector(
        self,
        state: np.ndarray,
        *,
        clip: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        p = probability_vector_from_povm_state(
            self.effects,
            state,
            validate_inputs=False,
            validate_output=False,
            atol=self.atol,
        )
        if clip:
            p = np.maximum(p, 0.0)
        if normalize:
            total = p.sum()
            if total <= 0:
                raise ValueError("Probability vector has non-positive total mass.")
            p = p / total
        return p

    def probability_matrix_from_rows(
        self,
        state_rows: np.ndarray,
        *,
        clip: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        # this just computes the probability matrix <mu,rho> of shape (nout, ntr), using the 
        # vectorized descriptions of effects and states
        P = probability_matrix_from_operator_rows(self.effect_rows, state_rows, clip=clip)
        # in theory this normalize shouldn't do nothing b/c things are already normalized...
        if normalize:
            P /= P.sum(axis=0, keepdims=True)
        return P

    def haar_probability_moments(self) -> tuple[np.ndarray, np.ndarray]:
        if self.isometry is not None:
            return haar_probability_moments_from_isometry(self.isometry)
        return haar_moments_from_operator_rows(self.effect_rows, dim=self.dim)

    def effect_frame_dual_rows(self, rcond: float) -> np.ndarray:
        return dual_rows_for_operator_frame(self.effect_rows, rcond=rcond)


@dataclass(frozen=True)
class QELMQuantumDataset:
    povm: POVMEffects
    train_states: QuantumStateBatch
    test_states: QuantumStateBatch | None = None
    rcond: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.povm, POVMEffects):
            raise TypeError("povm must be a POVMEffects object.")
        if not isinstance(self.train_states, QuantumStateBatch):
            raise TypeError("train_states must be a QuantumStateBatch.")
        if self.train_states.dim != self.povm.dim:
            raise ValueError(
                f"Training states have dimension d={self.train_states.dim}, "
                f"but the POVM has dimension d={self.povm.dim}."
            )
        if self.test_states is not None:
            if not isinstance(self.test_states, QuantumStateBatch):
                raise TypeError("test_states must be a QuantumStateBatch.")
            if self.test_states.dim != self.povm.dim:
                raise ValueError(
                    f"Test states have dimension d={self.test_states.dim}, "
                    f"but the POVM has dimension d={self.povm.dim}."
                )

    @classmethod
    def from_povm(
        cls,
        povm: POVMEffects,
        *,
        train_states: QuantumStateBatch,
        rcond: float | None = None,
        test_states: QuantumStateBatch | None = None,
    ) -> "QELMQuantumDataset":
        """
        Create a quantum dataset from a POVM and already-resolved states.
        """
        return cls(povm=povm, train_states=train_states, test_states=test_states, rcond=rcond)

    @classmethod
    def random_isometry(
        cls,
        *,
        nout: int,
        ntr: int,
        dim: int,
        rng: np.random.Generator | int | None = None,
        rcond: float = 1e-12,
        ntest: int | None = None,
    ) -> "QELMQuantumDataset":
        rng = get_rng(rng)
        povm = POVMEffects.random_isometry(nout=nout, dim=dim, rng=rng)
        train_states = QuantumStateBatch.haar_pure_from_columns(
            num_states=ntr,
            dim=dim,
            rng=rng,
        )
        test_states = None
        if ntest is not None:
            if ntest <= 0:
                raise ValueError("ntest must be positive.")
            test_states = QuantumStateBatch.haar_pure_from_columns(
                num_states=ntest,
                dim=dim,
                rng=rng,
            )
        return cls.from_povm(
            povm,
            train_states=train_states,
            test_states=test_states,
            rcond=rcond,
        )

    @property
    def effect_rows(self) -> np.ndarray:
        return self.povm.effect_rows

    @property
    def train_state_rows(self) -> np.ndarray:
        return self.train_states.state_rows

    @property
    def dual_effect_rows(self) -> np.ndarray:
        return self.training_dual_effect_rows

    @property
    def training_dual_effect_rows(self) -> np.ndarray:
        """Compute POVM-dual rows represented through the training-state dual frame.
        
        The leading-bias term uses <tilde mu_a, tilde sigma>, where tilde
        mu_a is the canonical dual of the POVM frame and tilde sigma is the
        state represented through the training-state dual frame. Equivalently,
        we can dualize the POVM-dual rows by the training-state frame and then
        pair the resulting rows with ordinary state rows.
        """
        frame = self.train_state_rows.T @ self.train_state_rows.conj()
        frame_pinv = np.linalg.pinv(frame, rcond=self.rcond)
        return (frame_pinv @ self.povm_dual_effect_rows.T).T

    @property
    def povm_dual_effect_rows(self) -> np.ndarray:
        """Compute the dual effect rows for the effects, via the POVM effects themselves."""
        return self.povm.effect_frame_dual_rows(rcond=self.rcond)

    @property
    def P_train(self) -> np.ndarray:
        return self.povm.probability_matrix(self.train_states)

    @property
    def dual_P_train(self) -> np.ndarray:
        """Compute the probability matrix but using the training dual effect rows instead of the actual POVM effects.
        
        The end result is the matrix <tilde mu_a, tilde rho_i>, where
        tilde mu_a are the POVM-frame dual effects and tilde rho_i are the
        training-state-frame dual states.
        """
        # this computes the probability matrix but using the training dual effect rows instead of the actual POVM effects
        return probability_matrix_from_operator_rows(
            self.training_dual_effect_rows,
            self.train_state_rows,
            clip=False,
        )

    @property
    def P_test(self) -> np.ndarray | None:
        if self.test_states is None:
            return None
        return self.povm.probability_matrix(self.test_states)

    @property
    def dual_P_test(self) -> np.ndarray | None:
        if self.test_states is None:
            return None
        return probability_matrix_from_operator_rows(
            self.training_dual_effect_rows,
            self.test_states.state_rows,
            clip=False,
        )

    def haar_test_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.povm.haar_probability_moments()

    def dual_haar_test_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return haar_moments_from_operator_rows(
            self.training_dual_effect_rows,
            dim=self.povm.dim,
        )


def probability_matrix_from_isometry_state_vectors(
    isometry: np.ndarray,
    state_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute P[a, i] = |(V psi_i)[a]|^2 from an isometry and state columns.

    Parameters
    ----------
    isometry:
        Array with shape (nout, dim).
    state_vectors:
        Array with shape (dim, ntr), whose columns are state vectors.
    """
    V = np.asarray(isometry, dtype=complex)
    psi = np.asarray(state_vectors, dtype=complex)

    if V.ndim != 2:
        raise ValueError("isometry must be a 2D array.")
    if psi.ndim != 2:
        raise ValueError("state_vectors must be a 2D array.")
    if V.shape[1] != psi.shape[0]:
        raise ValueError("isometry and state vectors have incompatible dimensions.")

    P = np.abs(V @ psi) ** 2
    P = np.maximum(P.real, 0.0)
    P /= P.sum(axis=0, keepdims=True)
    return P


def haar_probability_moments_from_isometry(
    isometry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact first and second moments of POVM probabilities over Haar pure states.

    For effects mu_a = |v_a><v_a| from the rows of an isometry V,
        E[p_a] = Tr(mu_a) / d
        E[p_a p_b] = (Tr(mu_a) Tr(mu_b) + Tr(mu_a mu_b)) / (d (d + 1)).

    Returns (mean, second_moment), with second_moment[a, b] = E[p_a p_b].
    """
    V = np.asarray(isometry, dtype=complex)
    if V.ndim != 2:
        raise ValueError("isometry must be a 2D array.")

    nout, dim = V.shape
    if dim <= 0 or nout <= 0:
        raise ValueError("isometry must have positive shape.")

    row_traces = np.sum(np.abs(V) ** 2, axis=1).real
    gram = V @ V.conj().T
    trace_products = np.abs(gram) ** 2

    mean = row_traces / dim
    second = (np.outer(row_traces, row_traces) + trace_products) / (dim * (dim + 1))
    return mean, second
