"""Structured QELM training API and actual training simulations.

This module owns the user-facing QELM objects: declarative specs, resolved
training/test/target arrays, deterministic leading-error formulas, and Monte
Carlo estimates of the actual noisy least-squares fit.  It builds exact
training probability matrices from POVMs and states, then samples noisy design
matrices ``P_hat`` through :func:`qelm.noise.noisy_probability_matrix`.

The lower-level ``qelm.trials`` module studies projected scaled noise
``Xi = sqrt(N) * (P_hat - P)`` for a fixed matrix and block decomposition.  It
is a proof-diagnostic layer, not the high-level training API.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from os import PathLike
from typing import ClassVar, Sequence

import numpy as np

from .blocks import schur_covariance_blocks, svd_probability_blocks
from .linalg import opnorm, psd_solve
from .noise import noisy_probability_matrix
from .quantum import (
    POVM,
    QELMQuantumDataset,
    QuantumStateBatch,
    _operator_row_inner_products,
    generate_haar_random_kets,
    get_rng,
    haar_moments_from_operator_rows,
)


RANDOM_POVM_KIND = "random_rank1"
EXPLICIT_POVM_KIND = "effects"
HAAR_PURE_TRAIN_STATES_KIND = "haar_pure"
HAAR_PURE_AVERAGE_SELECTOR = "haar_pure_average"
TEST_STATE_SELECTORS = {
    HAAR_PURE_AVERAGE_SELECTOR,
    "haar_sample",
    "training_mean",
    "training_subset",
    "training_column",
}
TILDE_U_DEFAULT_HAAR_SAMPLE_POINTS = 64
TILDE_U_TEST_STATE_ERROR = (
    "test_state must be None, one of 'haar_pure_average', 'haar_sample', "
    "'training_mean', 'training_subset', or 'training_column', "
    "a tuple like ('haar_sample', num_points), a dictionary like "
    "{'kind': 'training_subset', 'num_points': num_points}, "
    "or an explicit state vector/density matrix."
)

QELM_TRAIN_STATES_ERROR = (
    "train_states is required. Pass 'haar_pure' in an ntr sweep, "
    "{'kind': 'haar_pure', 'num_states': ntr} "
    "to sample Haar-random pure training states, a QuantumStateBatch, "
    "a density-matrix batch with shape (ntr, d, d), or a dictionary such "
    "as {'kind': 'state_vectors', 'vectors': psi, 'axis': 'columns'}."
)
QELM_HAAR_TRAIN_STATES_COUNT_ERROR = (
    "Haar pure training-state specs require 'num_states'. Use 'haar_pure' only in an "
    "ntr sweep study, use {'kind': 'haar_pure'} only in an ntr sweep study, or "
    "pass {'kind': 'haar_pure', 'num_states': ntr}."
)
_TRAIN_STATES_MISSING = object()


def _reject_alias_keys(mapping: dict, aliases: set[str], *, context: str) -> None:
    used_aliases = aliases.intersection(mapping)
    if used_aliases:
        names = ", ".join(sorted(used_aliases))
        raise ValueError(f"{context} uses unsupported alias key(s): {names}.")


def _infer_explicit_povm_nout(povm: object) -> int | None:
    if isinstance(povm, POVM):
        return povm.nout
    if povm is None or isinstance(povm, str):
        return None

    effects = povm
    if isinstance(povm, dict):
        if "effects" not in povm:
            return None
        effects = povm["effects"]

    shape = np.asarray(effects).shape
    if len(shape) == 0:
        return None
    return int(shape[0])


@dataclass(frozen=True)
class QELMDataSpec:
    """User-level data request for one QELM run.

    This records the Hilbert-space dimension, POVM source, number of outcomes,
    and training-state source. It is intentionally declarative: downstream code
    turns it into a concrete POVM and a concrete `QuantumStateBatch` when
    building `QELMTrainingContext`.
    """
    d: int
    nout: int | None = None
    povm: object = None
    train_states: object = field(default=_TRAIN_STATES_MISSING, kw_only=True)

    def __post_init__(self) -> None:
        if self.train_states is _TRAIN_STATES_MISSING:
            raise TypeError(QELM_TRAIN_STATES_ERROR)
        if self.nout is not None:
            object.__setattr__(self, "nout", int(self.nout))
            return

        inferred_nout = _infer_explicit_povm_nout(self.povm)
        if inferred_nout is None:
            raise ValueError("nout is required unless povm is an explicit POVM.")
        object.__setattr__(self, "nout", inferred_nout)


@dataclass(frozen=True)
class QELMTargetRequest:
    """User-level target request.

    `observable` may be an explicit operator/state-like target or an averaging
    selector such as a Haar target average. `resolve_qelm_target` converts this
    into outcome weights or a target second moment after the POVM dual frame and
    training context are known.
    """
    observable: object = None
    normalization: str = "none"


@dataclass(frozen=True)
class QELMTestRequest:
    """User-level test distribution request.

    The request can name an exact average, ask for sampled Haar states, select
    training columns, or provide an explicit test state. `resolve_qelm_test`
    converts it into test probability columns or second moments.
    """
    state: object = None


@dataclass(frozen=True)
class QELMNoiseSpec:
    """Finite-shot noise and numerical fitting settings for actual training.

    `N` and `noise` define how noisy feature matrices are sampled. `N` may be
    omitted when a higher-level study supplies it through an `N` sweep. The
    `actual_noise_trials` count controls Monte Carlo repeats for direct
    training simulations; `lstsq_rcond` is passed to the least-squares fit.
    """
    N: int | None = None
    noise: str = "multinomial"
    actual_noise_trials: int = 200
    lstsq_rcond: float | None = None


@dataclass(frozen=True)
class QELMNumericsSpec:
    """Numerical choices for rank truncation and stable linear solves."""
    rank: int | None = None
    rcond: float = 1e-12
    ridge: float = 0.0


@dataclass(frozen=True)
class QELMTrainingSpec:
    """Complete declarative input for one QELM analysis.

    This is the public object passed into high-level routines. The logic flow is:
    `QELMTrainingSpec` -> `make_qelm_training_context` for concrete matrices,
    then `resolve_qelm_target` and `resolve_qelm_test` for array-level target
    and test objects, then actual/leading error computations.
    """
    data: QELMDataSpec
    target: QELMTargetRequest = field(default_factory=QELMTargetRequest)
    noise: QELMNoiseSpec = field(default_factory=lambda: QELMNoiseSpec(N=1000))
    test: QELMTestRequest = field(default_factory=QELMTestRequest)
    numerics: QELMNumericsSpec = field(default_factory=QELMNumericsSpec)


@dataclass(frozen=True)
class QELMTrainingContext:
    """Resolved quantum objects plus lazy pairwise arrays for QELM training.

    Object-intrinsic representations such as POVM effect rows and state rows
    live on their owning quantum objects. This context owns only quantities
    that combine those objects, such as training/test probability matrices.
    """
    povm: POVM
    train_states: QuantumStateBatch
    test_states: QuantumStateBatch | None = None
    rcond: float | None = None
    _P_train_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _dual_effect_rows_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _dual_P_train_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _test_second_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _dual_test_second_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _P_test_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _dual_P_test_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.povm, POVM):
            raise TypeError("povm must be a POVM object.")
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
    def from_context(cls, context) -> "QELMTrainingContext":
        return cls(
            povm=context.povm,
            train_states=context.train_states,
            test_states=context.test_states,
            rcond=getattr(context, "rcond", None),
        )

    @property
    def povm_dual_effect_rows(self) -> np.ndarray:
        return self.povm.dual_effect_rows(self.rcond)

    @property
    def P_train(self) -> np.ndarray:
        if self._P_train_cache is None:
            P = self.povm.probability_matrix(self.train_states)
            P /= P.sum(axis=0, keepdims=True)
            object.__setattr__(self, "_P_train_cache", P)
        return self._P_train_cache

    @property
    def dual_effect_rows(self) -> np.ndarray:
        if self._dual_effect_rows_cache is None:
            rows = (self.train_states._frame_pinv(self.rcond) @ self.povm_dual_effect_rows.T).T
            object.__setattr__(self, "_dual_effect_rows_cache", rows)
        return self._dual_effect_rows_cache

    @property
    def dual_P_train(self) -> np.ndarray:
        if self._dual_P_train_cache is None:
            P = _operator_row_inner_products(
                self.dual_effect_rows,
                self.train_states._state_rows,
                clip=False,
            )
            object.__setattr__(self, "_dual_P_train_cache", P)
        return self._dual_P_train_cache

    @property
    def test_second(self) -> np.ndarray:
        if self._test_second_cache is None:
            _, second = self.povm.haar_probability_moments()
            object.__setattr__(self, "_test_second_cache", second)
        return self._test_second_cache

    @property
    def dual_test_second(self) -> np.ndarray:
        if self._dual_test_second_cache is None:
            _, second = haar_moments_from_operator_rows(
                self.dual_effect_rows,
                dim=self.povm.dim,
            )
            object.__setattr__(self, "_dual_test_second_cache", second)
        return self._dual_test_second_cache

    @property
    def P_test(self) -> np.ndarray | None:
        if self.test_states is None:
            return None
        if self._P_test_cache is None:
            P = self.povm.probability_matrix(self.test_states)
            P /= P.sum(axis=0, keepdims=True)
            object.__setattr__(self, "_P_test_cache", P)
        return self._P_test_cache

    @property
    def dual_P_test(self) -> np.ndarray | None:
        if self.test_states is None:
            return None
        if self._dual_P_test_cache is None:
            P = _operator_row_inner_products(
                self.dual_effect_rows,
                self.test_states._state_rows,
                clip=False,
            )
            object.__setattr__(self, "_dual_P_test_cache", P)
        return self._dual_P_test_cache

    def __getitem__(self, key: str):
        if not hasattr(self, key):
            raise KeyError(key)
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value


@dataclass(frozen=True)
class ResolvedTest:
    """Array-level test object used by error formulas.

    Exactly one representation is stored: explicit probability columns or an
    exact/sample second moment. Dual-space counterparts are kept with the same
    representation because the leading-bias formulas need both primal and dual
    test quantities.
    """
    mode: str
    average: str
    num_points: int
    probabilities: np.ndarray | None = None
    second_moment: np.ndarray | None = None
    dual_probabilities: np.ndarray | None = None
    dual_second_moment: np.ndarray | None = None
    # Raw density operators for column-based finite test sets.
    states: np.ndarray | None = None

    def __post_init__(self) -> None:
        has_columns = self.probabilities is not None
        has_moments = self.second_moment is not None
        if has_columns == has_moments:
            raise ValueError("ResolvedTest must have exactly one of probabilities or second_moment.")
        if has_columns and self.dual_probabilities is None:
            raise ValueError("Column-based ResolvedTest requires dual_probabilities.")
        if has_moments and self.dual_second_moment is None:
            raise ValueError("Moment-based ResolvedTest requires dual_second_moment.")

    @classmethod
    def from_test(cls, test) -> "ResolvedTest":
        return cls(
            mode=test.mode,
            average=test.average,
            num_points=test.num_points,
            probabilities=test.probabilities,
            second_moment=test.second_moment,
            dual_probabilities=test.dual_probabilities,
            dual_second_moment=test.dual_second_moment,
            states=getattr(test, "states", None),
        )


@dataclass(frozen=True)
class ResolvedTarget:
    """Array-level target object used by training and leading-error formulas.

    Fixed targets keep the representation supplied by the caller: explicit
    outcome weights remain weights, while operator/state targets remain
    Hilbert-space operators. Averaged targets are stored as a second-moment
    matrix over outcome weights because those formulas are explicitly
    weight-space formulas.
    """
    mode: str
    kind: str
    average: str
    normalization: str
    scale: float
    weights: np.ndarray | None = None
    second_moment: np.ndarray | None = None
    # Hilbert-space operator corresponding to the normalized weights, when finite.
    operator: np.ndarray | None = None
    # Hilbert-space operator before target normalization, when finite.
    raw_operator: np.ndarray | None = None

    @classmethod
    def fixed(
        cls,
        *,
        kind: str,
        normalization: str,
        scale: float,
        weights: np.ndarray | None = None,
        operator: np.ndarray | None = None,
        raw_operator: np.ndarray | None = None,
    ) -> "ResolvedTarget":
        return cls(
            mode="fixed",
            kind=kind,
            average="single_observable",
            normalization=normalization,
            scale=scale,
            weights=weights,
            operator=operator,
            raw_operator=raw_operator,
        )

    @classmethod
    def average_over(
        cls,
        *,
        kind: str,
        average: str,
        normalization: str,
        scale: float,
        second_moment: np.ndarray,
    ) -> "ResolvedTarget":
        return cls(
            mode="average",
            kind=kind,
            average=average,
            normalization=normalization,
            scale=scale,
            second_moment=second_moment,
        )

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "average"}:
            raise ValueError("ResolvedTarget mode must be 'fixed' or 'average'.")
        if self.is_average:
            if self.second_moment is None or self.weights is not None:
                raise ValueError("Averaged ResolvedTarget requires second_moment and no weights.")
            if self.operator is not None or self.raw_operator is not None:
                raise ValueError("Averaged ResolvedTarget cannot carry a fixed operator.")
        elif self.second_moment is not None:
            raise ValueError("Fixed ResolvedTarget cannot carry second_moment.")
        elif self.weights is None and self.operator is None:
            raise ValueError("Fixed ResolvedTarget requires weights or an operator.")

    @classmethod
    def from_target(cls, target) -> "ResolvedTarget":
        return cls(
            mode=target.mode,
            kind=target.kind,
            average=target.average,
            normalization=target.normalization,
            scale=target.scale,
            weights=target.weights,
            second_moment=target.second_moment,
            operator=getattr(target, "operator", None),
            raw_operator=getattr(target, "raw_operator", None),
        )

    @property
    def is_average(self) -> bool:
        return self.mode == "average"


@dataclass(frozen=True)
class QELMTrainingResults:
    """Monte Carlo estimate of noisy finite-shot QELM training error.

    The scalar fields are the reported metrics. Optional arrays expose fitted
    weights or fit matrices for diagnostics when requested by the caller.
    """
    mse: float
    bias_sq: float
    variance: float
    abs_bias_mean: float
    noise_trials: int
    mean_weights: np.ndarray | None = None
    mean_fit_matrix: np.ndarray | None = None
    fitted_weights: np.ndarray | None = None
    fit_matrices: np.ndarray | None = None

    def to_metrics_dict(self) -> dict:
        return {
            "actual_mse": self.mse,
            "actual_bias_sq": self.bias_sq,
            "actual_variance": self.variance,
            "actual_abs_bias_mean": self.abs_bias_mean,
            "actual_noise_trials": self.noise_trials,
        }

    def __getitem__(self, key: str):
        return self.to_metrics_dict()[key]


@dataclass(frozen=True)
class QELMLeadingErrorResult:
    """Leading-order bias/variance prediction for a resolved training problem.

    `corrected` records whether the tilde-U/C22 correction was used or replaced
    by the identity approximation. The remaining fields are summary metrics
    produced by the analytic leading-error formulas.
    """
    corrected: bool
    bias_sq: float
    variance: float
    mse: float
    bias_abs_mean: float = np.nan
    bias_sq_max: float = np.nan
    variance_max: float = np.nan

    @classmethod
    def from_metrics(cls, metrics: dict, *, corrected: bool) -> "QELMLeadingErrorResult":
        return cls(
            corrected=corrected,
            bias_sq=metrics["bias_sq"],
            variance=metrics["variance"],
            mse=metrics["mse"],
            bias_abs_mean=metrics.get("bias_abs_mean", np.nan),
            bias_sq_max=metrics.get("bias_sq_max", np.nan),
            variance_max=metrics.get("variance_max", np.nan),
        )

    def to_metrics_dict(self) -> dict:
        return {
            "bias_sq": self.bias_sq,
            "variance": self.variance,
            "mse": self.mse,
            "bias_abs_mean": self.bias_abs_mean,
            "bias_sq_max": self.bias_sq_max,
            "variance_max": self.variance_max,
        }

    def __getitem__(self, key: str):
        return self.to_metrics_dict()[key]


@dataclass(frozen=True)
class TildeUDiagnostics(Mapping[str, object]):
    """Deterministic tilde-U/Schur-correction diagnostics.

    The object is mapping-compatible to preserve the older
    ``diag["C22_inv_C21"]`` API while also exposing typed attributes and common
    transform helpers.
    """
    blocks: dict
    C12: np.ndarray
    C22: np.ndarray
    C22_inv_C21: np.ndarray
    correction_matrix: np.ndarray
    C22_inv_C21_op: float
    correction_op: float
    correction_op_relative_difference: float
    C22_lambda_min: float
    C22_lambda_max: float
    C22_cond: float
    C22_kept_rank: int

    _KEYS: ClassVar[tuple[str, ...]] = (
        "blocks",
        "C12",
        "C22",
        "C22_inv_C21",
        "correction_matrix",
        "C22_inv_C21_op",
        "correction_op",
        "correction_op_relative_difference",
        "C22_lambda_min",
        "C22_lambda_max",
        "C22_cond",
        "C22_kept_rank",
    )

    def __getitem__(self, key: str):
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self._KEYS}

    @property
    def U1(self) -> np.ndarray:
        return self.blocks["U1"]

    @property
    def U2(self) -> np.ndarray:
        return self.blocks["U2"]

    @property
    def singular_values(self) -> np.ndarray:
        return self.blocks["singular_values"]

    def training_transform_matrix(self, *, approximate_identity: bool) -> np.ndarray:
        return _c22_training_transform_matrix(
            self.U1.shape[0],
            self.U1,
            self.U2,
            self.C22_inv_C21,
            approximate_identity=approximate_identity,
        )

    def bias_transform_matrix(self, *, approximate_identity: bool) -> np.ndarray:
        return _bias_test_transform_matrix(
            self.U1.shape[0],
            self.U1,
            self.U2,
            self.C22_inv_C21,
            approximate_identity=approximate_identity,
        )


@dataclass(frozen=True)
class QELMTrainingAnalysisResult:
    """Bundle returned by `analyze_qelm_training`.

    It keeps the original spec, the concrete context, resolved target/test
    objects, deterministic diagnostics, and any requested actual or leading
    error results. This is the main end-to-end result object for one run.
    """
    spec: QELMTrainingSpec
    context: QELMTrainingContext
    target: ResolvedTarget
    test: ResolvedTest
    diagnostics: TildeUDiagnostics
    mse: QELMTrainingResults | None = None
    leading_corrected: QELMLeadingErrorResult | None = None
    leading_identity: QELMLeadingErrorResult | None = None


@dataclass(frozen=True)
class TildeUTrainingApproxStudySpec:
    """Declarative sweep specification for tilde-U approximation studies.

    `base` is a single `QELMTrainingSpec`; `sweep_col` and `sweep_values`
    generate concrete specs, for example by injecting `num_states` into a
    flexible Haar training-state request for an `ntr` sweep. The remaining
    fields control repetitions, summaries, slopes, plotting, and failure mode.
    `plots` accepts "all" or short keys such as "mse", "leading_mse",
    "correction", "bias", "variance", "mse_ratio", "actual_ratio", and
    "relative_error". If `output_file` is set, `run_tilde_u_training_approx_report`
    writes a compact .zip report containing a raw Parquet table and JSON metadata.
    Existing report files are not overwritten unless `overwrite=True`; by
    default, the report writer appends a numeric suffix to the output filename.
    """
    base: QELMTrainingSpec
    sweep_col: str = "ntr"
    sweep_values: Sequence[int] | None = None
    repetitions: int = 10
    seed: int | None = None
    quantiles: Sequence[float] = (0.25, 0.75)
    quantile_band: tuple[float, float] = (0.25, 0.75)
    x_col: str | None = None
    slope_ycols: Sequence[str] = (
        "C22_inv_C21_op_median",
        "correction_op_median",
        "leading_mse_exact_median",
        "leading_mse_identity_median",
        "actual_mse_median",
    )
    show_summary: bool = True
    show_slopes: bool = True
    make_plots: bool = True
    plots: Sequence[str] | str | None = ("all",)
    verbose: bool = True
    fail_soft: bool = False
    output_file: str | PathLike[str] | None = None
    overwrite: bool = False


@dataclass
class QELMRun:
    """Lazy object for one concrete QELM training run.

    Resolved context, test distribution, deterministic diagnostics, and target
    are computed once and cached. The module-level wrapper functions below are
    intentionally thin frontends over this object.
    """
    spec: QELMTrainingSpec
    rng: np.random.Generator | int | None = None

    _context: QELMTrainingContext | None = field(default=None, init=False, repr=False)
    _test: ResolvedTest | None = field(default=None, init=False, repr=False)
    _diagnostics: TildeUDiagnostics | None = field(default=None, init=False, repr=False)
    _target: ResolvedTarget | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = get_rng(self.rng)

    @classmethod
    def from_context(
        cls,
        spec: QELMTrainingSpec,
        context: QELMTrainingContext,
        rng: np.random.Generator | int | None = None,
    ) -> "QELMRun":
        run = cls(spec, rng=rng)
        run._context = context
        return run

    @property
    def context(self) -> QELMTrainingContext:
        if self._context is None:
            self._context = make_qelm_training_context(self.spec, self.rng)
        return self._context

    @property
    def test(self) -> ResolvedTest:
        if self._test is None:
            self._test = resolve_qelm_test(self.spec, self.context, self.rng)
        return self._test

    @property
    def diagnostics(self) -> TildeUDiagnostics:
        if self._diagnostics is None:
            self._diagnostics = compute_qelm_diagnostics(self.spec, self.context)
        return self._diagnostics

    @property
    def target(self) -> ResolvedTarget:
        if self._target is None:
            self._target = resolve_qelm_target(
                self.spec,
                self.context,
                rng=self.rng,
            )
        return self._target

    def leading_error(self, *, corrected: bool = True) -> QELMLeadingErrorResult:
        return _compute_qelm_leading_error_resolved(
            spec=self.spec,
            context=self.context,
            target=self.target,
            test=self.test,
            diagnostics=self.diagnostics,
            corrected=corrected,
        )

    def train_model(
        self,
        *,
        return_fits: bool = False,
        return_fit_matrix: bool = False,
    ) -> QELMTrainingResults:
        return _run_qelm_training_resolved(
            spec=self.spec,
            rng=self.rng,
            context=self.context,
            target=self.target,
            test=self.test,
            return_fits=return_fits,
            return_fit_matrix=return_fit_matrix,
        )

    def analyze(
        self,
        *,
        compute_mse: bool = True,
        include_leading: bool = True,
        return_fits: bool = False,
        return_fit_matrix: bool = False,
    ) -> QELMTrainingAnalysisResult:
        mse = None
        if compute_mse:
            mse = self.train_model(
                return_fits=return_fits,
                return_fit_matrix=return_fit_matrix,
            )

        leading_corrected = None
        leading_identity = None
        if include_leading:
            leading_corrected = self.leading_error(corrected=True)
            leading_identity = self.leading_error(corrected=False)

        return QELMTrainingAnalysisResult(
            spec=self.spec,
            context=self.context,
            target=self.target,
            test=self.test,
            diagnostics=self.diagnostics,
            mse=mse,
            leading_corrected=leading_corrected,
            leading_identity=leading_identity,
        )


def _rank_from_spec(spec: QELMTrainingSpec) -> int:
    return spec.data.d * spec.data.d if spec.numerics.rank is None else int(spec.numerics.rank)


def _required_noise_N(noise: QELMNoiseSpec) -> int:
    if noise.N is None:
        raise ValueError(
            "noise.N is required for this operation. Provide QELMNoiseSpec(N=...) "
            "or use an N sweep with explicit sweep_values."
        )
    if int(noise.N) != noise.N or int(noise.N) <= 0:
        raise ValueError("noise.N must be a positive integer.")
    return int(noise.N)


def _povm_kind_from_spec(povm) -> str:
    if povm is None:
        return RANDOM_POVM_KIND

    if isinstance(povm, str):
        kind = povm.lower()
        if kind != RANDOM_POVM_KIND:
            raise ValueError("Only 'random_rank1' is supported as a string POVM spec.")
        return RANDOM_POVM_KIND

    if isinstance(povm, dict):
        _reject_alias_keys(
            povm,
            {"type", "povm_effects", "num_outcomes", "d"},
            context="POVM spec",
        )
        if "effects" in povm:
            return EXPLICIT_POVM_KIND

        kind = str(povm.get("kind", RANDOM_POVM_KIND)).lower()
        if kind == RANDOM_POVM_KIND:
            return RANDOM_POVM_KIND
        if kind == EXPLICIT_POVM_KIND:
            return EXPLICIT_POVM_KIND
        raise ValueError(f"Unknown POVM spec kind: {kind!r}.")

    if isinstance(povm, POVM):
        return EXPLICIT_POVM_KIND

    return EXPLICIT_POVM_KIND


def _povm_shape_from_spec(povm) -> tuple[int | None, int | None]:
    kind = _povm_kind_from_spec(povm)

    if kind == EXPLICIT_POVM_KIND:
        if isinstance(povm, POVM):
            return povm.nout, povm.dim
        effects = povm
        if isinstance(povm, dict):
            effects = povm.get("effects")
            if effects is None:
                raise ValueError("Explicit POVM dictionary specs must include 'effects'.")
        povm_effects = POVM.from_effects(effects)
        return povm_effects.nout, povm_effects.dim

    if isinstance(povm, dict):
        nout = povm.get("nout")
        dim = povm.get("dim")
        return (
            None if nout is None else int(nout),
            None if dim is None else int(dim),
        )

    return None, None


def _validate_random_povm_spec_for_config(
    povm,
    *,
    nout: int,
    dim: int,
) -> None:
    if not isinstance(povm, dict):
        return

    _reject_alias_keys(
        povm,
        {"type", "povm_effects", "num_outcomes", "d"},
        context="POVM spec",
    )
    spec_nout = povm.get("nout")
    spec_dim = povm.get("dim")
    if spec_nout is not None and int(spec_nout) != int(nout):
        raise ValueError(
            f"Random POVM spec has nout={spec_nout}, but the config has nout={nout}."
        )
    if spec_dim is not None and int(spec_dim) != int(dim):
        raise ValueError(
            f"Random POVM spec has d={spec_dim}, but the config has d={dim}."
        )


def _povm_from_spec(povm, *, nout: int, dim: int, rng: np.random.Generator) -> POVM:
    kind = _povm_kind_from_spec(povm)
    if kind == RANDOM_POVM_KIND:
        _validate_random_povm_spec_for_config(povm, nout=nout, dim=dim)
        return POVM.random_rank1(nout=nout, dim=dim, rng=rng)

    if isinstance(povm, POVM):
        if povm.nout != int(nout):
            raise ValueError(f"Explicit POVM has nout={povm.nout}, but the config has nout={nout}.")
        if povm.dim != int(dim):
            raise ValueError(f"Explicit POVM has dimension d={povm.dim}, but the config has d={dim}.")
        return povm

    effects = povm
    if isinstance(povm, dict):
        effects = povm.get("effects")
    return POVM.from_effects(effects, dim=dim, nout=nout)


def _context_from_dataset(dataset: QELMQuantumDataset) -> QELMTrainingContext:
    return QELMTrainingContext(
        povm=dataset.povm,
        train_states=dataset.train_states,
        test_states=dataset.test_states,
        rcond=dataset.rcond,
    )


def _validate_training_state_batch(
    batch: QuantumStateBatch,
    *,
    dim: int,
) -> QuantumStateBatch:
    if batch.dim != int(dim):
        raise ValueError(
            f"Training states have dimension d={batch.dim}, but the config has d={dim}."
        )
    return batch


def _training_states_from_spec(
    train_states,
    *,
    dim: int,
    rng: np.random.Generator,
) -> QuantumStateBatch:
    if train_states is _TRAIN_STATES_MISSING:
        raise TypeError(QELM_TRAIN_STATES_ERROR)

    if isinstance(train_states, QuantumStateBatch):
        return _validate_training_state_batch(train_states, dim=dim)

    if isinstance(train_states, str):
        if train_states.lower() == HAAR_PURE_TRAIN_STATES_KIND:
            train_states = {"kind": HAAR_PURE_TRAIN_STATES_KIND}
        else:
            raise ValueError(QELM_TRAIN_STATES_ERROR)

    if isinstance(train_states, dict):
        _reject_alias_keys(
            train_states,
            {"mode", "d", "state_vectors", "density_matrices", "rhos"},
            context="train_states spec",
        )
        if "kind" in train_states:
            kind = str(train_states["kind"]).lower()
        elif "vectors" in train_states:
            kind = "state_vectors"
        else:
            kind = "states"
        if kind == HAAR_PURE_TRAIN_STATES_KIND:
            if "num_states" not in train_states:
                raise ValueError(QELM_HAAR_TRAIN_STATES_COUNT_ERROR)
            requested_dim = int(train_states.get("dim", dim))
            if requested_dim != int(dim):
                raise ValueError(
                    f"Haar training states request d={requested_dim}, but the config has d={dim}."
                )
            batch = QuantumStateBatch.haar_pure_from_columns(
                num_states=int(train_states["num_states"]),
                dim=int(dim),
                rng=rng,
            )
            return _validate_training_state_batch(batch, dim=dim)

        if kind == "state_vectors":
            vectors = train_states.get("vectors")
            if vectors is None:
                raise ValueError(QELM_TRAIN_STATES_ERROR)
            batch = QuantumStateBatch.from_state_vectors(
                vectors,
                dim=int(dim),
                axis=str(train_states.get("axis", "auto")),
                name="train_states",
            )
            return _validate_training_state_batch(batch, dim=dim)

        if kind != "states":
            raise ValueError(QELM_TRAIN_STATES_ERROR)
        states = train_states.get("states")
        if states is None:
            raise ValueError(QELM_TRAIN_STATES_ERROR)
        batch = QuantumStateBatch.from_state_like(states, dim=int(dim), name="train_states")
        return _validate_training_state_batch(batch, dim=dim)

    batch = QuantumStateBatch.from_state_like(train_states, dim=int(dim), name="train_states")
    return _validate_training_state_batch(batch, dim=dim)


def _training_state_count_from_spec(train_states) -> int | None:
    if train_states is _TRAIN_STATES_MISSING:
        raise TypeError(QELM_TRAIN_STATES_ERROR)
    if isinstance(train_states, QuantumStateBatch):
        return int(train_states.num_states)
    if isinstance(train_states, dict):
        _reject_alias_keys(
            train_states,
            {"mode", "d", "state_vectors", "density_matrices", "rhos"},
            context="train_states spec",
        )
        if "kind" in train_states:
            kind = str(train_states["kind"]).lower()
        elif "vectors" in train_states:
            kind = "state_vectors"
        else:
            kind = "states"
        if kind == HAAR_PURE_TRAIN_STATES_KIND:
            if "num_states" in train_states:
                return int(train_states["num_states"])
            return None
        if kind in {"state_vectors", "states"}:
            return None
        raise ValueError(QELM_TRAIN_STATES_ERROR)
    return None


def _with_training_num_states(train_states, num_states: int):
    if train_states is _TRAIN_STATES_MISSING:
        raise TypeError(QELM_TRAIN_STATES_ERROR)
    count = int(num_states)
    if count <= 0:
        raise ValueError("ntr sweep values must be positive.")
    if isinstance(train_states, str):
        if train_states.lower() != HAAR_PURE_TRAIN_STATES_KIND:
            raise ValueError("ntr sweeps can only update Haar train_states specs.")
        train_states = {"kind": train_states.lower()}
    if not isinstance(train_states, dict):
        raise ValueError("ntr sweeps require a Haar train_states spec.")
    _reject_alias_keys(
        train_states,
        {"mode", "d", "state_vectors", "density_matrices", "rhos"},
        context="train_states spec",
    )
    updated = dict(train_states)
    updated["num_states"] = count
    if str(updated.get("kind", "states")).lower() != HAAR_PURE_TRAIN_STATES_KIND:
        raise ValueError("ntr sweeps can only update Haar train_states specs.")
    return updated


def _test_state_point_count(value, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    count = int(value)
    if count <= 0:
        raise ValueError("test_state num_points must be positive.")
    return count


def _test_state_selector_from_string(value: str) -> str:
    key = value.lower()
    if key == "fixed_state":
        raise ValueError("test_state='fixed_state' is ambiguous; pass the state vector or density matrix instead.")
    if key not in TEST_STATE_SELECTORS:
        raise ValueError(TILDE_U_TEST_STATE_ERROR)
    return key


def _default_test_state_point_count(selector: str) -> int | None:
    if selector == "haar_sample":
        return TILDE_U_DEFAULT_HAAR_SAMPLE_POINTS
    return None


def _resolve_test_state_request(
    test_state,
) -> tuple[str, object | None, int | None]:
    if test_state is None:
        return HAAR_PURE_AVERAGE_SELECTOR, None, None

    if isinstance(test_state, str):
        selector = _test_state_selector_from_string(test_state)
        return selector, None, _default_test_state_point_count(selector)

    if isinstance(test_state, tuple) and test_state and isinstance(test_state[0], str):
        if test_state[0].lower() == "fixed_state":
            if len(test_state) != 2:
                raise ValueError("test_state=('fixed_state', state) requires a state vector or density matrix.")
            return "fixed_state", test_state[1], None
        selector = _test_state_selector_from_string(test_state[0])
        if len(test_state) > 2:
            raise ValueError("test_state tuple selectors must be (kind,) or (kind, num_points).")
        count = test_state[1] if len(test_state) == 2 else None
        return selector, None, _test_state_point_count(
            count,
            default=_default_test_state_point_count(selector),
        )

    if isinstance(test_state, dict):
        _reject_alias_keys(
            test_state,
            {"mode", "type", "value", "points", "count", "sample_size"},
            context="test_state spec",
        )
        kind = test_state.get("kind")
        state = test_state.get("state")
        if kind is None:
            if state is None:
                raise ValueError(TILDE_U_TEST_STATE_ERROR)
            return "fixed_state", state, None
        if str(kind).lower() == "fixed_state":
            if state is None:
                raise ValueError("test_state fixed-state dictionaries require a 'state' value.")
            return "fixed_state", state, None
        selector = _test_state_selector_from_string(str(kind))
        count = test_state.get("num_points")
        return selector, None, _test_state_point_count(
            count,
            default=_default_test_state_point_count(selector),
        )

    return "fixed_state", test_state, None


def make_qelm_training_context(
    spec: QELMTrainingSpec,
    rng: np.random.Generator | int | None = None,
) -> QELMTrainingContext:
    rng = get_rng(rng)
    povm = _povm_from_spec(
        spec.data.povm,
        nout=spec.data.nout,
        dim=spec.data.d,
        rng=rng,
    )
    # here we parse the requested training states converting them to actual states
    # (unless they were already in a QuantumStateBatch)
    train_states = _training_states_from_spec(
        spec.data.train_states,
        dim=spec.data.d,
        rng=rng,
    )
    test_selector, _, test_num_points = _resolve_test_state_request(spec.test.state)
    ntest = None
    if test_selector == "haar_sample":
        ntest = train_states.num_states if test_num_points is None else int(test_num_points)

    test_states = None
    if ntest is not None:
        if ntest <= 0:
            raise ValueError("ntest must be positive.")
        test_states = QuantumStateBatch.haar_pure_from_columns(
            num_states=ntest,
            dim=spec.data.d,
            rng=rng,
        )
    return QELMTrainingContext(
        povm=povm,
        train_states=train_states,
        test_states=test_states,
        rcond=spec.numerics.rcond,
    )


def resolve_qelm_test(
    spec: QELMTrainingSpec,
    context: QELMTrainingContext,
    rng: np.random.Generator | int | None = None,
) -> ResolvedTest:
    rng = get_rng(rng)
    test_selector, fixed_test_state, test_num_points = _resolve_test_state_request(spec.test.state)

    if test_selector == HAAR_PURE_AVERAGE_SELECTOR:
        return ResolvedTest(
            mode=test_selector,
            average="exact_haar_second_moment",
            num_points=0,
            second_moment=context.test_second,
            dual_second_moment=context.dual_test_second,
        )

    if test_selector == "haar_sample":
        P_test = context.P_test
        dual_P_test = context.dual_P_test
        if P_test is None or dual_P_test is None or context.test_states is None:
            raise ValueError("Haar-sample test resolution requires sampled test states.")
        return ResolvedTest(
            mode=test_selector,
            average="sampled_haar_states",
            num_points=P_test.shape[1],
            probabilities=P_test,
            dual_probabilities=dual_P_test,
            states=context.test_states.states,
        )

    if test_selector == "fixed_state":
        if fixed_test_state is None:
            raise ValueError("Fixed test state resolution requires an explicit state.")
        state_batch = QuantumStateBatch.from_state_like(
            fixed_test_state,
            dim=spec.data.d,
            name="test_state",
        )
        probabilities = context.povm.probability_matrix(state_batch)
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        dual_probabilities = _operator_row_inner_products(
            context.dual_effect_rows,
            state_batch._state_rows,
            clip=False,
        )
        return ResolvedTest(
            mode=test_selector,
            average="fixed_state",
            num_points=1,
            probabilities=probabilities,
            dual_probabilities=dual_probabilities,
            states=state_batch.states,
        )

    P = context.P_train
    ntr = P.shape[1]

    if test_selector == "training_mean":
        return ResolvedTest(
            mode=test_selector,
            average="training_columns",
            num_points=ntr,
            probabilities=P,
            dual_probabilities=context.dual_P_train,
            states=context.train_states.states,
        )

    if test_selector == "training_subset":
        count = ntr if test_num_points is None else min(ntr, int(test_num_points))
        indices = rng.choice(ntr, size=count, replace=False)
        return ResolvedTest(
            mode=test_selector,
            average="training_column_subset",
            num_points=count,
            probabilities=P[:, indices],
            dual_probabilities=context.dual_P_train[:, indices],
            states=context.train_states.states[indices],
        )

    if test_selector == "training_column":
        index = int(rng.integers(0, ntr))
        return ResolvedTest(
            mode=test_selector,
            average="single_training_column",
            num_points=1,
            probabilities=P[:, [index]],
            dual_probabilities=context.dual_P_train[:, [index]],
            states=context.train_states.states[[index]],
        )

    raise ValueError(TILDE_U_TEST_STATE_ERROR)


def _normalize_target_vector(
    w_observable: np.ndarray,
    P: np.ndarray,
    target_normalization: str,
) -> tuple[np.ndarray, float]:
    w_observable = np.asarray(w_observable, dtype=float)
    if target_normalization == "training_rms":
        scale = float(np.sqrt(np.mean((P.T @ w_observable) ** 2)))
    elif target_normalization == "euclidean":
        scale = float(np.linalg.norm(w_observable))
    elif target_normalization == "none":
        scale = 1.0
    else:
        raise ValueError("target_normalization must be 'training_rms', 'euclidean', or 'none'.")

    if scale > 0:
        return w_observable / scale, scale
    return w_observable, scale


def _normalize_target_second_moment(
    target_second_moment: np.ndarray,
    P: np.ndarray,
    target_normalization: str,
) -> tuple[np.ndarray, float]:
    target_second_moment = np.asarray(target_second_moment, dtype=float)
    if target_normalization == "training_rms":
        scale_sq = float(np.trace((P @ P.T / P.shape[1]) @ target_second_moment))
    elif target_normalization == "euclidean":
        scale_sq = float(np.trace(target_second_moment))
    elif target_normalization == "none":
        scale_sq = 1.0
    else:
        raise ValueError("target_normalization must be 'training_rms', 'euclidean', or 'none'.")

    scale = float(np.sqrt(max(scale_sq, 0.0)))
    if scale > 0:
        return target_second_moment / (scale**2), scale
    return target_second_moment, scale


def _operator_to_outcome_weights(
    operator: np.ndarray,
    povm_dual_effect_rows: np.ndarray,
) -> np.ndarray:
    operator = np.asarray(operator, dtype=complex)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("Operator target must have shape (d, d).")
    operator_row = operator.reshape(1, -1)
    weights = _operator_row_inner_products(
        povm_dual_effect_rows,
        operator_row,
        clip=False,
    )[:, 0]
    return weights.real


def _fixed_target_from_weights(
    *,
    kind: str,
    weights: np.ndarray,
    P: np.ndarray,
    target_normalization: str,
    raw_operator: np.ndarray | None = None,
) -> ResolvedTarget:
    weights, scale = _normalize_target_vector(weights, P, target_normalization)
    return ResolvedTarget.fixed(
        kind=kind,
        normalization=target_normalization,
        scale=scale,
        weights=weights,
        operator=None,
        raw_operator=raw_operator,
    )


def _fixed_target_from_operator(
    *,
    kind: str,
    operator: np.ndarray,
    povm_dual_effect_rows: np.ndarray,
    P: np.ndarray,
    target_normalization: str,
) -> ResolvedTarget:
    weights = _operator_to_outcome_weights(operator, povm_dual_effect_rows)
    _, scale = _normalize_target_vector(weights, P, target_normalization)
    normalized_operator = operator
    if scale > 0:
        normalized_operator = np.asarray(operator, dtype=complex) / scale
    return ResolvedTarget.fixed(
        kind=kind,
        normalization=target_normalization,
        scale=scale,
        weights=None,
        operator=normalized_operator,
        raw_operator=operator,
    )


def _average_target_from_second_moment(
    *,
    kind: str,
    average: str,
    second_moment: np.ndarray,
    P: np.ndarray,
    target_normalization: str,
) -> ResolvedTarget:
    second_moment, scale = _normalize_target_second_moment(
        second_moment,
        P,
        target_normalization,
    )
    return ResolvedTarget.average_over(
        kind=kind,
        average=average,
        normalization=target_normalization,
        scale=scale,
        second_moment=second_moment,
    )


def resolve_qelm_target(
    spec: QELMTrainingSpec,
    context: QELMTrainingContext,
    rng: np.random.Generator | int | None = None,
) -> ResolvedTarget:
    rng = get_rng(rng)
    target_observable = spec.target.observable
    target_normalization = spec.target.normalization
    P = context.P_train
    dim = spec.data.d
    nout = P.shape[0]

    if target_observable is None:
        raise ValueError("Target observable is required.")

    if isinstance(target_observable, str):
        key = target_observable.lower()
        if key == HAAR_PURE_AVERAGE_SELECTOR:
            _, target_second = haar_moments_from_operator_rows(
                context.povm_dual_effect_rows,
                dim=dim,
            )
            return _average_target_from_second_moment(
                kind=HAAR_PURE_TRAIN_STATES_KIND,
                average="exact_haar_second_moment",
                second_moment=target_second,
                P=P,
                target_normalization=target_normalization,
            )
        if key == "random_haar_pure_state":
            vector = generate_haar_random_kets(
                num_states=1,
                dim=dim,
                rng=rng,
            )[:, 0]
            operator = np.outer(vector, vector.conj())
            return _fixed_target_from_operator(
                kind="random_haar_pure_state",
                operator=operator,
                povm_dual_effect_rows=context.povm_dual_effect_rows,
                P=P,
                target_normalization=target_normalization,
            )
        raise ValueError(
            "Unknown target_observable string. Use 'random_haar_pure_state' "
            "or 'haar_pure_average'."
        )

    # regardless of how target is provided, it is converted to a vector of outcome weights via the POVM dual effects, and then normalized according to target_normalization.
    # The kind of target is recorded in the ResolvedTarget for later analysis.
    target = np.asarray(target_observable)
    if target.ndim == 1:
        # Interpret 1D targets as either pure states or outcome weights, depending on length.
        # outcome weights mean here a vector w such that the target is the observable O = sum_b w_b mu_b, where mu_b are the POVM effects.
        if target.shape[0] == dim:
            operator = QuantumStateBatch.from_state_like(
                target,
                dim=dim,
                name="target_observable",
            ).states[0]
            return _fixed_target_from_operator(
                kind="pure_state",
                operator=operator,
                povm_dual_effect_rows=context.povm_dual_effect_rows,
                P=P,
                target_normalization=target_normalization,
            )
        if target.shape[0] != nout:
            raise ValueError(
                f"1D target_observable must have length d={dim} for a pure state "
                f"or length nout={nout} for outcome weights."
            )
        return _fixed_target_from_weights(
            kind="outcome_weights",
            weights=target.astype(float),
            P=P,
            target_normalization=target_normalization,
        )

    if target.ndim == 2:
        # interpret 2D targets as operators, which we convert to outcome weights via the POVM dual effects.
        if target.shape != (dim, dim):
            raise ValueError(f"Operator target must have shape ({dim}, {dim}).")
        operator = np.asarray(target, dtype=complex)
        return _fixed_target_from_operator(
            kind="operator",
            operator=operator,
            povm_dual_effect_rows=context.povm_dual_effect_rows,
            P=P,
            target_normalization=target_normalization,
        )

    raise ValueError(
        "target_observable must be a recognized string, a length-nout "
        "weight vector, or a (d, d) operator."
    )


def _c22_covariance_columns(
    P: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    B = np.asarray(B, dtype=float)
    return P.T @ (a[:, None] * B) - (P.T @ a)[:, None] * (P.T @ B)


def _dual_state_probability_columns(
    test_probabilities: np.ndarray,
    U1: np.ndarray,
    singular_values: np.ndarray,
    rcond: float,
) -> np.ndarray:
    s = np.asarray(singular_values[: U1.shape[1]], dtype=float)
    cutoff = rcond * np.max(s) if s.size else 0.0
    inv_s = np.zeros_like(s)
    keep = s > cutoff
    inv_s[keep] = 1.0 / s[keep]
    return U1 @ (inv_s[:, None] * (U1.T @ test_probabilities))


def _dual_state_probability_second_moment(
    test_second_moment: np.ndarray,
    U1: np.ndarray,
    singular_values: np.ndarray,
    rcond: float,
) -> np.ndarray:
    s = np.asarray(singular_values[: U1.shape[1]], dtype=float)
    cutoff = rcond * np.max(s) if s.size else 0.0
    inv_s = np.zeros_like(s)
    keep = s > cutoff
    inv_s[keep] = 1.0 / s[keep]

    core = U1.T @ test_second_moment @ U1
    core = inv_s[:, None] * core * inv_s[None, :]
    return U1 @ core @ U1.T


def _maybe_apply_correction_term(
    x: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    C22_inv_C21: np.ndarray,
    *,
    approximate_identity: bool,
) -> np.ndarray:
    if approximate_identity:
        return np.asarray(x, dtype=float)
    return U1 @ (U1.T @ x) - U2 @ (C22_inv_C21 @ (U1.T @ x))


def _c22_training_transform_matrix(
    nout: int,
    U1: np.ndarray,
    U2: np.ndarray,
    C22_inv_C21: np.ndarray,
    *,
    approximate_identity: bool,
) -> np.ndarray:
    if approximate_identity:
        return np.eye(nout)
    return U1 @ U1.T - U2 @ C22_inv_C21 @ U1.T


def _bias_test_transform_matrix(
    nout: int,
    U1: np.ndarray,
    U2: np.ndarray,
    C22_inv_C21: np.ndarray,
    *,
    approximate_identity: bool,
) -> np.ndarray:
    if approximate_identity:
        return np.eye(nout)
    return U1 @ U1.T - U1 @ C22_inv_C21.T @ U2.T


def _weighted_training_covariance_sum(P: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return np.diag(P @ weights) - (P * weights[None, :]) @ P.T


def tilde_u_correction_operator_diagnostics(
    *,
    P: np.ndarray,
    rank: int,
    rcond: float = 1e-12,
    ridge: float = 0.0,
) -> TildeUDiagnostics:
    blocks = svd_probability_blocks(P, rank=rank)
    U1 = blocks["U1"]
    U2 = blocks["U2"]
    pi2_diag = blocks["Pi2_diag"]

    C12, C22 = schur_covariance_blocks(P, U1, U2, pi2_diag)
    C22_inv_C21, eigvals_C22, kept_C22 = psd_solve(
        C22,
        C12.T,
        rcond=rcond,
        ridge=ridge,
    )
    correction = U2 @ C22_inv_C21 @ U1.T
    C22_inv_C21_op = opnorm(C22_inv_C21)
    correction_op = opnorm(correction)

    positive_eigs = eigvals_C22[eigvals_C22 > 0]
    lam_min = float(np.min(positive_eigs)) if positive_eigs.size else np.nan
    lam_max = float(np.max(eigvals_C22)) if eigvals_C22.size else np.nan

    return TildeUDiagnostics(
        blocks=blocks,
        C12=C12,
        C22=C22,
        C22_inv_C21=C22_inv_C21,
        correction_matrix=correction,
        C22_inv_C21_op=C22_inv_C21_op,
        correction_op=correction_op,
        correction_op_relative_difference=(
            abs(correction_op - C22_inv_C21_op) / max(C22_inv_C21_op, 1e-15)
        ),
        C22_lambda_min=lam_min,
        C22_lambda_max=lam_max,
        C22_cond=lam_max / lam_min if lam_min > 0 else np.inf,
        C22_kept_rank=int(np.sum(kept_C22)),
    )


def leading_training_bias_variance_terms(
    *,
    P: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    C22_inv_C21: np.ndarray,
    w_observable: np.ndarray,
    singular_values: np.ndarray,
    N: int,
    approximate_identity: bool,
    test_probabilities: np.ndarray | None = None,
    test_second_moment: np.ndarray | None = None,
    dual_test_probabilities: np.ndarray | None = None,
    dual_test_second_moment: np.ndarray | None = None,
    pinv_rcond: float = 1e-12,
) -> dict:
    """Leading-order bias/variance formula for one fixed target observable.

    This function is the fixed-target counterpart of
    ``leading_training_bias_variance_terms_target_average``.  Here
    ``w_observable`` is one outcome-weight vector representing the target.

    The test distribution can be represented in either of two equivalent ways:

    * explicit probability columns ``test_probabilities``, in which case the
      function computes the leading error per column and then averages;
    * a second moment ``test_second_moment = E[p_test p_test.T]``, in which case
      the test average is performed exactly by quadratic contraction.

    Exact Haar test averages enter through the second-moment branch.  This is
    the correct object for squared errors: the error is linear in the test
    probability vector, but MSE and squared bias are quadratic in it.
    """
    if test_second_moment is None and test_probabilities is None:
        raise ValueError("Provide either test_second_moment or test_probabilities.")

    w_eff = _maybe_apply_correction_term(
        w_observable,
        U1,
        U2,
        C22_inv_C21,
        approximate_identity=approximate_identity,
    )
    cov_w_w = _c22_covariance_columns(P, w_eff, w_eff[:, None])[:, 0]
    P_pinv = np.linalg.pinv(P, rcond=pinv_rcond)

    if test_second_moment is not None:
        # Moment-based test average.  This covers exact Haar averages and any
        # caller-provided empirical second moment.  The dual moment is needed
        # because the leading bias is evaluated against the dual test
        # probabilities, not the primal POVM probabilities.
        test_second_moment = np.asarray(test_second_moment, dtype=float)
        if dual_test_second_moment is None:
            dual_probability_second_moment = _dual_state_probability_second_moment(
                test_second_moment,
                U1,
                singular_values,
                pinv_rcond,
            )
        else:
            dual_probability_second_moment = np.asarray(dual_test_second_moment, dtype=float)

        sum_sigma_w = w_eff * P.sum(axis=1) - P @ (P.T @ w_eff)
        if approximate_identity:
            bias_coeff = sum_sigma_w
        else:
            bias_coeff = U1 @ (U1.T @ sum_sigma_w) - U1 @ (
                C22_inv_C21.T @ (U2.T @ sum_sigma_w)
            )
        bias_sq = float(bias_coeff.T @ dual_probability_second_moment @ bias_coeff / (N**2))

        alpha_second_diag = np.diag(P_pinv @ test_second_moment @ P_pinv.T)
        variance = float(np.dot(cov_w_w, alpha_second_diag) / N)

        return {
            "bias_sq": bias_sq,
            "variance": variance,
            "mse": bias_sq + variance,
            "bias_abs_mean": np.nan,
            "bias_sq_max": np.nan,
            "variance_max": np.nan,
        }

    # Column-based test average.  This is algebraically equivalent to the
    # moment branch with test_second_moment = P_test @ P_test.T / ntest, but it
    # also exposes per-test diagnostics such as max and mean absolute bias.
    test_probabilities = np.asarray(test_probabilities, dtype=float)
    if dual_test_probabilities is None:
        dual_probabilities = _dual_state_probability_columns(
            test_probabilities,
            U1,
            singular_values,
            pinv_rcond,
        )
    else:
        dual_probabilities = np.asarray(dual_test_probabilities, dtype=float)
    test_eff = _maybe_apply_correction_term(
        dual_probabilities,
        U1,
        U2,
        C22_inv_C21,
        approximate_identity=approximate_identity,
    )

    cov_w_test = _c22_covariance_columns(P, w_eff, test_eff)
    bias_by_test = np.sum(cov_w_test, axis=0) / N
    bias_sq_by_test = bias_by_test**2

    dual_coefficients = P_pinv @ test_probabilities
    var_by_test = (dual_coefficients**2).T @ cov_w_w / N

    bias_sq = float(np.mean(bias_sq_by_test))
    variance = float(np.mean(var_by_test))

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "mse": bias_sq + variance,
        "bias_abs_mean": float(np.mean(np.abs(bias_by_test))),
        "bias_sq_max": float(np.max(bias_sq_by_test)),
        "variance_max": float(np.max(var_by_test)),
    }


def leading_training_bias_variance_terms_target_average(
    *,
    P: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    C22_inv_C21: np.ndarray,
    target_second_moment: np.ndarray,
    singular_values: np.ndarray,
    N: int,
    approximate_identity: bool,
    test_probabilities: np.ndarray | None = None,
    test_second_moment: np.ndarray | None = None,
    dual_test_probabilities: np.ndarray | None = None,
    dual_test_second_moment: np.ndarray | None = None,
    pinv_rcond: float = 1e-12,
) -> dict:
    """Leading-order bias/variance formula averaged over target observables.

    This is the target-average analogue of
    ``leading_training_bias_variance_terms``.  Instead of receiving one target
    vector ``w_observable``, it receives
    ``target_second_moment = E[w w.T]``.

    For a fixed noisy-training linearized map ``Delta``, target/test error has
    the form ``p_test.T @ Delta @ w``.  Averaging its square therefore needs only
    the second moments ``E[p_test p_test.T]`` and ``E[w w.T]``.  In particular,
    exact Haar averages are population averages here, not Monte Carlo samples.

    The function accepts tests either as explicit columns or as a second moment.
    Column tests are converted once to their empirical second moment because the
    target-averaged formula has no per-target/per-test diagnostics to preserve.
    """
    if test_second_moment is None and test_probabilities is None:
        raise ValueError("Provide either test_second_moment or test_probabilities.")

    target_second_moment = np.asarray(target_second_moment, dtype=float)
    nout = P.shape[0]

    if test_second_moment is not None:
        test_second_moment = np.asarray(test_second_moment, dtype=float)
        if dual_test_second_moment is None:
            dual_probability_second_moment = _dual_state_probability_second_moment(
                test_second_moment,
                U1,
                singular_values,
                pinv_rcond,
            )
        else:
            dual_probability_second_moment = np.asarray(dual_test_second_moment, dtype=float)
    else:
        # Convert an explicit finite test set into the same representation used
        # by exact Haar tests: S_test = average_j p_j p_j.T.
        test_probabilities = np.asarray(test_probabilities, dtype=float)
        test_second_moment = test_probabilities @ test_probabilities.T / test_probabilities.shape[1]
        if dual_test_probabilities is None:
            dual_probabilities = _dual_state_probability_columns(
                test_probabilities,
                U1,
                singular_values,
                pinv_rcond,
            )
        else:
            dual_probabilities = np.asarray(dual_test_probabilities, dtype=float)
        dual_probability_second_moment = (
            dual_probabilities @ dual_probabilities.T / dual_probabilities.shape[1]
        )

    # The target-averaged leading terms are matrix contractions of the same
    # fixed-target linear maps.  If target_second_moment were replaced by
    # outer(w, w), these contractions reduce to the fixed-target formula averaged
    # over a singleton target ensemble.
    target_transform = _c22_training_transform_matrix(
        nout,
        U1,
        U2,
        C22_inv_C21,
        approximate_identity=approximate_identity,
    )
    bias_transform = _bias_test_transform_matrix(
        nout,
        U1,
        U2,
        C22_inv_C21,
        approximate_identity=approximate_identity,
    )

    sum_sigma = _weighted_training_covariance_sum(P, np.ones(P.shape[1]))
    bias_linear_map = bias_transform @ sum_sigma @ target_transform
    bias_matrix = bias_linear_map.T @ dual_probability_second_moment @ bias_linear_map
    bias_sq = float(np.trace(bias_matrix @ target_second_moment) / (N**2))

    P_pinv = np.linalg.pinv(P, rcond=pinv_rcond)
    alpha_second_diag = np.diag(P_pinv @ test_second_moment @ P_pinv.T)
    variance_sigma = _weighted_training_covariance_sum(P, alpha_second_diag)
    target_eff_second = target_transform @ target_second_moment @ target_transform.T
    variance = float(np.trace(variance_sigma @ target_eff_second) / N)

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "mse": bias_sq + variance,
        "bias_abs_mean": np.nan,
        "bias_sq_max": np.nan,
        "variance_max": np.nan,
    }


def estimate_actual_training_mse(
    *,
    P: np.ndarray,
    w_observable: np.ndarray,
    N: int,
    rng: np.random.Generator | int | None = None,
    noise: str = "multinomial",
    actual_noise_trials: int = 200,
    test_probabilities: np.ndarray | None = None,
    test_second_moment: np.ndarray | None = None,
    lstsq_rcond: float | None = None,
) -> dict:
    """Estimate actual training error for one fixed target observable.

    The input ``P`` is the exact training probability/design matrix.  Each
    Monte Carlo repeat samples a noisy design matrix ``P_hat`` with
    ``noisy_probability_matrix`` and refits the observable weights against the
    noiseless training labels ``P.T @ w_observable``.

    This is the fixed-target counterpart of
    ``estimate_actual_training_mse_target_average``.  It tracks only
    ``delta = w_hat - w_observable`` for each noisy training repeat, so it is
    cheaper than fitting the full target map.  Test averaging follows the same
    convention as the leading fixed-target function: explicit test columns are
    averaged columnwise, while a provided second moment gives the exact
    quadratic average ``E[(p_test.T @ delta)^2]``.
    """
    rng = get_rng(rng)
    if actual_noise_trials <= 0:
        raise ValueError("actual_noise_trials must be positive.")
    if test_second_moment is None and test_probabilities is None:
        raise ValueError("Provide either test_second_moment or test_probabilities.")

    y_train = P.T @ w_observable
    deltas = []

    for _ in range(actual_noise_trials):
        P_hat = noisy_probability_matrix(P, rng, Nshots=N, noise=noise)
        w_hat = np.linalg.lstsq(P_hat.T, y_train, rcond=lstsq_rcond)[0]
        deltas.append(w_hat - w_observable)

    deltas = np.asarray(deltas, dtype=float)
    mean_delta = np.mean(deltas, axis=0)

    if test_second_moment is not None:
        # Exact/moment test average for the squared error.  This branch is used
        # for exact Haar test averages.
        test_second_moment = np.asarray(test_second_moment, dtype=float)
        mse_by_training_noise = np.einsum(
            "ti,ij,tj->t",
            deltas,
            test_second_moment,
            deltas,
            optimize=True,
        )
        actual_mse = float(np.mean(mse_by_training_noise))
        actual_bias_sq = float(mean_delta.T @ test_second_moment @ mean_delta)
        actual_variance = actual_mse - actual_bias_sq

        return {
            "actual_mse": actual_mse,
            "actual_bias_sq": actual_bias_sq,
            "actual_variance": float(actual_variance),
            "actual_abs_bias_mean": np.nan,
            "actual_noise_trials": int(actual_noise_trials),
        }

    # Explicit finite test set: compute every test error and average over test
    # columns.  This preserves finite-sample diagnostics such as abs-bias mean.
    test_probabilities = np.asarray(test_probabilities, dtype=float)
    errors = test_probabilities.T @ deltas.T
    errors = errors.T

    mean_error_by_test = np.mean(errors, axis=0)
    var_by_test = np.var(errors, axis=0)

    return {
        "actual_mse": float(np.mean(errors**2)),
        "actual_bias_sq": float(np.mean(mean_error_by_test**2)),
        "actual_variance": float(np.mean(var_by_test)),
        "actual_abs_bias_mean": float(np.mean(np.abs(mean_error_by_test))),
        "actual_noise_trials": int(actual_noise_trials),
    }


def estimate_actual_training_mse_target_average(
    *,
    P: np.ndarray,
    target_second_moment: np.ndarray,
    N: int,
    rng: np.random.Generator | int | None = None,
    noise: str = "multinomial",
    actual_noise_trials: int = 200,
    test_probabilities: np.ndarray | None = None,
    test_second_moment: np.ndarray | None = None,
    lstsq_rcond: float | None = None,
) -> dict:
    """Estimate actual training error averaged over target observables.

    This is the target-average counterpart of ``estimate_actual_training_mse``:
    each repeat samples a noisy design matrix ``P_hat`` and fits the linear map
    from noisy training probabilities back to exact training probabilities.

    The fitted map is compared with the identity:

        ``Delta = lstsq(P_hat.T, P.T)[0] - I``.

    For any target weight vector ``w``, this gives ``w_hat - w = Delta @ w``.
    Therefore the target/test averaged squared error is

        ``E[(p_test.T @ Delta @ w)^2]``

    and is computed by contracting ``Delta`` against
    ``test_second_moment = E[p_test p_test.T]`` and
    ``target_second_moment = E[w w.T]``.  This is why exact Haar target and test
    averages are handled with second moments rather than sampled states.
    """
    rng = get_rng(rng)
    if actual_noise_trials <= 0:
        raise ValueError("actual_noise_trials must be positive.")
    if test_second_moment is None and test_probabilities is None:
        raise ValueError("Provide either test_second_moment or test_probabilities.")

    target_second_moment = np.asarray(target_second_moment, dtype=float)
    if test_second_moment is None:
        # Put finite test columns in the same quadratic representation as an
        # exact Haar test average.
        test_probabilities = np.asarray(test_probabilities, dtype=float)
        test_second_moment = test_probabilities @ test_probabilities.T / test_probabilities.shape[1]
    else:
        test_second_moment = np.asarray(test_second_moment, dtype=float)

    deltas = []
    identity = np.eye(P.shape[0])

    for _ in range(actual_noise_trials):
        P_hat = noisy_probability_matrix(P, rng, Nshots=N, noise=noise)
        # Fit every target simultaneously.  Multiplying this matrix by a target
        # weight vector gives the same result as fitting that target alone, but
        # lets us average over target ensembles via E[w w.T].
        fit_matrix = np.linalg.lstsq(P_hat.T, P.T, rcond=lstsq_rcond)[0]
        deltas.append(fit_matrix - identity)

    deltas = np.asarray(deltas, dtype=float)
    mean_delta = np.mean(deltas, axis=0)
    mse_by_training_noise = np.einsum(
        "tai,ab,tbj,ij->t",
        deltas,
        test_second_moment,
        deltas,
        target_second_moment,
        optimize=True,
    )
    actual_mse = float(np.mean(mse_by_training_noise))
    actual_bias_sq = float(
        np.trace(mean_delta.T @ test_second_moment @ mean_delta @ target_second_moment)
    )

    return {
        "actual_mse": actual_mse,
        "actual_bias_sq": actual_bias_sq,
        "actual_variance": float(actual_mse - actual_bias_sq),
        "actual_abs_bias_mean": np.nan,
        "actual_noise_trials": int(actual_noise_trials),
    }


def _target_outcome_weights(target: ResolvedTarget, context: QELMTrainingContext) -> np.ndarray:
    if target.is_average:
        raise ValueError("Averaged targets do not have a single outcome-weight vector.")
    if target.weights is not None:
        return np.asarray(target.weights, dtype=float)
    if target.operator is None:
        raise ValueError("Fixed target must provide weights or an operator.")
    return _operator_to_outcome_weights(target.operator, context.povm_dual_effect_rows)


def _leading_terms_for_target(
    *,
    context: QELMTrainingContext,
    target: ResolvedTarget,
    P: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    C22_inv_C21: np.ndarray,
    singular_values: np.ndarray,
    N: int,
    approximate_identity: bool,
    test: ResolvedTest,
    pinv_rcond: float,
) -> dict:
    test_kwargs = {
        "test_probabilities": test.probabilities,
        "test_second_moment": test.second_moment,
        "dual_test_probabilities": test.dual_probabilities,
        "dual_test_second_moment": test.dual_second_moment,
        "pinv_rcond": pinv_rcond,
    }

    if target.is_average:
        return leading_training_bias_variance_terms_target_average(
            P=P,
            U1=U1,
            U2=U2,
            C22_inv_C21=C22_inv_C21,
            target_second_moment=target.second_moment,
            singular_values=singular_values,
            N=N,
            approximate_identity=approximate_identity,
            **test_kwargs,
        )

    return leading_training_bias_variance_terms(
        P=P,
        U1=U1,
        U2=U2,
        C22_inv_C21=C22_inv_C21,
        w_observable=_target_outcome_weights(target, context),
        singular_values=singular_values,
        N=N,
        approximate_identity=approximate_identity,
        **test_kwargs,
    )


def compute_qelm_diagnostics(
    spec: QELMTrainingSpec,
    context: QELMTrainingContext,
) -> TildeUDiagnostics:
    return tilde_u_correction_operator_diagnostics(
        P=context.P_train,
        rank=_rank_from_spec(spec),
        rcond=spec.numerics.rcond,
        ridge=spec.numerics.ridge,
    )


def _compute_qelm_leading_error_resolved(
    spec: QELMTrainingSpec,
    context: QELMTrainingContext,
    *,
    target: ResolvedTarget,
    test: ResolvedTest,
    diagnostics: TildeUDiagnostics,
    corrected: bool = True,
) -> QELMLeadingErrorResult:
    blocks = diagnostics["blocks"]
    N = _required_noise_N(spec.noise)
    metrics = _leading_terms_for_target(
        context=context,
        target=target,
        test=test,
        P=context.P_train,
        U1=blocks["U1"],
        U2=blocks["U2"],
        C22_inv_C21=diagnostics["C22_inv_C21"],
        singular_values=blocks["singular_values"],
        N=N,
        approximate_identity=not corrected,
        pinv_rcond=spec.numerics.rcond,
    )
    return QELMLeadingErrorResult.from_metrics(metrics, corrected=corrected)


def compute_qelm_leading_error(
    spec: QELMTrainingSpec,
    context: QELMTrainingContext,
    rng: np.random.Generator | int | None = None,
    *,
    corrected: bool = True,
) -> QELMLeadingErrorResult:
    run = QELMRun(spec, rng=rng)
    run._context = context
    test = run.test
    diagnostics = run.diagnostics
    target = run.target
    return _compute_qelm_leading_error_resolved(
        spec,
        context,
        target=target,
        test=test,
        diagnostics=diagnostics,
        corrected=corrected,
    )


def _run_qelm_training_resolved(
    spec: QELMTrainingSpec,
    rng: np.random.Generator | int | None = None,
    *,
    context: QELMTrainingContext,
    target: ResolvedTarget,
    test: ResolvedTest,
    return_fits: bool = False,
    return_fit_matrix: bool = False,
) -> QELMTrainingResults:
    """Run the actual noisy-training simulation for resolved target/test data.

    This is the implementation used by ``QELMRun.train_model``.  It mirrors the
    two standalone helpers above:

    * fixed target: fit one weight vector and evaluate ``p_test.T @ (w_hat-w)``;
    * target average: fit the whole linear map and evaluate
      ``p_test.T @ Delta @ w`` through test/target second moments.

    The split is intentional for performance.  Fixed-target runs avoid fitting
    the full map unless ``return_fit_matrix`` is requested.
    """
    rng = get_rng(rng)
    P = context.P_train
    N = _required_noise_N(spec.noise)
    noise = spec.noise.noise
    actual_noise_trials = spec.noise.actual_noise_trials
    lstsq_rcond = spec.noise.lstsq_rcond

    if actual_noise_trials <= 0:
        raise ValueError("actual_noise_trials must be positive.")

    if test.second_moment is None and test.probabilities is None:
        raise ValueError("Resolved test must provide either second_moment or probabilities.")

    if target.is_average:
        # Target-average actual MSE.  We cannot fit a single weight vector,
        # because the target is an ensemble represented by E[w w.T].  Instead
        # each noisy training repeat fits the full linear map from exact
        # training probabilities to fitted outcome weights.
        target_second_moment = np.asarray(target.second_moment, dtype=float)
        # Put fixed, finite-sample, and exact-Haar tests into one quadratic
        # representation.  For a fixed test state this is outer(p, p); for a
        # finite sample it is the empirical average; for Haar it is exact.
        test_second_moment = _test_second_moment(test)
        fit_matrices = []
        identity = np.eye(P.shape[0])

        for _ in range(actual_noise_trials):
            P_hat = noisy_probability_matrix(P, rng, Nshots=N, noise=noise)
            # this produces (P_hat.T^+ @ P.T) = (P @ P_hat^+)^T
            fit_matrix = np.linalg.lstsq(P_hat.T, P.T, rcond=lstsq_rcond)[0]
            fit_matrices.append(fit_matrix)

        fit_matrices = np.asarray(fit_matrices, dtype=float)
        deltas = fit_matrices - identity
        mean_delta = np.mean(deltas, axis=0)
        # For each noisy repeat, average (p.T @ Delta @ w)^2 over the resolved
        # test and target ensembles using only their second moments.
        mse_by_training_noise = np.einsum(
            "tai,ab,tbj,ij->t",
            deltas,
            test_second_moment,
            deltas,
            target_second_moment,
            optimize=True,
        )
        mse = float(np.mean(mse_by_training_noise))
        bias_sq = float(
            np.trace(mean_delta.T @ test_second_moment @ mean_delta @ target_second_moment)
        )
        variance = float(mse - bias_sq)
        stored_fit_matrices = fit_matrices if (return_fits or return_fit_matrix) else None
        return QELMTrainingResults(
            mse=mse,
            bias_sq=bias_sq,
            variance=variance,
            abs_bias_mean=np.nan,
            noise_trials=int(actual_noise_trials),
            mean_fit_matrix=np.mean(fit_matrices, axis=0),
            fit_matrices=stored_fit_matrices,
        )

    # Fixed-target actual MSE.  This path is cheaper than the target-average
    # path because it fits only the requested target weight vector.
    w_observable = _target_outcome_weights(target, context)
    y_train = P.T @ w_observable
    fitted_weights = []
    fit_matrices = [] if return_fit_matrix else None

    # this is where we actually compute P^+ for noisy matrices
    for _ in range(actual_noise_trials):
        P_hat = noisy_probability_matrix(P, rng, Nshots=N, noise=noise)
        if return_fit_matrix and fit_matrices is not None:
            fit_matrix = np.linalg.lstsq(P_hat.T, P.T, rcond=lstsq_rcond)[0]
            fit_matrices.append(fit_matrix)
            w_hat = fit_matrix @ w_observable
        else:
            w_hat = np.linalg.lstsq(P_hat.T, y_train, rcond=lstsq_rcond)[0]
        fitted_weights.append(w_hat)

    fitted_weights = np.asarray(fitted_weights, dtype=float)
    deltas = fitted_weights - w_observable
    mean_delta = np.mean(deltas, axis=0)

    if test.second_moment is not None:
        # Exact/moment test average for a fixed target:
        # E[(p_test.T @ delta)^2] = delta.T E[p_test p_test.T] delta.
        test_second_moment = np.asarray(test.second_moment, dtype=float)
        mse_by_training_noise = np.einsum(
            "ti,ij,tj->t",
            deltas,
            test_second_moment,
            deltas,
            optimize=True,
        )
        mse = float(np.mean(mse_by_training_noise))
        bias_sq = float(mean_delta.T @ test_second_moment @ mean_delta)
        variance = float(mse - bias_sq)
        abs_bias_mean = np.nan
    else:
        # Explicit finite test columns.  Average over columns directly so the
        # reported bias diagnostics remain per-test finite-sample quantities.
        test_probabilities = np.asarray(test.probabilities, dtype=float)
        errors = deltas @ test_probabilities
        mean_error_by_test = np.mean(errors, axis=0)
        var_by_test = np.var(errors, axis=0)
        mse = float(np.mean(errors**2))
        bias_sq = float(np.mean(mean_error_by_test**2))
        variance = float(np.mean(var_by_test))
        abs_bias_mean = float(np.mean(np.abs(mean_error_by_test)))

    fit_matrices_array = None
    mean_fit_matrix = None
    if return_fit_matrix:
        fit_matrices_array = np.asarray(fit_matrices, dtype=float)
        mean_fit_matrix = np.mean(fit_matrices_array, axis=0)

    return QELMTrainingResults(
        mse=mse,
        bias_sq=bias_sq,
        variance=variance,
        abs_bias_mean=abs_bias_mean,
        noise_trials=int(actual_noise_trials),
        mean_weights=np.mean(fitted_weights, axis=0),
        mean_fit_matrix=mean_fit_matrix,
        fitted_weights=fitted_weights if return_fits else None,
        fit_matrices=fit_matrices_array if return_fit_matrix else None,
    )


def run_qelm_actual_training(
    spec: QELMTrainingSpec,
    rng: np.random.Generator | int | None = None,
    *,
    return_fits: bool = False,
    return_fit_matrix: bool = False,
) -> QELMTrainingResults:
    return QELMRun(spec, rng=rng).train_model(
        return_fits=return_fits,
        return_fit_matrix=return_fit_matrix,
    )


def analyze_qelm_training(
    spec: QELMTrainingSpec,
    rng: np.random.Generator | int | None = None,
    *,
    include_actual: bool = True,
    include_leading: bool = True,
    return_fits: bool = False,
    return_fit_matrix: bool = False,
) -> QELMTrainingAnalysisResult:
    return QELMRun(spec, rng=rng).analyze(
        compute_mse=include_actual,
        include_leading=include_leading,
        return_fits=return_fits,
        return_fit_matrix=return_fit_matrix,
    )


def with_training_sweep_value(
    spec: QELMTrainingSpec,
    sweep_col: str,
    value: int,
) -> QELMTrainingSpec:
    if sweep_col == "ntr":
        return replace(
            spec,
            data=replace(
                spec.data,
                train_states=_with_training_num_states(spec.data.train_states, int(value)),
            ),
        )
    if sweep_col == "nout":
        return replace(spec, data=replace(spec.data, nout=int(value)))
    if sweep_col == "N":
        return replace(spec, noise=replace(spec.noise, N=int(value)))
    raise ValueError("sweep_col must be 'ntr', 'nout', or 'N'.")


def _test_second_moment(test: ResolvedTest) -> np.ndarray:
    """Return E[p p.T] for any resolved test representation.

    Exact Haar tests already store the population second moment.  Column-based
    tests, including one fixed state, are converted to the matching empirical
    second moment.
    """
    if test.second_moment is not None:
        return np.asarray(test.second_moment, dtype=float)
    probabilities = np.asarray(test.probabilities, dtype=float)
    return probabilities @ probabilities.T / probabilities.shape[1]
