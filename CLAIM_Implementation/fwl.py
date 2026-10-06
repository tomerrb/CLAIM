"""The full-query FWL surrogate for CLAIM.

Both evaluations use treatment statistics computed from the already-private
PGM model at the start of each round.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FWLWeights:
    propensity: dict[tuple, float]
    variance: float


def _scaled_columns(df, cols, bounds, treatment_col):
    missing = [col for col in cols if col not in df.columns or col not in bounds]
    if missing:
        raise ValueError(f"Missing columns or public bounds: {missing}")
    out = df[cols].copy()
    for col in cols:
        values = out[col].astype(float)
        if col == treatment_col:
            if not values.isin([0.0, 1.0]).all():
                raise ValueError("The treatment must be encoded as 0 or 1")
            out[col] = values
            continue
        lo, hi = bounds[col]
        if hi <= lo or not np.isfinite([lo, hi]).all():
            raise ValueError(f"Invalid public bounds for {col}: {(lo, hi)}")
        if not np.isfinite(values).all() or ((values < lo) | (values > hi)).any():
            raise ValueError(f"Values for {col} fall outside the public bounds")
        out[col] = (values - lo) / (hi - lo)
    return out


def _normalized_model_marginal(df, cols, bounds, treatment_col, weight_col):
    if weight_col not in df.columns:
        raise ValueError(f"Missing probability column: {weight_col}")
    scaled = _scaled_columns(df, cols, bounds, treatment_col)
    weights = df[weight_col].astype(float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Model probabilities must be finite and nonnegative")
    scaled[weight_col] = weights.to_numpy()
    marginal = scaled.groupby(cols, dropna=False)[weight_col].sum()
    total = float(marginal.sum())
    if total <= 0:
        raise ValueError("The model distribution has zero probability mass")
    return marginal / total


def model_fwl_weights(
    model_cells: pd.DataFrame,
    treatment_col: str,
    adjustment_set: list[str],
    bounds: dict[str, tuple[float, float]],
    overlap_eta: float,
    weight_col: str = "_w",
) -> FWLWeights:
    """Compute e(z) and max(E_model[e(Z)(1-e(Z))], eta) from the current PGM."""
    if not 0 < overlap_eta <= 0.25:
        raise ValueError("overlap_eta must be in (0, 1/4]")
    cols = [treatment_col, *adjustment_set]
    marginal = _normalized_model_marginal(
        model_cells, cols, bounds, treatment_col, weight_col
    )
    mass_by_z = {}
    treated_by_z = {}
    for cell, prob in marginal.items():
        cell = cell if isinstance(cell, tuple) else (cell,)
        treatment = float(cell[0])
        z = tuple(cell[1:])
        mass_by_z[z] = mass_by_z.get(z, 0.0) + float(prob)
        treated_by_z[z] = treated_by_z.get(z, 0.0) + treatment * float(prob)
    propensity = {
        z: treated_by_z[z] / mass
        for z, mass in mass_by_z.items()
        if mass > 0
    }
    variance = sum(
        mass_by_z[z] * e * (1.0 - e) for z, e in propensity.items()
    )
    return FWLWeights(propensity, max(float(variance), overlap_eta))


def legacy_empirical_variance(df, treatment_col, adjustment_set, bounds):
    """Preserve the separate legacy DoWhy selection path."""
    cols = [treatment_col, *adjustment_set]
    scaled = _scaled_columns(df, cols, bounds, treatment_col)
    groups = scaled.groupby(cols, dropna=False).size().astype(float)
    model_cells = groups.reset_index(name="_w")
    return model_fwl_weights(
        model_cells, treatment_col, adjustment_set, bounds,
        overlap_eta=1e-12,
    ).variance


def _fwl_sum(marginal, treatment_col, outcome_col, adjustment_set, weights):
    result = 0.0
    for cell, mass in marginal.items():
        cell = cell if isinstance(cell, tuple) else (cell,)
        record = dict(zip(marginal.index.names, cell))
        z = tuple(record[col] for col in adjustment_set)
        if z not in weights.propensity:
            raise ValueError(f"The current model has no treatment probability for Z={z}")
        result += float(mass) * float(record[outcome_col]) * (
            float(record[treatment_col]) - weights.propensity[z]
        ) / weights.variance
    return float(result)


def claim_fwl_ate_estimator(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    adjustment_set: list[str],
    bounds: dict[str, tuple[float, float]],
    *,
    weights: FWLWeights,
    n_tilde: int,
) -> float:
    """Evaluate the data-side FWL surrogate on M_s(D) / n_tilde."""
    if n_tilde < 1:
        raise ValueError("n_tilde must be positive")
    cols = [treatment_col, outcome_col, *adjustment_set]
    scaled = _scaled_columns(df, cols, bounds, treatment_col)
    counts = scaled.groupby(cols, dropna=False).size().astype(float)
    return _fwl_sum(counts / n_tilde, treatment_col, outcome_col, adjustment_set, weights)


def claim_fwl_ate_from_distribution(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    adjustment_set: list[str],
    bounds: dict[str, tuple[float, float]],
    *,
    weights: FWLWeights,
    weight_col: str = "_w",
) -> float:
    """Evaluate the same FWL coefficients on the PGM joint probabilities."""
    cols = [treatment_col, outcome_col, *adjustment_set]
    marginal = _normalized_model_marginal(
        df, cols, bounds, treatment_col, weight_col
    )
    return _fwl_sum(marginal, treatment_col, outcome_col, adjustment_set, weights)


def claim_adjusted_ate(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    adjustment_set: list[str],
    bounds: dict[str, tuple[float, float]],
    n_min: int = 1,
) -> float:
    """Ordinary plug-in ATE, used for evaluation, not the score."""
    if n_min < 1:
        raise ValueError("n_min must be at least one")
    cols = [treatment_col, outcome_col, *adjustment_set]
    scaled = _scaled_columns(df, cols, bounds, treatment_col)
    if scaled.empty:
        raise ValueError("An ATE cannot be estimated from an empty dataset")
    groups = {}
    for row in scaled.itertuples(index=False, name=None):
        treatment, outcome, *z = row
        key = (tuple(z), int(treatment))
        count, total = groups.get(key, (0, 0.0))
        groups[key] = (count + 1, total + float(outcome))
    strata = {z for z, _ in groups}
    ate = 0.0
    for z in strata:
        n0, s0 = groups.get((z, 0), (0, 0.0))
        n1, s1 = groups.get((z, 1), (0, 0.0))
        ate += (n0 + n1) / len(scaled) * (
            s1 / max(n1, n_min) - s0 / max(n0, n_min)
        )
    return float(ate)
