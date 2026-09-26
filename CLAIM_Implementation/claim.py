"""Implementation of CLAIM: Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data.

This implementation supports two selection modes:
- "marginal" (default): Original L1-based selection using the exponential mechanism.
  Selects the worst-approximated marginal based on L1 distance.
- "ate": ATE-based selection for causal inference applications.
  Selects the marginal that most improves Average Treatment Effect estimation.
  Requires DoWhy library and causal structure specification.

Note that with the default settings, CLAIM can take many hours to run. You can configure
the runtime/utility tradeoff via the max_model_size flag. We recommend setting it to 1.0
for debugging, but keeping the default value of 80 for any official comparisons.

Note that we assume the data has been appropriately preprocessed so that there are no
large-cardinality categorical attributes. If there are, we recommend using something like
"compress_domain" from mst.py.
"""

import gc
import jax
import numpy as np
import itertools
from mbi import (
    Dataset,
    Domain,
    estimation,
    junction_tree,
    LinearMeasurement,
)
from mechanism import Mechanism
from collections import defaultdict
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse
from cdp2adp import cdp_rho, cdp_eps
import warnings


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def compile_workload(workload):
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
            junction_tree.hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class CLAIM(Mechanism):
    """Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data.

    LAMBDA_UPDATE_FACTOR = 1.25
    TOLERANCE_Z = 2.0
    
    Args:
        epsilon: Privacy parameter (epsilon for zCDP conversion).
        delta: Privacy parameter.
        prng: Optional random number generator.
        rounds: Number of selection rounds (default: 16 * domain size).
        max_model_size: Maximum model size in MB (default: 80).
        max_iters: Maximum iterations for mirror descent (default: 1000).
        structural_zeros: Dict of structural zeros constraints.
        selection_mode: "marginal" (L1-based) or "ate" (ATE-based) selection.
        ate_configs: List of ATE configurations (required for ATE mode). Each config is a dict:
            - "name": str, identifier for this ATE
            - "treatment": str, treatment variable name
            - "outcome": str, outcome variable name  
            - "confounders": list[str], confounder variable names
            - "alpha": float, weight in [0, 1] for this ATE in utility function
              (this is β_i in claim_algorithm_fwl.tex; the field name is kept
              as "alpha" for backward compatibility with existing config files)
        causal_graph_path: Path to GML file with causal graph (required for ATE mode).
        ate_sample_size: Samples for final ATE computation (default: 10000).
        sim_sample_size: Samples for simulation during selection (default: 5000).
        marginal_weight: λ in claim_algorithm_fwl.tex. Weight for the
            statistical term L_r in q_r(D) = λ·L_r + (1-λ)·κ·A_r (default: 0.3).
            0.0 = pure causal (κ·A_r) selection, 1.0 = pure statistical (L_r)
            selection. Only used when selection_mode="ate".
    """

    LAMBDA_UPDATE_FACTOR = 1.25
    TOLERANCE_Z = 2.0

    def __init__(
        self,
        epsilon,
        delta,
        prng=None,
        rounds=None,
        max_model_size=80,
        max_iters=1000,
        structural_zeros={},
        selection_mode="marginal",
        ate_configs=None,
        causal_graph_path=None,
        ate_sample_size=10000,
        sim_sample_size=5000,
        marginal_weight=0.3,
        ate_method="dowhy",
        mu_eta=1e-6,
        kappa_eta=1e-6,
        adaptive_lambda=True,
        lambda_min=0.1,
        lambda_max=0.9,
        ate_tolerance=None,
        tvd_tolerance=None,
        reference_ates=None,
        ate_sensitivity=None,
        reference_ate_rho_fraction=0.05,
        ate_outcome_range=(-1.0, 1.0),
    ):
        super(CLAIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

        # Selection mode configuration
        self.selection_mode = selection_mode
        if selection_mode not in ("marginal", "ate"):
            raise ValueError(f"selection_mode must be 'marginal' or 'ate', got '{selection_mode}'")

        # ATE estimation backend
        if ate_method not in ("dowhy", "fwl"):
            raise ValueError(f"ate_method must be 'dowhy' or 'fwl', got '{ate_method}'")
        self.ate_method = ate_method

        # ATE mode configuration
        self.ate_configs = ate_configs or []
        self.ate_sample_size = ate_sample_size
        self.sim_sample_size = sim_sample_size
        self.causal_graph = None
        self._true_ates = {}  # Dict of {name: true_ate_value}
        # FWL components cached at the start of run() (ate_method="fwl" only):
        # v_i = E_D[(T_i - T̄(Z_i))^2] per ATE config and κ = 1 / Σ_i β_i / v_i
        self._fwl_v = {}
        self._fwl_kappa = None

        # Hybrid selection configuration
        self.marginal_weight = float(marginal_weight)
        self.adaptive_lambda = bool(adaptive_lambda)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_update_factor = self.LAMBDA_UPDATE_FACTOR
        self.ate_tolerance = None if ate_tolerance is None else float(ate_tolerance)
        self.tvd_tolerance = None if tvd_tolerance is None else float(tvd_tolerance)
        self.sigma_ate = 0.0
        self.ate_sensitivity = ate_sensitivity
        self.reference_ate_rho_fraction = float(reference_ate_rho_fraction)
        self.ate_outcome_range = (float(ate_outcome_range[0]), float(ate_outcome_range[1]))
        self.mu_eta = float(mu_eta)
        self.kappa_eta = float(kappa_eta)
        self.reference_ates = reference_ates
        
        if not (0.0 <= self.marginal_weight <= 1.0):
            raise ValueError(f"marginal_weight must be in [0, 1], got {self.marginal_weight}")
        if not (0.0 <= self.lambda_min <= self.lambda_max <= 1.0):
            raise ValueError("lambda_min/lambda_max must satisfy 0 <= min <= max <= 1")
        if self.adaptive_lambda and self.lambda_min <= 0:
            raise ValueError("adaptive lambda requires lambda_min > 0")

        if selection_mode == "ate":
            if not ate_configs:
                raise ValueError("ATE mode requires ate_configs (list of ATE configurations)")
            if ate_method == "dowhy" and not causal_graph_path:
                raise ValueError("ATE mode with ate_method='dowhy' requires causal_graph_path")

            # Validate each ATE config
            for i, config in enumerate(ate_configs):
                required_keys = ["name", "treatment", "outcome", "confounders", "alpha"]
                for key in required_keys:
                    if key not in config:
                        raise ValueError(f"ATE config {i} missing required key: '{key}'")
                if not (0 <= config["alpha"] <= 1):
                    raise ValueError(f"ATE config '{config['name']}' alpha must be in [0, 1], got {config['alpha']}")

                if ate_method == "fwl":
                    if "bounds" not in config:
                        raise ValueError(
                            f"ATE config '{config['name']}' missing required key 'bounds' for ate_method='fwl'"
                        )
                    needed = [config["treatment"], config["outcome"], *config["confounders"]]
                    missing = [c for c in needed if c not in config["bounds"]]
                    if missing:
                        raise ValueError(
                            f"ATE config '{config['name']}' bounds missing entries for: {missing}"
                        )

            # Validate alpha weights sum to 1
            alpha_sum = sum(config["alpha"] for config in ate_configs)
            if abs(alpha_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"ATE alpha weights must sum to 1.0, got {alpha_sum:.6f}. "
                    f"Weights: {[config['alpha'] for config in ate_configs]}"
                )

            # Load GML causal graph (shared across all ATEs) — DoWhy only
            if ate_method == "dowhy":
                with open(causal_graph_path, 'r') as f:
                    self.causal_graph = f.read()

    def _binarize_for_ate(self, df, config):
        """Apply binarization to treatment and outcome columns for ATE calculation.
        
        This converts encoded (integer) data to binary (0/1) based on the config.
        Called just before ATE calculation to prepare the data for DoWhy.
        
        Args:
            df: DataFrame with encoded data.
            config: ATE config dict with optional 'binarization' field:
                {
                    "treatment": "educational-num",
                    "outcome": "income",
                    "binarization": {
                        "treatment": {"type": "threshold", "threshold": 9, "comparison": ">"},
                        "outcome": {"type": "threshold", "threshold": 0, "comparison": ">"}
                    }
                }
                If binarization not specified, assumes data is already binary.
        
        Returns:
            DataFrame: Copy with binarized treatment and outcome.
        """
        df = df.copy()
        treatment = config["treatment"]
        outcome = config["outcome"]
        binarization = config.get("binarization", {})
        
        # Binarize treatment if config provided
        if "treatment" in binarization:
            bin_config = binarization["treatment"]
            if bin_config.get("type") == "threshold":
                threshold = bin_config.get("threshold", 0)
                comparison = bin_config.get("comparison", ">")
                if comparison == ">":
                    df[treatment] = (df[treatment] > threshold).astype(int)
                elif comparison == ">=":
                    df[treatment] = (df[treatment] >= threshold).astype(int)
                elif comparison == "<":
                    df[treatment] = (df[treatment] < threshold).astype(int)
                elif comparison == "<=":
                    df[treatment] = (df[treatment] <= threshold).astype(int)
            elif bin_config.get("type") == "value":
                # For categorical: specific value = 1, others = 0
                # In encoded data, this is a specific integer
                positive_value = bin_config.get("positive_value", 1)
                df[treatment] = (df[treatment] == positive_value).astype(int)
            elif bin_config.get("type") == "categorical":
                # For categorical type: support list (encoded_positive_values) or single value
                encoded_positive_values = bin_config.get("encoded_positive_values")
                if encoded_positive_values is not None:
                    df[treatment] = df[treatment].isin(encoded_positive_values).astype(int)
                else:
                    encoded_positive_value = bin_config.get("encoded_positive_value")
                    if encoded_positive_value is not None:
                        df[treatment] = (df[treatment] == encoded_positive_value).astype(int)
                    else:
                        positive_value = bin_config.get("positive_value", 1)
                        df[treatment] = (df[treatment] == positive_value).astype(int)

        # Binarize outcome if config provided
        if "outcome" in binarization:
            bin_config = binarization["outcome"]
            if bin_config.get("type") == "threshold":
                threshold = bin_config.get("threshold", 0)
                comparison = bin_config.get("comparison", ">")
                if comparison == ">":
                    df[outcome] = (df[outcome] > threshold).astype(int)
                elif comparison == ">=":
                    df[outcome] = (df[outcome] >= threshold).astype(int)
            elif bin_config.get("type") == "value":
                positive_value = bin_config.get("positive_value", 1)
                df[outcome] = (df[outcome] == positive_value).astype(int)
            elif bin_config.get("type") == "categorical":
                # For categorical type: use encoded_positive_value for encoded data
                encoded_positive_value = bin_config.get("encoded_positive_value")
                if encoded_positive_value is not None:
                    df[outcome] = (df[outcome] == encoded_positive_value).astype(int)
                else:
                    # Fallback to positive_value if encoded_positive_value not specified
                    positive_value = bin_config.get("positive_value", 1)
                    df[outcome] = (df[outcome] == positive_value).astype(int)
        
        return df

    def compute_ate_from_data(self, df, config):
        """Compute ATE for a single treatment-outcome pair.

        Dispatches on self.ate_method:
          - "dowhy": DoWhy CausalModel with backdoor.linear_regression.
          - "fwl":   CLAIM-style FWL estimator from fwl.claim_fwl_ate_estimator.

        Args:
            df: DataFrame containing treatment, outcome, and confounder columns (encoded).
            config: ATE config dict with treatment, outcome, optional binarization,
                and (for FWL) a 'bounds' dict.

        Returns:
            float: Estimated ATE value.
        """
        treatment = config["treatment"]
        outcome = config["outcome"]

        # Apply binarization for ATE calculation
        df_binary = self._binarize_for_ate(df, config)

        if self.ate_method == "fwl":
            from fwl import claim_fwl_ate_estimator

            bounds = {k: tuple(v) for k, v in config["bounds"].items()}
            return claim_fwl_ate_estimator(
                df=df_binary,
                treatment_col=treatment,
                outcome_col=outcome,
                adjustment_set=list(config["confounders"]),
                bounds=bounds,
            )

        # Default: DoWhy backdoor adjustment
        from dowhy import CausalModel

        model = CausalModel(
            data=df_binary,
            treatment=treatment,
            outcome=outcome,
            graph=self.causal_graph
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        return estimate.value

    def compute_all_ates(self, df):
        """Compute all configured ATEs from a DataFrame.

        Args:
            df: DataFrame containing all required columns (encoded).

        Returns:
            dict: {ate_name: ate_value} for each configured ATE.
        """
        ates = {}
        for config in self.ate_configs:
            ate_value = self.compute_ate_from_data(df, config)
            ates[config["name"]] = ate_value
        return ates

    def _compute_fwl_v_for_config(self, df, config):
        """Compute v_i = E_D[(T_i - T̄(Z_i))^2] for a single ATE config.

        Uses the same binarization and bounds as the FWL ATE estimator so the
        v values stay consistent with τ_i^*.
        """
        from fwl import claim_fwl_v

        df_binary = self._binarize_for_ate(df, config)
        bounds = {k: tuple(v) for k, v in config["bounds"].items()}
        return claim_fwl_v(
            df=df_binary,
            treatment_col=config["treatment"],
            adjustment_set=list(config["confounders"]),
            bounds=bounds,
        )

    def _cache_fwl_components(self, df):
        """Cache v_i for each ATE and the κ scaling factor used in q_r.

        κ = 1 / Σ_i (β_i / v_i), where β_i is the per-ATE weight (config['alpha']).
        Called once at the start of run() when ate_method='fwl'.
        """
        self._fwl_v = {
            config["name"]: self._compute_fwl_v_for_config(df, config)
            for config in self.ate_configs
        }
        denom = 0.0
        for config in self.ate_configs:
            v_i = self._fwl_v[config["name"]]
            if v_i <= 0:
                raise ValueError(
                    f"Degenerate v_i=0 for ATE '{config['name']}'; cannot form κ."
                )
            denom += config["alpha"] / v_i
        if denom <= 0:
            raise ValueError("Cannot compute FWL κ: denominator is zero")
        self._fwl_kappa = 1.0 / denom

    def compute_weighted_ate_error(self, true_ates, model_ates):
        """Compute weighted sum of ATE errors: Σ β_i · |τ_i* - τ_i(model)|.

        The per-ATE weight ``config["alpha"]`` corresponds to β_i in
        claim_algorithm_fwl.tex (kept named "alpha" in code for backward
        compatibility with existing ATE config files).
        """
        weighted_error = 0.0
        for config in self.ate_configs:
            name = config["name"]
            beta = config["alpha"]
            error = abs(true_ates[name] - model_ates[name])
            weighted_error += beta * error
        return weighted_error

    def _ate_error_for_model(self, model, seed=42):
        """Compute absolute ATE error for the current PGM model."""
        if not self.ate_configs:
            raise ValueError("ATE configs required for ATE error computation")
        
        if self.ate_method == "fwl":
            current_ates = {
                cfg["name"]: self._ate_from_model(model, cfg)
                for cfg in self.ate_configs
            }
        else:
            df = self._model_to_dataframe(model, self.sim_sample_size, seed=seed)
            current_ates = self.compute_all_ates(df)
            
        error = self.compute_weighted_ate_error(self._true_ates, current_ates)
        return error, current_ates

    def _tvd_error_for_model(self, model, measurements, cliques):
        """Average TVD between the model and already-released noisy marginals.

        Pure post-processing of DP-released measurements — no additional privacy cost.

        Args:
            model: Current fitted PGM model.
            measurements: List of LinearMeasurement objects released so far.
            cliques: Cliques to average TVD over (typically one-way marginals).

        Returns:
            float: Mean approximate TVD across matched cliques.
        """
        noisy_by_clique = {m.clique: m.noisy_measurement for m in measurements if m.clique in cliques}
        used_cliques = [cl for cl in cliques if cl in noisy_by_clique]
        if not used_cliques:
            return 0.0
        total = 0.0
        for cl in used_cliques:
            y = noisy_by_clique[cl]
            xest = model.project(cl).datavector()
            total += 0.5 * np.linalg.norm(y - xest, 1) / max(y.sum(), 1e-12)
        return total / len(used_cliques)

    def _derived_tolerances(self, measurements, cliques):
        """Noise-floor-derived tolerances (theta_A, theta_L), fully public.

        The principle: a tolerance below the error a *perfect* model would
        exhibit is incoherent -- the schedule would chase pure noise. So each
        tolerance is z times its perfect-model noise floor:

        - theta_A = z * (sigma_ATE + sigma_MC). Even if the model's ATE were
          exact, the measured error |ref - tau(model)| includes the Gaussian
          noise of the reference release (std sigma_ATE, public) and the
          Monte-Carlo error of estimating the model ATE from sim_sample_size
          synthetic rows, conservatively bounded by
          sigma_MC = (y_max - y_min) / sqrt(sim_sample_size).
        - theta_L = z * mean_r sqrt(2/pi) * sigma_r * n_r / (2 * sum(y_r)),
          the expected TVD between a perfect model and the *noisy* released
          one-way marginals (E|N(0,s)| = s * sqrt(2/pi) per cell), using the
          released noisy total as the public stand-in for N.

        z = TOLERANCE_Z = 2 converts the mean noise level into a
        pass-with-high-probability threshold (~95% for the Gaussian case).
        Every ingredient is part of the DP transcript, so deriving the
        tolerances is post-processing.
        """
        lo, hi = self.ate_outcome_range
        sigma_mc = (hi - lo) / np.sqrt(self.sim_sample_size)
        theta_a = self.TOLERANCE_Z * (self.sigma_ate + sigma_mc)

        by_clique = {m.clique: m for m in measurements if m.clique in cliques}
        floors = []
        for cl in cliques:
            if cl not in by_clique:
                continue
            m = by_clique[cl]
            y = m.noisy_measurement
            total = max(float(np.sum(y)), 1e-12)
            floors.append(np.sqrt(2 / np.pi) * m.stddev * y.size / (2 * total))
        if not floors:
            raise ValueError("no noisy measurements available to derive tvd_tolerance")
        theta_l = self.TOLERANCE_Z * float(np.mean(floors))
        return theta_a, theta_l

    def _adjust_lambda(self, current_lambda, tvd_error, ate_error, tvd_tolerance, ate_tolerance):
        """Dual-criterion lambda update from Maria Vologdin (de3bee5).

        Lambda (marginal_weight) decreases toward the causal term only when TVD
        is already within tolerance — ensuring TVD preservation cannot be
        sacrificed without bound to chase ATE improvements.
        """
        ate_ok = ate_error <= ate_tolerance
        tvd_ok = tvd_error <= tvd_tolerance

        if ate_ok and tvd_ok:
            return current_lambda
        if tvd_ok and not ate_ok:
            return max(self.lambda_min, current_lambda / self.lambda_update_factor)
        if ate_ok and not tvd_ok:
            return min(self.lambda_max, current_lambda * self.lambda_update_factor)

        # Neither preserved: favor whichever is relatively further from its tolerance.
        ate_ratio = ate_error / ate_tolerance
        tvd_ratio = tvd_error / tvd_tolerance
        if tvd_ratio >= ate_ratio:
            return min(self.lambda_max, current_lambda * self.lambda_update_factor)
        return max(self.lambda_min, current_lambda / self.lambda_update_factor)

    def _maybe_update_marginal_weight(self, model, measurements, measured_cliques):
        """Update marginal_weight using both ATE error and TVD error (dual criterion)."""
        if not self.adaptive_lambda or self.selection_mode != "ate":
            return

        ate_tolerance = self.ate_tolerance
        tvd_tolerance = self.tvd_tolerance
        if ate_tolerance is None or tvd_tolerance is None:
            derived_a, derived_l = self._derived_tolerances(measurements, measured_cliques)
            if ate_tolerance is None:
                ate_tolerance = derived_a
            if tvd_tolerance is None:
                tvd_tolerance = derived_l

        current_error, current_ates = self._ate_error_for_model(model, seed=42)
        tvd_error = self._tvd_error_for_model(model, measurements, measured_cliques)
        old_weight = self.marginal_weight
        self.marginal_weight = self._adjust_lambda(old_weight, tvd_error, current_error, tvd_tolerance, ate_tolerance)
        print(
            "Adaptive lambda: "
            f"ATE error={current_error:.6f} (tol={ate_tolerance:.4g}"
            f"{'' if self.ate_tolerance is not None else ', derived'}), "
            f"TVD error={tvd_error:.6f} (tol={tvd_tolerance:.4g}"
            f"{'' if self.tvd_tolerance is not None else ', derived'}), "
            f"marginal_weight {old_weight:.4f} -> {self.marginal_weight:.4f}"
        )

    def _model_to_dataframe(self, model, num_samples, seed=None):
        """Generate synthetic DataFrame from PGM model with optional seed for reproducibility.

        Args:
            model: Fitted PGM model.
            num_samples: Number of samples to generate.
            seed: Optional random seed for reproducibility.

        Returns:
            DataFrame: Synthetic data.
        """
        if seed is not None:
            np.random.seed(seed)
        synth = model.synthetic_data(rows=num_samples)
        return synth.df

    def _factor_to_weighted_df(self, factor, weight_col="_w"):
        """Flatten a PGM Factor into a one-row-per-cell DataFrame with a normalized weight column.

        The Factor's values are interpreted as joint masses; rows enumerate
        attribute cells in row-major (C) order — matching ``factor.values.ravel()``
        and ``itertools.product(*[range(s) for s in sizes])``. The weight column
        is normalized so it sums to 1.

        Args:
            factor: PGM Factor over the attributes of interest.
            weight_col: Name of the normalized-mass column on the output.

        Returns:
            DataFrame with one row per cell, columns = factor attributes + weight_col.
        """
        attrs = list(factor.domain.attrs)
        sizes = [factor.domain.size(a) for a in attrs]
        flat = np.asarray(factor.values).ravel()
        total = float(flat.sum())
        if total <= 0:
            raise ValueError("Cannot flatten an empty / zero-mass factor")
        probs = flat / total

        cells = list(itertools.product(*[range(s) for s in sizes]))
        out = {a: [c[i] for c in cells] for i, a in enumerate(attrs)}
        out[weight_col] = probs
        return pd.DataFrame(out)

    def _ate_from_model(self, model, config):
        """Compute FWL ATE on the model's joint distribution, no sampling.

        Projects the model to the joint over (T, Y, *Z), binarizes T and Y
        per the config, and runs claim_fwl_ate_from_distribution on the
        resulting weighted cell DataFrame.

        Only valid when self.ate_method == "fwl".
        """
        from fwl import claim_fwl_ate_from_distribution

        treatment = config["treatment"]
        outcome = config["outcome"]
        confounders = list(config["confounders"])
        cols = (treatment, outcome, *confounders)

        factor = model.project(cols)
        df_cells = self._factor_to_weighted_df(factor)

        # Preserve the per-cell weight across binarization (which only edits T/Y).
        df_binary = self._binarize_for_ate(df_cells, config)

        bounds = {k: tuple(v) for k, v in config["bounds"].items()}
        return claim_fwl_ate_from_distribution(
            df=df_binary,
            treatment_col=treatment,
            outcome_col=outcome,
            adjustment_set=confounders,
            bounds=bounds,
        )

    def _simulate_measurement(self, model, data, clique, measurements):
        """Simulate measuring a clique without actually adding noise.
        
        Returns a new model fitted with the additional measurement.
        Used to predict how measuring a particular clique would affect ATE.
        
        Args:
            model: Current fitted PGM model.
            data: Original Dataset.
            clique: Clique to simulate measuring.
            measurements: Current list of measurements.
            
        Returns:
            Model: New model fitted with the simulated measurement.
        """
        # Get true marginal (no noise since we're simulating)
        x = data.project(clique).datavector()
        
        # Small stddev = high confidence measurement
        temp_measurement = LinearMeasurement(x, clique, stddev=1e-10)
        
        # Copy and extend measurements
        sim_measurements = measurements.copy()
        sim_measurements.append(temp_measurement)
        
        # Warm start from current model
        pcliques = list(set(M.clique for M in sim_measurements))
        potentials = model.potentials.expand(pcliques)
        
        # Fit with fewer iterations for speed during simulation
        sim_model = estimation.mirror_descent(
            data.domain, 
            sim_measurements, 
            iters=min(self.max_iters, 500),
            potentials=potentials,
            callback_fn=lambda *_: None
        )
        
        return sim_model

    def _fallback_worst_approximated(self, candidates, answers, model, sigma):
        """Original L1-based selection as fallback (deterministic, no exponential mechanism).
        
        Args:
            candidates: Dict of candidate cliques with weights.
            answers: Dict of true marginals.
            model: Current fitted model.
            sigma: Noise standard deviation.
            
        Returns:
            tuple: Selected clique.
        """
        errors = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            xest = model.project(cl).datavector()
            errors[cl] = wgt * np.linalg.norm(x - xest, 1)
        
        # Deterministic: pick max error
        return max(errors, key=errors.get)

    def _compute_stat_term(self, candidates, answers, model, sigma):
        """Statistical term L_r(D) for the FWL/ATE path.

        Per claim_algorithm_fwl.tex (line:stat-term):
            L_r(D) = ||M_r(D) - M_r(p̂)||_1 - sqrt(2/π) · σ_t · n_r

        M_r(D) and M_r(p̂) are normalized to probability distributions
        (divided by their respective totals) so that L_r lives on the
        same O(1) scale as κ·A_r.

        Workload weights w_r are intentionally dropped (line:remove-wr); all
        candidates are weighted equally.

        Args:
            candidates: Dict of candidate cliques (values, the legacy w_r
                weights, are ignored here).
            answers: Dict of true marginals M_r(D) as count vectors.
            model: Current fitted model p̂.
            sigma: Current Gaussian-noise stddev σ_t.

        Returns:
            dict: {clique: L_r(D)}
        """
        scores = {}
        for cl in candidates:
            x = answers[cl]
            xest = model.project(cl).datavector()
            n_r = model.domain.size(cl)
            N = x.sum()
            bias = np.sqrt(2 / np.pi) * sigma * n_r / N
            # Normalize count vectors to probability distributions
            x_prob = x / N
            xest_prob = xest / xest.sum()
            scores[cl] = np.linalg.norm(x_prob - xest_prob, 1) - bias
        return scores

    def _compute_dynamic_mu(self, candidates, model, prev_model, sigma,
                            ate_scores, kappa, N, eta=None):
        """Compute the dynamic scale parameter μ_t.

        μ_t adaptively calibrates the causal term's influence so that it
        operates on the same empirical scale as the statistical term,
        regardless of how both evolve across iterations.

        Per claim_algorithm_fwl.tex (line:scaling):
            μ_t = median_r |ΔL_r_proxy| / (κ · median_r |A_r| + η)

        The numerator uses the model-to-model L1 change as a proxy for the
        current spread of L_r values.  Both numerator and denominator are
        computed from public (model-derived) quantities only, so μ_t does
        not increase the DP sensitivity of q_r.

        Args:
            candidates: Dict of candidate cliques.
            model: Current fitted model p̂_{t-1}.
            prev_model: Previous fitted model p̂_{t-2} (None at t=1).
            sigma: Current Gaussian-noise stddev σ_t.
            ate_scores: Dict {clique: A_r(D)} already computed.
            kappa: The κ scaling factor.
            N: Dataset size (for bias normalization).
            eta: Small constant to prevent division by zero.

        Returns:
            float: μ_t (falls back to 10.0 when prev_model is None).
        """
        if prev_model is None:
            return 10.0

        # Numerator: median_r ||| M_r(p̂_{t-1}) - M_r(p̂_{t-2}) ||_1 - bias/N |
        l_proxy_values = []
        for cl in candidates:
            xest_curr = model.project(cl).datavector()
            xest_prev = prev_model.project(cl).datavector()
            n_r = model.domain.size(cl)
            bias = np.sqrt(2 / np.pi) * sigma * n_r / N
            # Both are model distributions; normalize to probability scale
            p_curr = xest_curr / xest_curr.sum()
            p_prev = xest_prev / xest_prev.sum()
            l_proxy = abs(np.linalg.norm(p_curr - p_prev, 1) - bias)
            l_proxy_values.append(l_proxy)

        # Denominator: κ · median_r |A_r| + η
        a_abs_values = [abs(ate_scores.get(cl, 0.0)) for cl in candidates]

        median_l = np.median(l_proxy_values)
        median_a = np.median(a_abs_values)

        if eta is None:
            eta = self.mu_eta
        mu_t = median_l / (kappa * median_a + eta)

        print(f"  Dynamic μ_t={mu_t:.4f}  (median|ΔL_proxy|={median_l:.6f}, "
              f"median|A_r|={median_a:.6f}, κ={kappa:.4f})")

        return mu_t

    def worst_ate_approximated(
        self, candidates, answers, data, model, prev_model, measurements,
        epsilon, sigma
    ):
        """Select marginal r_t via the exponential mechanism with quality q_r(D).

        Implements claim_algorithm_fwl.tex line:quality-score:
            q_r(D) = [λ · L_r(D) + (1-λ) · μ_t · κ · A_r(D)] / [λ + (1-λ) · μ_t]
        with λ = self.marginal_weight, κ = self._fwl_kappa, μ_t from
        _compute_dynamic_mu, L_r from _compute_stat_term, and A_r from
        per-ATE short-circuits + sampling-free evaluation.

        Args:
            candidates: Dict of candidate cliques (workload weights ignored).
            answers: Dict of true marginals M_r(D) as count vectors.
            data: Original Dataset.
            model: Current fitted PGM model p̂_{t-1}.
            prev_model: Previous fitted model p̂_{t-2} (None at t=1).
            measurements: Current list of measurements (for refits).
            epsilon: ε_t for the exponential mechanism.
            sigma: σ_t for the bias-correction term in L_r.

        Returns:
            tuple: Selected clique.
        """
        # Use cached true ATEs
        true_ates = self._true_ates
        
        # Step 1: Compute statistical term L_r(D) (fast, no simulation)
        l1_scores = self._compute_stat_term(candidates, answers, model, sigma)
        
        # Step 2: Compute causal term A_r(D)
        # A_r(D) = Σ β_i [|τ_i* - τ_i(p̂_{t-1})|  -  |τ_i* - τ̂_i^r(D)|]
        # All ATEs are evaluated directly on PGM marginals — no Monte Carlo.
        ate_scores = {}

        # Current model ATEs τ_i(p̂_{t-1}) — sampling-free
        current_ates = {
            cfg["name"]: self._ate_from_model(model, cfg)
            for cfg in self.ate_configs
        }
        current_error = self.compute_weighted_ate_error(true_ates, current_ates)

        # Print current status for each ATE
        print(f"Current weighted ATE error: {current_error:.6f}")
        print(f"Hybrid selection: marginal_weight={self.marginal_weight:.2f}")
        for config in self.ate_configs:
            name = config["name"]
            alpha = config["alpha"]
            true_val = true_ates[name]
            model_val = current_ates[name]
            print(f"  {name} (α={alpha}): true={true_val:.4f}, model={model_val:.4f}, error={abs(true_val - model_val):.4f}")

        # Pre-compute per-ATE attribute sets so we can short-circuit per candidate.
        ate_attr_sets = [
            (cfg, {cfg["treatment"], cfg["outcome"], *cfg["confounders"]})
            for cfg in self.ate_configs
        ]

        for cl in candidates:
            cl_set = set(cl)
            # Classify each ATE wrt the candidate clique:
            #   "supset"   — cl ⊇ ATE's (T,Y,Z): selecting cl reveals the joint, τ̂^r = τ*
            #   "disjoint" — cl ∩ (T,Y,Z) = ∅: cl carries no info about τ_i, τ̂^r = τ(p̂_{t-1})
            #   "partial"  — partial overlap: requires a refit to evaluate τ̂^r
            relations = []
            needs_refit = False
            for cfg, ate_set in ate_attr_sets:
                if ate_set <= cl_set:
                    relations.append(("supset", cfg))
                elif cl_set & ate_set:
                    relations.append(("partial", cfg))
                    needs_refit = True
                else:
                    relations.append(("disjoint", cfg))

            sim_model = None
            try:
                if needs_refit:
                    # TODO(DP): _simulate_measurement reads data.project(cl) with
                    # stddev=1e-10. This raw-marginal access is non-DP and is
                    # tracked separately in the DP plan.
                    sim_model = self._simulate_measurement(model, data, cl, measurements)

                simulated_ates = {}
                for relation, cfg in relations:
                    name = cfg["name"]
                    if relation == "supset":
                        simulated_ates[name] = true_ates[name]
                    elif relation == "disjoint":
                        simulated_ates[name] = current_ates[name]
                    else:  # partial
                        simulated_ates[name] = self._ate_from_model(sim_model, cfg)

                simulated_error = self.compute_weighted_ate_error(true_ates, simulated_ates)
                ate_scores[cl] = current_error - simulated_error

            except Exception as e:
                print(f"  Candidate {cl}: failed ({e})")
                ate_scores[cl] = 0.0
            finally:
                if sim_model is not None:
                    del sim_model
                    gc.collect()

        # Clear JAX JIT cache once after all candidates are scored, not per-candidate.
        jax.clear_caches()
        
        # Step 3: Compute dynamic μ_t and combine scores.
        # q_r(D) = [λ · L_r + (1-λ) · μ_t · κ · A_r] / [λ + (1-λ) · μ_t]
        # μ_t is data-independent (model-derived), so it does not affect DP.
        kappa = self._fwl_kappa
        N = data.records
        mu_t = self._compute_dynamic_mu(
            candidates, model, prev_model, sigma, ate_scores, kappa, N
        )
        norm_factor = self.marginal_weight + mu_t * (1 - self.marginal_weight)
        combined_scores = {}
        for cl in candidates:
            l1_score = l1_scores.get(cl, 0.0)
            ate_score = ate_scores.get(cl, 0.0)
            combined_scores[cl] = (
                self.marginal_weight * l1_score
                + mu_t * (1 - self.marginal_weight) * kappa * ate_score
            ) / norm_factor

        # Step 4: Exponential mechanism on q_r(D).
        # Sensitivity bound:
        #   Δq = [λ·ΔL_r + (1-λ)·μ_t·κ·ΔA_r] / [λ + (1-λ)·μ_t]
        #   Since μ_t is data-independent, Δq ≤ 2/N regardless of μ_t.
        # TODO(DP): ΔA_r is not bounded today. The FWL estimator τ̂ has
        # data-dependent sensitivity (driven by per-cell propensity supports
        # and v). Treating ΔA_r as 2/N here is a placeholder; calibrating
        # it properly requires either row clipping on τ̂ or a public-bound
        # substitute for the per-cell terms, both deferred per the plan.
        delta_l = 2.0 / N
        delta_a = 2.0 / N
        sensitivity = (
            self.marginal_weight * delta_l
            + mu_t * (1 - self.marginal_weight) * abs(kappa) * delta_a
        ) / norm_factor

        best_candidate = self.exponential_mechanism(
            combined_scores, epsilon, sensitivity
        )
        best_combined = combined_scores[best_candidate]
        best_l1 = l1_scores.get(best_candidate, 0.0)
        best_ate = ate_scores.get(best_candidate, 0.0)

        print(
            f"Selected {best_candidate} (EM, ε={epsilon:.3f}, Δq={sensitivity:.3f}): "
            f"L_r={best_l1:.3f}, A_r={best_ate:.3f}, κ={kappa:.3f}, "
            f"μ_t={mu_t:.3f}, q_r={best_combined:.3f}"
        )
        
        return best_candidate


    def worst_approximated(self, candidates, answers, model, eps, sigma):
        """Original L1-based selection using the exponential mechanism.
        
        Args:
            candidates: Dict of candidate cliques with weights.
            answers: Dict of true marginals.
            model: Current fitted model.
            eps: Epsilon for exponential mechanism.
            sigma: Noise standard deviation.
            
        Returns:
            tuple: Selected clique.
        """
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def _release_private_reference_ates(self, data):
        """Auto-release a privately DP-noised ATE for each config from a slice of rho."""
        sensitivity = self.ate_sensitivity
        if sensitivity is None:
            lo, hi = self.ate_outcome_range
            sensitivity = (hi - lo) / data.records
        
        ate_rho_total = self.reference_ate_rho_fraction * self.rho
        if not self.ate_configs:
            return {}
            
        ate_rho_each = ate_rho_total / len(self.ate_configs)
        sigma_ate = sensitivity / np.sqrt(2 * ate_rho_each)
        
        # Store for derived tolerances
        self.sigma_ate = float(sigma_ate)
        
        raw_ates = self.compute_all_ates(data.df)
        private_ates = {}
        for config in self.ate_configs:
            name = config["name"]
            raw_val = raw_ates[name]
            private_val = float(raw_val + self.gaussian_noise(sigma_ate, 1)[0])
            private_ates[name] = private_val
            print(
                f"Auto-released private reference ATE for {name}: "
                f"{private_val:.6f} (raw={raw_val:.6f}, sensitivity={sensitivity:.6g}, "
                f"rho spent={ate_rho_each:.6g}, sigma={sigma_ate:.6g})"
            )
            
        self.rho -= ate_rho_total
        print(f"Total rho spent on reference ATEs: {ate_rho_total:.6g}, remaining rho: {self.rho:.6g}")
        return private_ates

    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        # Cache reference ATEs at the start if using ATE mode.
        if self.selection_mode == "ate":
            if self.reference_ates is None:
                self._true_ates = self._release_private_reference_ates(data)
            else:
                self._true_ates = self.reference_ates
                
            print("Reference ATEs:")
            for config in self.ate_configs:
                name = config["name"]
                alpha = config["alpha"]
                print(f"  {name} (α={alpha}): {self._true_ates[name]:.6f}")

            # κ is needed by worst_ate_approximated regardless of ate_method,
            # because _ate_from_model always uses FWL regression from the model.
            # TODO(DP): v_i and κ are derived from raw D and currently used
            # without DP accounting. See claim_algorithm_fwl.tex for the κ
            # role in q_r(D); a public-bound or noised substitute is needed
            # before the FWL path is fully DP.
            self._cache_fwl_components(data.df)
            if self.ate_method == "fwl":
                print("FWL components:")
                for config in self.ate_configs:
                    name = config["name"]
                    print(f"  v[{name}] = {self._fwl_v[name]:.6f}")
            print(f"  κ = {self._fwl_kappa:.6f}")

        rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        if not initial_cliques:
            initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]  # use one-way marginals

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []
        print("Initial Sigma", sigma)
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in initial_cliques:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))

        zeros = self.structural_zeros
        # NOTE: Haven't incorproated structural zeros back yet after refactoring
        model = estimation.mirror_descent(
                data.domain, measurements, iters=self.max_iters, callback_fn=lambda *_: None
        )

        t = 0
        terminate = False
        prev_model = None  # p̂_{t-2}; None at first iteration
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            print('Budget Used', rho_used, '/', self.rho)
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            
            # Branch on selection mode
            if self.selection_mode == "ate":
                cl = self.worst_ate_approximated(
                    small_candidates, answers, data, model, prev_model,
                    measurements, epsilon, sigma,
                )
            else:
                cl = self.worst_approximated(
                    small_candidates, answers, model, epsilon, sigma
                )
            print('Measuring Clique', cl)
            n = data.domain.size(cl)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            z = model.project(cl).datavector()

            # Warm start potentials from prior round
            # TODO: check if it helps to call maximal_subsets here
            pcliques = list(set(M.clique for M in measurements))
            potentials = model.potentials.expand(pcliques)
            prev_model = model  # store p̂_{t-1} as p̂_{t-2} for next iteration
            model = estimation.mirror_descent(
                    data.domain, measurements, iters=self.max_iters, potentials=potentials, callback_fn=lambda *_: None
            )
            
            # Optional adaptive lambda update (dual criterion: ATE + TVD).
            measured_cliques = list(dict.fromkeys(M.clique for M in measurements))
            self._maybe_update_marginal_weight(model, measurements, measured_cliques)

            w = model.project(cl).datavector()
            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
                sigma /= 2
                epsilon *= 2

        print("Generating Data...")
        model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters, potentials=potentials
        )
        synth = model.synthetic_data(rows=num_synth_rows)

        return model, synth


def _epsilon_for_rho(rho, delta):
    """Return an epsilon whose cdp_rho(epsilon, delta) is approximately rho."""
    if rho <= 0:
        return 0.0
    return cdp_eps(rho, delta)

def pilot_select_lambda(
    data,
    workload,
    epsilon,
    delta,
    lambda_values=(0.1, 0.5, 0.9),
    pilot_fraction=0.15,
    pilot_samples=5,
    num_synth_rows=None,
    **claim_kwargs,
):
    """Select lambda with cheap pilot CLAIM runs, then run the final CLAIM."""
    if not lambda_values:
        raise ValueError("lambda_values must contain at least one value")
    if not (0.0 < pilot_fraction < 1.0):
        raise ValueError("pilot_fraction must be in (0, 1)")

    total_rho = cdp_rho(epsilon, delta)
    pilot_rho_each = total_rho * pilot_fraction / len(lambda_values)
    final_rho = total_rho * (1.0 - pilot_fraction)
    pilot_epsilon = _epsilon_for_rho(pilot_rho_each, delta)
    final_epsilon = _epsilon_for_rho(final_rho, delta)

    base_claim_kwargs = dict(claim_kwargs)
    base_claim_kwargs.pop("marginal_weight", None)
    base_claim_kwargs.pop("adaptive_lambda", None)

    scores = {}
    pilot_outputs = {}
    for lam in lambda_values:
        print(f"\\n[Pilot marginal_weight={lam}] budget rho={pilot_rho_each:.6g}")
        mech = CLAIM(
            pilot_epsilon,
            delta,
            marginal_weight=lam,
            adaptive_lambda=False,
            **base_claim_kwargs,
        )
        model, synth = mech.run(data, workload, num_synth_rows=num_synth_rows)
        
        errors = []
        # Pre-compute true ATEs
        true_ates = mech._true_ates
        if not true_ates:
            true_ates = mech.compute_all_ates(data.df)

        for b in range(pilot_samples):
            rows = num_synth_rows or data.records
            pilot_df = model.synthetic_data(rows=rows).df
            pilot_ates = mech.compute_all_ates(pilot_df)
            pilot_error = mech.compute_weighted_ate_error(true_ates, pilot_ates)
            errors.append(pilot_error)
        
        avg_error = float(np.mean(errors))
        scores[lam] = avg_error
        pilot_outputs[lam] = (model, synth)
        print(f"[Pilot marginal_weight={lam}] average weighted ATE error={avg_error:.6f}")

    selected_lambda = min(scores, key=scores.get)
    print(f"\\nSelected marginal_weight={selected_lambda} with pilot ATE error={scores[selected_lambda]:.6f}")
    print(f"[Final marginal_weight={selected_lambda}] budget rho={final_rho:.6g}")

    final_mech = CLAIM(
        final_epsilon,
        delta,
        marginal_weight=selected_lambda,
        adaptive_lambda=False,
        **base_claim_kwargs,
    )
    final_model, final_synth = final_mech.run(
        data, workload, num_synth_rows=num_synth_rows
    )
    return selected_lambda, scores, final_model, final_synth


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params["dataset"] = "../data/adult.csv"
    params["domain"] = "../data/adult-domain.json"
    params["epsilon"] = 1.0
    params["delta"] = 1e-9
    params["noise"] = "laplace"
    params["max_model_size"] = 80
    params["max_iters"] = 1000
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000
    # Selection mode: "marginal" (L1-based) or "ate" (ATE-based)
    params["selection_mode"] = "marginal"
    params["mu_eta"] = 1e-6
    params["kappa_eta"] = 1e-6
    params["adaptive_lambda"] = True
    params["lambda_min"] = 0.1
    params["lambda_max"] = 0.9
    params["ate_tolerance"] = None
    params["tvd_tolerance"] = None
    params["reference_ates"] = None
    params["ate_sensitivity"] = None
    params["reference_ate_rho_fraction"] = 0.05
    params["ate_outcome_min"] = -1.0
    params["ate_outcome_max"] = 1.0
    params["fixed_lambda_from_list"] = False
    params["lambda_values"] = "0.1,0.5,0.9"
    params["pilot_fraction"] = 0.15
    params["pilot_samples"] = 5
    # Causal parameters (required when selection_mode="ate")
    params["ate_configs"] = None  # Path to JSON file with ATE configs
    params["causal_graph"] = None
    # Hybrid selection weight: 0.0 = pure ATE, 1.0 = pure marginal
    params["marginal_weight"] = 0.3
    # ATE estimation backend: "dowhy" or "fwl"
    params["ate_method"] = "dowhy"

    return params


if __name__ == "__main__":
    import json

    description = "CLAIM: Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data"
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--max_model_size", type=float, help="maximum size (in megabytes) of model"
    )
    parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
    parser.add_argument("--degree", type=int, help="degree of marginals in workload")
    parser.add_argument(
        "--num_marginals", type=int, help="number of marginals in workload"
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        help="maximum number of cells for marginals in workload",
    )
    parser.add_argument("--save", type=str, help="path to save synthetic data")
    
    # Selection mode arguments
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["marginal", "ate"],
        help="Selection mode: 'marginal' (L1-based) or 'ate' (ATE-based)"
    )
    
    import argparse
    parser.add_argument(
        "--mu_eta",
        type=float,
        help="Small positive constant preventing division by zero in mu"
    )
    parser.add_argument(
        "--kappa_eta",
        type=float,
        help="Small positive constant preventing division by zero in kappa"
    )
    parser.add_argument(
        "--adaptive_lambda",
        action=argparse.BooleanOptionalAction,
        help="Adapt marginal_weight during one CLAIM run using the current model ATE error"
    )
    parser.add_argument("--lambda_min", type=float, help="Minimum adaptive lambda")
    parser.add_argument("--lambda_max", type=float, help="Maximum adaptive lambda")
    parser.add_argument(
        "--ate-tolerance",
        "--ate_tolerance",
        dest="ate_tolerance",
        type=float,
        help="ATE-error tolerance for adaptive lambda"
    )
    parser.add_argument(
        "--tvd-tolerance",
        "--tvd_tolerance",
        dest="tvd_tolerance",
        type=float,
        help="TVD-error tolerance for adaptive lambda"
    )
    parser.add_argument(
        "--ate_sensitivity",
        type=float,
        help="Assumed L2 sensitivity of the ATE estimator."
    )
    parser.add_argument(
        "--reference_ate_rho_fraction",
        type=float,
        help="Fraction of total zCDP budget spent auto-releasing a private reference ATE"
    )
    parser.add_argument(
        "--ate_outcome_min",
        type=float,
        help="Minimum value the outcome column is clipped to before ATE computation"
    )
    parser.add_argument(
        "--ate_outcome_max",
        type=float,
        help="Maximum value the outcome column is clipped to before ATE computation"
    )
    parser.add_argument(
        "--fixed_lambda_from_list",
        action="store_true",
        help="Select lambda from --lambda_values using small pilot runs before the final run"
    )
    parser.add_argument(
        "--lambda_values",
        type=str,
        help="Comma-separated lambda grid for pilot selection, e.g. 0.1,0.5,0.9"
    )
    parser.add_argument(
        "--pilot_fraction",
        type=float,
        help="Fraction of zCDP budget spent on all pilot lambda runs"
    )
    parser.add_argument(
        "--pilot_samples",
        type=int,
        help="Number of synthetic samples used to average pilot ATE error"
    )

    # Multi-ATE configuration (required when selection_mode="ate")
    parser.add_argument(
        "--ate_configs",
        type=str,
        help="Path to JSON file with ATE configurations. Each config should have: "
             "name, treatment, outcome, confounders (list), alpha (weight in [0,1])"
    )
    parser.add_argument(
        "--causal_graph",
        type=str,
        help="Path to GML file with causal graph (required for ATE mode)"
    )
    parser.add_argument(
        "--marginal_weight",
        type=float,
        help="Weight for L1 marginal error in hybrid selection (0.0 = pure ATE, 1.0 = pure marginal)."
             " Only used when selection_mode='ate'. Default: 0.3"
    )
    parser.add_argument(
        "--ate_method",
        type=str,
        choices=["dowhy", "fwl"],
        help="ATE estimation backend: 'dowhy' (CausalModel + backdoor) or 'fwl' "
             "(claim_fwl_ate_estimator from fwl.py). 'fwl' requires per-config 'bounds'."
    )

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    
    # Load ATE configs from JSON file if provided
    ate_configs_list = None
    if args.ate_configs is not None:
        with open(args.ate_configs, 'r') as f:
            ate_configs_list = json.load(f)

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        prng = np.random
        workload = [
            workload[i]
            for i in prng.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    
    adaptive_lambda = args.adaptive_lambda and not args.fixed_lambda_from_list

    claim_kwargs = dict(
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
        selection_mode=args.selection_mode,
        ate_configs=ate_configs_list,
        causal_graph_path=args.causal_graph,
        marginal_weight=args.marginal_weight,
        ate_method=args.ate_method,
        mu_eta=args.mu_eta,
        kappa_eta=args.kappa_eta,
        adaptive_lambda=adaptive_lambda,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        ate_tolerance=args.ate_tolerance,
        tvd_tolerance=args.tvd_tolerance if hasattr(args, 'tvd_tolerance') else None,
        reference_ates=None,
        ate_sensitivity=args.ate_sensitivity,
        reference_ate_rho_fraction=args.reference_ate_rho_fraction,
        ate_outcome_range=(args.ate_outcome_min, args.ate_outcome_max),
    )

    selected_lambda = None
    lambda_scores = None
    if args.fixed_lambda_from_list:
        lambda_values = [float(x.strip()) for x in args.lambda_values.split(',') if x.strip()]
        selected_lambda, lambda_scores, model, synth = pilot_select_lambda(
            data,
            workload,
            args.epsilon,
            args.delta,
            lambda_values=lambda_values,
            pilot_fraction=args.pilot_fraction,
            pilot_samples=args.pilot_samples,
            **claim_kwargs,
        )
    else:
        mech = CLAIM(args.epsilon, args.delta, **claim_kwargs)
        model, synth = mech.run(data, workload)

    if args.save is not None:
        synth.df.to_csv(args.save, index=False)

    # Print ATE comparison if in ATE mode
    if args.selection_mode == "ate":
        true_ates = mech._true_ates
        synth_ates = mech.compute_all_ates(synth.df)
        print("\nATE Comparison:")
        total_weighted_error = 0.0
        for config in mech.ate_configs:
            name = config["name"]
            alpha = config["alpha"]
            true_val = true_ates[name]
            synth_val = synth_ates[name]
            error = abs(true_val - synth_val)
            weighted_error = alpha * error
            total_weighted_error += weighted_error
            print(f"  {name} (α={alpha}): true={true_val:.6f}, synth={synth_val:.6f}, error={error:.6f}")
        print(f"  Weighted Total Error: {total_weighted_error:.6f}")

    if selected_lambda is not None:
        print(f"Selected Lambda: {selected_lambda}")
        print(f"Pilot Lambda Scores: {lambda_scores}")

    # Print marginal errors
    synth_errors = []
    model_errors = []
    for proj, wgt in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        Z = model.project(proj).datavector()
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        synth_errors.append(e)
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Z / Z.sum(), 1)
        model_errors.append(e)
    print("Average Marginal Error: ", np.mean(model_errors), np.mean(synth_errors))


