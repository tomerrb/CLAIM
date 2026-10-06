"""Implementation of CLAIM: Causally-Learned Adaptive and Iterative Mechanism for DP Synthetic Data.

This implementation supports two selection modes:
- "marginal" (default): Original L1-based selection using the exponential mechanism.
  Selects the worst-approximated marginal based on L1 distance.
- "ate": causal selection with FWL scores on complete query marginals,
  or the separate legacy DoWhy backend.

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
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AIM_Implementation/mechanisms')))
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

def _to_frame(ds):
    """pandas view of an mbi Dataset (the new mbi has no Dataset.df)."""
    return pd.DataFrame({a: np.asarray(v) for a, v in ds.to_dict().items()})
from aim import powerset, downward_closure, compile_workload, filter_candidates
from aim import default_params as aim_default_params

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
        max_candidate_arity: Candidate marginal arity ell, at least the size
            of the largest causal query (default: that query size).
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
            None = the released record count n_tilde (FWL path).
        sigma_n: Noise scale for the released record count.
        reference_ate_rho: Absolute zCDP budget for the reference ATE release.
        ate_method: "fwl" (selection on all marginals up to the candidate arity,
            default) or
            "dowhy" (the legacy causal selection backend).
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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        selection_mode=None,
        #################
        ate_configs=None,
        causal_graph_path=None,
        ate_sample_size=10000,
        sim_sample_size=5000,
        marginal_weight=0.3,
        ############ CHANGED TO MATCH PSEUDOCODE ############
        ate_method="fwl",
        #################
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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        overlap_eta=0.01,
        n_min=1,
        sigma_n=None,
        max_candidate_arity=None,
        reference_ate_rho=None,
        #################
    ):
        ############ CHANGED TO MATCH PSEUDOCODE ############
        super(CLAIM, self).__init__(
            epsilon, delta, prng=np.random if prng is None else prng
        )
        self._initial_rho = self.rho
        #################
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

        # Selection mode configuration
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if selection_mode is None:
            selection_mode = "ate" if ate_configs else "marginal"
        #################
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
        # In the FWL path these come from the current PGM each round.
        # The legacy DoWhy path retains its original empirical scaling method.
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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        self.reference_ate_rho = (None if reference_ate_rho is None
                                  else float(reference_ate_rho))
        if max_candidate_arity is not None and (
                isinstance(max_candidate_arity, bool)
                or not isinstance(max_candidate_arity, (int, np.integer))):
            raise ValueError("max_candidate_arity must be an integer")
        self.max_candidate_arity = (None if max_candidate_arity is None
                                    else int(max_candidate_arity))
        self._fwl_reference_rho = None
        self.sigma_n = None if sigma_n is None else float(sigma_n)
        self.overlap_eta = float(overlap_eta)
        self.n_min = int(n_min)
        self._fwl_tilde_n = None
        self._reference_release = {}
        #################
        self.ate_outcome_range = (float(ate_outcome_range[0]), float(ate_outcome_range[1]))
        self.mu_eta = float(mu_eta)
        self.kappa_eta = float(kappa_eta)
        self.reference_ates = reference_ates

        ############ CHANGED TO MATCH PSEUDOCODE ############
        if not 0 < self.overlap_eta <= 0.25 or self.n_min < 1:
            raise ValueError("FWL requires overlap_eta in (0, 1/4] and n_min >= 1")
        if selection_mode == "ate" and ate_method == "fwl":
            if self._initial_rho <= 0:
                raise ValueError("FWL selection requires a positive zCDP budget")
            if self.marginal_weight <= 0:
                raise ValueError("FWL selection requires marginal_weight > 0")
            if reference_ates is not None:
                raise ValueError("FWL selection releases its own private reference ATEs")
            if self.sigma_n is None:
        #################
                # Default to spending 1% of rho on the noisy count.  A supplied
                # sigma_n is used directly, as in the pseudocode.
                ############ CHANGED TO MATCH PSEUDOCODE ############
                self.sigma_n = np.sqrt(1 / (2 * 0.01 * self._initial_rho))
            if not np.isfinite(self.sigma_n) or self.sigma_n <= 0:
                raise ValueError("FWL sigma_n must be finite and positive")
            self._fwl_reference_rho = (
                self.reference_ate_rho if self.reference_ate_rho is not None
                else self.reference_ate_rho_fraction * self._initial_rho
            )
            if not np.isfinite(self._fwl_reference_rho) or self._fwl_reference_rho <= 0:
                raise ValueError("FWL reference release requires a positive budget")
            if (1 / (2 * self.sigma_n**2)
                    + self._fwl_reference_rho
                    >= 0.1 * self._initial_rho):
                raise ValueError("FWL count and reference releases must leave AIM's 10% selection budget")
                #################
        
        if not (0.0 <= self.marginal_weight <= 1.0):
            raise ValueError(f"marginal_weight must be in [0, 1], got {self.marginal_weight}")
        if not (0.0 <= self.lambda_min <= self.lambda_max <= 1.0):
            raise ValueError("lambda_min/lambda_max must satisfy 0 <= min <= max <= 1")
        if self.adaptive_lambda and self.lambda_min <= 0:
            raise ValueError("adaptive lambda requires lambda_min > 0")
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if (selection_mode == "ate" and ate_method == "fwl" and self.adaptive_lambda
                and not self.lambda_min <= self.marginal_weight <= self.lambda_max):
            raise ValueError("FWL initial lambda must lie within its adaptive bounds")
        #################

        if selection_mode == "ate":
            if not ate_configs:
                raise ValueError("ATE mode requires ate_configs (list of ATE configurations)")
            if ate_method == "dowhy" and not causal_graph_path:
                raise ValueError("ATE mode with ate_method='dowhy' requires causal_graph_path")
            ############ CHANGED TO MATCH PSEUDOCODE ############
            if ate_method == "fwl" and len({cfg.get("name") for cfg in ate_configs}) != len(ate_configs):
                raise ValueError("FWL query names must be unique")
            #################

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
          - "fwl":   ordinary adjusted ATE for evaluation and AdaptLambda;
                     selection evaluates the FWL surrogate separately.

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
            ############ CHANGED TO MATCH PSEUDOCODE ############
            from fwl import claim_adjusted_ate
            #################

            bounds = {k: tuple(v) for k, v in config["bounds"].items()}
            ############ CHANGED TO MATCH PSEUDOCODE ############
            return claim_adjusted_ate(
                df=df_binary,
                treatment_col=treatment,
                outcome_col=outcome,
                adjustment_set=list(config["confounders"]),
                bounds=bounds,
                n_min=self.n_min,
            )
            #################

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
        """Empirical variance kept only for the pre-existing legacy DoWhy path.
        """
        ############ CHANGED TO MATCH PSEUDOCODE ############
        from fwl import legacy_empirical_variance
        #################

        df_binary = self._binarize_for_ate(df, config)
        bounds = {k: tuple(v) for k, v in config["bounds"].items()}
        ############ CHANGED TO MATCH PSEUDOCODE ############
        return legacy_empirical_variance(
            df=df_binary,
            treatment_col=config["treatment"],
            adjustment_set=list(config["confounders"]),
            bounds=bounds,
        )
        #################

    def _cache_fwl_components(self, df):
        """Legacy DoWhy-only scaling; FWL obtains these from the PGM.

        κ = 1 / Σ_i (β_i / v_i), where β_i is the per-ATE weight (config['alpha']).
        Called once at the start of run() only in the legacy DoWhy path.
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

    def _model_sample_size(self):
        """Rows sampled from the model for its ATE: sim_sample_size, or the
        released record count n_tilde when sim_sample_size is None, so the
        sample is as sparse as the data (5000 if no count was released)."""
        ############ MODEL SAMPLE SIZE = n_tilde ############
        if self.sim_sample_size is not None:
            return int(self.sim_sample_size)
        return int(self._fwl_tilde_n) if self._fwl_tilde_n else 5000
        #################

    def _ate_error_for_model(self, model, seed=42):
        """Compute absolute ATE error for the current PGM model."""
        if not self.ate_configs:
            raise ValueError("ATE configs required for ATE error computation")
        
        ############ PRIVATE FWL REFERENCE (TWO NOISY SUMS) ############
        # AdaptLambda compares FWL with FWL: the model's FWL effect on m_s
        # model-generated tuples, with the same eta floor as the reference.
        df = self._model_to_dataframe(model, self._model_sample_size(), seed=seed)
        if self.ate_method == "fwl":
            current_ates = {}
            for cfg in self.ate_configs:
                n_sum, v_sum = self._fwl_ratio_sums(df, cfg)
                current_ates[cfg["name"]] = float(np.clip(
                    n_sum / max(v_sum, self.overlap_eta * len(df)), -1, 1))
        else:
            current_ates = self.compute_all_ates(df)
        #################
            
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
            ############ CHANGED TO MATCH PSEUDOCODE ############
            if self.ate_method == "fwl" and self._fwl_tilde_n:
                total += 0.5 * np.linalg.norm(
                    y / self._fwl_tilde_n - xest / xest.sum(), 1
                )
            else:
                total += 0.5 * np.linalg.norm(y - xest, 1) / max(y.sum(), 1e-12)
            #################
        return total / len(used_cliques)

    def _derived_tolerances(self, measurements, cliques):
        """Noise-floor-derived tolerances (theta_A, theta_L), fully public.

        The principle: a tolerance below the error a *perfect* model would
        exhibit is incoherent -- the schedule would chase pure noise. So each
        tolerance is z times its perfect-model noise floor:

        - For FWL, theta_A is z times the weighted expected absolute noise
          in the released reference ATEs, plus 1/sqrt(sim_sample_size).
          Each query's noise scale is the delta-method sd of the released
          ratio, sigma * sqrt(1 + tau^2) / max(V_noised, eta * n_tilde),
          capped at the public bound 1. The legacy route retains its existing
          tolerance.
        - theta_L = z * mean_r sqrt(2/pi) * sigma_r * n_r / (2 * sum(y_r)),
          the expected TVD between a perfect model and the *noisy* released
          one-way marginals (E|N(0,s)| = s * sqrt(2/pi) per cell), using the
          released noisy total as the public stand-in for N.

        z = TOLERANCE_Z = 2 scales the expected noise level.
        Every ingredient is part of the DP transcript, so deriving the
        tolerances is post-processing.
        """
        lo, hi = self.ate_outcome_range
        sigma_mc = (hi - lo) / np.sqrt(self._model_sample_size())
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if self.ate_method == "fwl":
            ############ PRIVATE FWL REFERENCE (TWO NOISY SUMS) ############
            expected_reference_error = 0.0
            floor = self.overlap_eta * self._fwl_tilde_n
            for config in self.ate_configs:
                _, noised_v, reference = self._reference_release[config["name"]]
                sd = min(self.sigma_ate * np.sqrt(1 + reference**2)
                         / max(noised_v, floor), 1.0)
                expected_reference_error += (
                    config["alpha"] * np.sqrt(2 / np.pi) * sd
                )
            #################
            theta_a = self.TOLERANCE_Z * (
                expected_reference_error + 1 / np.sqrt(self._model_sample_size())
            )
        else:
            theta_a = self.TOLERANCE_Z * (self.sigma_ate + sigma_mc)
        #################

        by_clique = {m.clique: m for m in measurements if m.clique in cliques}
        floors = []
        for cl in cliques:
            if cl not in by_clique:
                continue
            m = by_clique[cl]
            y = m.noisy_measurement
            ############ CHANGED TO MATCH PSEUDOCODE ############
            total = (
                self._fwl_tilde_n if self.ate_method == "fwl" and self._fwl_tilde_n
                else max(float(np.sum(y)), 1e-12)
            )
            #################
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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if self.ate_method == "fwl":
            measured_cliques = [cl for cl in measured_cliques if len(cl) == 1]
        #################

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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if seed is None:
            synth = model.synthetic_data(rows=num_samples)
        else:
        #################
            # Model sampling is post-processing.  Keep its reproducible seed
            # separate from the RNG stream used for subsequent DP releases.
            ############ CHANGED TO MATCH PSEUDOCODE ############
            private_rng_state = np.random.get_state()
            try:
                np.random.seed(seed)
                synth = model.synthetic_data(rows=num_samples)
            finally:
                np.random.set_state(private_rng_state)
            #################
        return _to_frame(synth)

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
        attrs = list(factor.domain.attributes)
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

        ############ CHANGED TO MATCH PSEUDOCODE ############
        weights, df_binary, bounds = self._fwl_model_components(model, config)
        return claim_fwl_ate_from_distribution(
            df=df_binary,
            treatment_col=config["treatment"],
            outcome_col=config["outcome"],
            adjustment_set=list(config["confounders"]),
            bounds=bounds,
            weights=weights,
        )
        #################

    def _fwl_model_components(self, model, config):
        """Freeze the model treatment probabilities and variance for one round."""
        ############ CHANGED TO MATCH PSEUDOCODE ############
        from fwl import model_fwl_weights
        #################

        ############ CHANGED TO MATCH PSEUDOCODE ############
        treatment = config["treatment"]
        cols = (treatment, config["outcome"], *config["confounders"])
        factor = model.project(cols)
        cells = self._binarize_for_ate(self._factor_to_weighted_df(factor), config)
        bounds = {name: tuple(pair) for name, pair in config["bounds"].items()}
        weights = model_fwl_weights(
            cells, treatment, list(config["confounders"]), bounds,
            overlap_eta=self.overlap_eta,
        )
        return weights, cells, bounds
        #################

    def _fwl_candidate_pool(self, domain):
        """All one-way and two-way marginals, every (T, Y, z) marginal for each
        query and confounder z, and each query's full (T, Y, Z) marginal.

        A full query marginal is included only if its own table fits
        max_model_size, since a larger one can never pass the model-size
        filter. max_candidate_arity is not used by this pool.
        """
        ############ CHANGED TO MATCH PSEUDOCODE ############
        attributes = tuple(domain.attributes)
        for cfg in self.ate_configs:
            query = (cfg["treatment"], cfg["outcome"], *cfg["confounders"])
            if len(set(query)) != len(query):
                raise ValueError(f"ATE query {cfg['name']} has repeated attributes")
            if any(name not in attributes for name in query):
                raise ValueError(f"ATE query {cfg['name']} names an unknown attribute")
        #################
        ############ SUB-TUPLE CAUSAL CREDIT ############
        order = {name: i for i, name in enumerate(attributes)}
        pool = [(name,) for name in attributes]
        pool += list(itertools.combinations(attributes, 2))
        for cfg in self.ate_configs:
            t, y = cfg["treatment"], cfg["outcome"]
            for z in cfg["confounders"]:
                pool.append(tuple(sorted((t, y, z), key=order.get)))
            full = tuple(sorted((t, y, *cfg["confounders"]), key=order.get))
            if junction_tree.hypothetical_model_size(domain, [full]) <= self.max_model_size:
                pool.append(full)
        return {cl: 1.0 for cl in dict.fromkeys(pool)}
        #################

    def _fwl_queries_for_candidate(self, clique):
        """Queries that the candidate scores: it contains the query's treatment,
        outcome and at least one of its confounders."""
        ############ SUB-TUPLE CAUSAL CREDIT ############
        attributes = set(clique)
        return [cfg for cfg in self.ate_configs
                if cfg["treatment"] in attributes and cfg["outcome"] in attributes
                and attributes.intersection(cfg["confounders"])]
        #################

    def _fwl_subquery(self, config, clique):
        """The query restricted to the confounders the candidate contains, and
        its key. Every FWL quantity of the candidate is computed as if this
        were the query; the reference ATE stays the full query's."""
        ############ SUB-TUPLE CAUSAL CREDIT ############
        attributes = set(clique)
        sub = dict(config, confounders=[z for z in config["confounders"]
                                        if z in attributes])
        return sub, (config["name"], tuple(sub["confounders"]))
        #################

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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        sim_model = estimation.MirrorDescent().estimate(
            data.domain, 
            sim_measurements, 
            iters=min(self.max_iters, 500),
            warm_start=potentials,
            callback_fn=lambda *_: None
        )
        #################
        
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
            ############ CHANGED TO MATCH PSEUDOCODE ############
            N = self._fwl_tilde_n if self.ate_method == "fwl" else x.sum()
            #################
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

    def _fwl_dynamic_mu(self, candidates, model, prev_model, sigma, kappa,
                        ############ CHANGED TO MATCH PSEUDOCODE ############
                        covered, current_components):
                        #################
        """Median score proxies using consecutive released PGM models.

        Substitute the current model for D and the previous model for the
        current model, including its frozen FWL coefficients.  No raw data or
        current-round selection result enters this calibration.
        """
        ############ CHANGED TO MATCH PSEUDOCODE ############
        from fwl import claim_fwl_ate_from_distribution
        #################

        ############ CHANGED TO MATCH PSEUDOCODE ############
        if prev_model is None:
            return 1.0
        eligible = [cl for cl in candidates if covered[cl]]
        if not eligible:
            return 1.0
        #################
        ############ SUB-TUPLE CAUSAL CREDIT ############
        # One proxy pair per (query, confounders in the candidate), computed
        # as if the candidate's sub-query were the query.
        proxy_values = {}
        for cl in eligible:
            for cfg in covered[cl]:
                sub, key = self._fwl_subquery(cfg, cl)
                if key in proxy_values:
                    continue
                weights, prev_cells, bounds = self._fwl_model_components(prev_model, sub)
                current_cells = current_components[key][1]
                args = (sub["treatment"], sub["outcome"],
                        list(sub["confounders"]), bounds)
                proxy_values[key] = (
                    claim_fwl_ate_from_distribution(
                        prev_cells, *args, weights=weights
                    ),
                    claim_fwl_ate_from_distribution(
                        current_cells, *args, weights=weights
                    ),
                )
        #################
        ############ CHANGED TO MATCH PSEUDOCODE ############
        l_values, a_values = [], []
        for cl in eligible:
            curr = model.project(cl).datavector()
            prev = prev_model.project(cl).datavector()
            discrepancy = np.linalg.norm(curr / curr.sum() - prev / prev.sum(), 1)
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl) / self._fwl_tilde_n
            l_values.append(abs(discrepancy - bias))
        #################

            ############ CHANGED TO MATCH PSEUDOCODE ############
            causal = 0.0
            for cfg in covered[cl]:
                name = cfg["name"]
                old_value, new_value = proxy_values[self._fwl_subquery(cfg, cl)[1]]
                reference = self._true_ates[name]
                causal += cfg["alpha"] * (
                    abs(reference - old_value) - abs(reference - new_value)
                )
            a_values.append(abs(causal))
        median_l, median_a = np.median(l_values), np.median(a_values)
        return 1.0 if median_l == 0 or median_a == 0 else float(median_l / (kappa * median_a))
            #################

    def _select_fwl(self, candidates, answers, data, model, prev_model,
                          ############ CHANGED TO MATCH PSEUDOCODE ############
                          epsilon, sigma):
                          #################
        """Select with FWL estimates computed on each candidate's own marginal.

        A candidate containing a query's treatment, outcome and some of its
        confounders is scored as if the query were adjusted for those
        confounders only; the reference ATE is the full query's.
        """
        ############ CHANGED TO MATCH PSEUDOCODE ############
        from fwl import claim_fwl_ate_estimator, claim_fwl_ate_from_distribution
        #################

        ############ CHANGED TO MATCH PSEUDOCODE ############
        if not candidates:
            raise ValueError("No marginal candidates satisfy the model-size bound")
        statistical = self._compute_stat_term(candidates, answers, model, sigma)
        covered = {cl: self._fwl_queries_for_candidate(cl) for cl in candidates}
        if not any(covered.values()):
            self._fwl_v = {}
            self._fwl_kappa = None
            print("FWL components: no eligible marginal covers a causal query")
            return self.exponential_mechanism(statistical, epsilon, 1 / self._fwl_tilde_n)
        #################

        ############ SUB-TUPLE CAUSAL CREDIT ############
        # For each (query, confounders in the candidate): the model's estimate
        # and our estimator on the data, both over the candidate's marginal,
        # with coefficients and v read off the current model's marginal.
        per_candidate = {cl: 0.0 for cl in candidates}
        variances, current_components, values = {}, {}, {}
        frame = None
        for cl in candidates:
            for cfg in covered[cl]:
                sub, key = self._fwl_subquery(cfg, cl)
                if key not in values:
                    weights, cells, bounds = self._fwl_model_components(model, sub)
                    current_components[key] = (weights, cells, bounds)
                    variances[key] = weights.variance
                    args = (sub["treatment"], sub["outcome"],
                            list(sub["confounders"]), bounds)
                    model_value = claim_fwl_ate_from_distribution(
                        cells, *args, weights=weights
                    )
                    if frame is None:
                        frame = _to_frame(data)
                    cols = [sub["treatment"], sub["outcome"], *sub["confounders"]]
                    real_rows = self._binarize_for_ate(frame[cols], sub)
                    data_value = claim_fwl_ate_estimator(
                        real_rows, *args, weights=weights,
                        n_tilde=self._fwl_tilde_n,
                    )
                    values[key] = (model_value, data_value)
                model_value, data_value = values[key]
                reference = self._true_ates[cfg["name"]]
                per_candidate[cl] += cfg["alpha"] * (
                    abs(reference - model_value) - abs(reference - data_value)
                )

        # κ bounds every candidate's causal sensitivity, sum_i β_i/(ñ v_{i,r}), by 1/ñ.
        denom = max(sum(cfg["alpha"] / variances[self._fwl_subquery(cfg, cl)[1]]
                        for cfg in covered[cl]) for cl in candidates)
        if denom <= 0:
            return self.exponential_mechanism(statistical, epsilon, 1 / self._fwl_tilde_n)
        kappa = 1 / denom
        self._fwl_v = variances
        self._fwl_kappa = kappa
        print("FWL components (current model, scored sub-queries):")
        for config in self.ate_configs:
            name = config["name"]
            vs = [v for (query, _), v in variances.items() if query == name]
            if vs:
                print(f"  v[{name}]: {len(vs)} sub-queries, "
                      f"min {min(vs):.6f}, max {max(vs):.6f}")
        print(f"  κ = {self._fwl_kappa:.6f}")
        #################
        ############ CHANGED TO MATCH PSEUDOCODE ############
        mu = self._fwl_dynamic_mu(
            candidates, model, prev_model, sigma, kappa, covered,
            current_components,
        )
        lam = self.marginal_weight
        if not 0 < lam <= 1:
            raise ValueError("FWL selection requires a positive lambda")
        normalizer = lam + (1 - lam) * mu
        qualities = {
            cl: (lam * statistical[cl]
                 + (1 - lam) * mu * kappa * per_candidate[cl]) / normalizer
            for cl in candidates
        }
        #################

        # Each sub-query contribution has sensitivity at most β_i/(n_tilde v_{i,r}).
        ############ CHANGED TO MATCH PSEUDOCODE ############
        return self.exponential_mechanism(qualities, epsilon, 1 / self._fwl_tilde_n)
        #################

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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if self.ate_method == "fwl":
            return self._select_fwl(
                candidates, answers, data, model, prev_model, epsilon, sigma
            )
        #################
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

    def _release_fwl_count(self, data):
        """Release the noisy dataset size before selection."""
        ############ CHANGED TO MATCH PSEUDOCODE ############
        rho_count = 1 / (2 * self.sigma_n**2)
        self._fwl_tilde_n = max(int(np.rint(data.records + self.gaussian_noise(self.sigma_n, 1)[0])), 1)
        self.rho -= rho_count
        #################

    ############ CHANGED TO MATCH PSEUDOCODE ############
    @staticmethod
    #################
    def _fwl_final_budget_guard(rho_used, total_rho, alpha, selection_rho, sigma):
        """Cap the next selection and measurement to the remaining budget."""
        ############ CHANGED TO MATCH PSEUDOCODE ############
        remaining = total_rho - rho_used
        if remaining <= 0:
            raise ValueError("No budget remains for a CLAIM selection round")
        if remaining <= 2 * (selection_rho + 1 / (2 * sigma**2)):
            return (1 - alpha) * remaining, np.sqrt(1 / (2 * alpha * remaining))
        return selection_rho, sigma
        #################

    def _fwl_ratio_sums(self, df, config):
        """N and V of the FWL effect tau = N / V of the rows in df:
        N = sum_z (n0 s1 - n1 s0) / n_z and V = sum_z n0 n1 / n_z over the
        occupied confounder groups z (an empty group contributes zero)."""
        ############ PRIVATE FWL REFERENCE (TWO NOISY SUMS) ############
        from fwl import _scaled_columns

        treatment, outcome = config["treatment"], config["outcome"]
        confounders = list(config["confounders"])
        bounds = {key: tuple(value) for key, value in config["bounds"].items()}
        rows = self._binarize_for_ate(df[[treatment, outcome, *confounders]], config)
        scaled = _scaled_columns(rows, [treatment, outcome], bounds, treatment)
        group = (rows.groupby(confounders, sort=False).ngroup().to_numpy()
                 if confounders else np.zeros(len(rows), dtype=int))
        groups = (pd.DataFrame({"g": group,
                                "t": scaled[treatment].to_numpy().astype(int),
                                "y": scaled[outcome].to_numpy()})
                  .groupby(["g", "t"])["y"].agg(["size", "sum"])
                  .unstack("t", fill_value=0)
                  .reindex(columns=pd.MultiIndex.from_product([["size", "sum"], [0, 1]]),
                           fill_value=0))
        n0, n1 = groups[("size", 0)].to_numpy(float), groups[("size", 1)].to_numpy(float)
        s0, s1 = groups[("sum", 0)].to_numpy(float), groups[("sum", 1)].to_numpy(float)
        n = n0 + n1
        return float(np.sum((n0 * s1 - n1 * s0) / n)), float(np.sum(n0 * n1 / n))
        #################

    def _release_fwl_reference_ates(self, data):
        """Release each query's FWL effect as two noisy sums, (N, V) + noise.

        One row changes N and V by less than 1 each (add/remove neighbours,
        outcomes in [0, 1]), so the 2k released numbers have L2 sensitivity
        sqrt(2k) and sigma = sqrt(k / rho_ATE). Only the noisy pair and the
        reference are kept; the observed groups are never released. The
        denominator is floored at eta * n_tilde, matching the model-side
        variance floor; dividing, flooring and clipping are post-processing.
        """
        ############ PRIVATE FWL REFERENCE (TWO NOISY SUMS) ############
        rho_ref = self._fwl_reference_rho
        sigma_ate = np.sqrt(len(self.ate_configs) / rho_ref)
        self.sigma_ate = float(sigma_ate)
        frame = _to_frame(data)
        floor = self.overlap_eta * self._fwl_tilde_n
        private_ates = {}
        for cfg in self.ate_configs:
            n_sum, v_sum = self._fwl_ratio_sums(frame, cfg)
            noise = self.gaussian_noise(sigma_ate, 2)
            noised_n, noised_v = n_sum + noise[0], v_sum + noise[1]
            reference = float(np.clip(noised_n / max(noised_v, floor), -1, 1))
            self._reference_release[cfg["name"]] = (noised_n, noised_v, reference)
            private_ates[cfg["name"]] = reference
        self.rho -= rho_ref
        return private_ates
        #################

    def _release_private_reference_ates(self, data):
        """Auto-release a privately DP-noised ATE for each config from a slice of rho."""
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if self.ate_method == "fwl":
            return self._release_fwl_reference_ates(data)
        #################
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
        
        raw_ates = self.compute_all_ates(_to_frame(data))
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

    def _run_fwl(self, data, workload, num_synth_rows=None,
                       ############ CHANGED TO MATCH PSEUDOCODE ############
                       initial_cliques=None):
                       #################
        """Run the FWL loop on all marginals of arity at most ell."""
        ############ CHANGED TO MATCH PSEUDOCODE ############
        alpha = 0.9
        total_rho = self._initial_rho
        attributes = tuple(data.domain.attributes)
        rounds = self.rounds or 16 * len(attributes)
        if rounds < len(attributes):
            raise ValueError("FWL rounds must include the initial one-way measurements")
        candidates = self._fwl_candidate_pool(data.domain)
        oneway = [cl for cl in candidates if len(cl) == 1]
        if initial_cliques is not None and set(initial_cliques) != set(oneway):
            raise ValueError("FWL initialization measures every one-way marginal")
        sigma_0 = np.sqrt(rounds / (2 * alpha * total_rho))
        #################

        # Release the noisy size, measure all one-way marginals,
        # fits the initial PGM, then releases the reference ATE statistics.
        ############ CHANGED TO MATCH PSEUDOCODE ############
        rho_used = 0.0
        self._release_fwl_count(data)
        rho_used += 1 / (2 * self.sigma_n**2)
        measurements = []
        for cl in oneway:
            x = data.project(cl).datavector()
            measurements.append(LinearMeasurement(
                x + self.gaussian_noise(sigma_0, x.size), cl, stddev=sigma_0
            ))
            rho_used += 1 / (2 * sigma_0**2)
        model = estimation.MirrorDescent().estimate(
            data.domain, measurements, iters=self.max_iters,
            callback_fn=lambda *_: None,
        )
        self._true_ates = self._release_fwl_reference_ates(data)
        rho_used += self._fwl_reference_rho
        if rho_used >= total_rho:
            raise ValueError("FWL initialization exhausted the zCDP budget")
        #################

        # Raw candidate marginals remain local;
        # the exponential mechanism is the only way their scores are released.
        ############ CHANGED TO MATCH PSEUDOCODE ############
        answers = {cl: data.project(cl).datavector() for cl in candidates}
        previous_model = None
        t = len(oneway)
        sigma = sigma_0
        selection_rho = (1 - alpha) * total_rho / rounds
        while total_rho - rho_used > 1e-12 * total_rho:
        #################
            if t == len(oneway):
                selection_rho, sigma = self._fwl_final_budget_guard(
                    rho_used, total_rho, alpha, selection_rho, sigma,
                )
            else:
                after = model.project(cl).datavector()
                if np.linalg.norm(after - before, 1) <= np.sqrt(2 / np.pi) * sigma * x.size:
                    selection_rho, sigma = 4 * selection_rho, sigma / 2
                selection_rho, sigma = self._fwl_final_budget_guard(
                    rho_used, total_rho, alpha, selection_rho, sigma,
                )
            t += 1
            ############ CHANGED TO MATCH PSEUDOCODE ############
            rho_used += selection_rho + 1 / (2 * sigma**2)
            size_limit = self.max_model_size * rho_used / total_rho
            allowed = {
                cl: weight for cl, weight in candidates.items()
                if junction_tree.hypothetical_model_size(
                    data.domain, list(model.cliques) + [cl]
                ) <= size_limit
            }
            #################
            ############ CHANGED TO MATCH PSEUDOCODE ############
            print(
                f"Maximum candidate marginal size: {max(map(len, allowed))}-way"
                if allowed else "Maximum candidate marginal size: none"
            )
            #################

            # Mechanism.exponential_mechanism uses eps*q/(2*sensitivity).
            # With eps=sqrt(8*rho_t) and sensitivity=1/n_tilde, this is
            # sqrt(2*rho_t)*n_tilde*q as in the pseudocode.
            ############ CHANGED TO MATCH PSEUDOCODE ############
            epsilon = np.sqrt(8 * selection_rho)
            cl = self._select_fwl(
                allowed, answers, data, model, previous_model,
                epsilon, sigma,
            )
            x = answers[cl]
            measurements.append(LinearMeasurement(
                x + self.gaussian_noise(sigma, x.size), cl, stddev=sigma
            ))
            before = model.project(cl).datavector()
            pcliques = list(set(m.clique for m in measurements))
            potentials = model.potentials.expand(pcliques)
            previous_model = model
            model = estimation.MirrorDescent().estimate(
                data.domain, measurements, iters=self.max_iters,
                warm_start=potentials, callback_fn=lambda *_: None,
            )
            self._maybe_update_marginal_weight(
                model, measurements[:len(oneway)], oneway
            )
            #################
            ############ CLEAR COMPILED CODE ONCE PER ROUND ############
            # Each round compiles a projection per candidate; without clearing,
            # the process runs out of memory mappings (vm.max_map_count).
            jax.clear_caches()
            #################

        ############ CHANGED TO MATCH PSEUDOCODE ############
        synth = model.synthetic_data(rows=self._fwl_tilde_n)
        return model, synth
        #################

    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if self.selection_mode == "ate" and self.ate_method == "fwl":
            return self._run_fwl(
                data, workload, num_synth_rows=num_synth_rows,
                initial_cliques=initial_cliques,
            )
        #################
        # Legacy marginal / DoWhy paths keep their existing AIM workload.
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

            # The legacy DoWhy coefficient uses raw D and has no separate
            # privacy accounting. The FWL route above uses the current PGM.
            self._cache_fwl_components(_to_frame(data))
            ############ CHANGED TO MATCH PSEUDOCODE ############
            # if self.ate_method == "fwl":
            #     print("FWL components:")
            #     for config in self.ate_configs:
            #         name = config["name"]
            #         print(f"  v[{name}] = {self._fwl_v[name]:.6f}")
            #################
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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        model = estimation.MirrorDescent().estimate(
                data.domain, measurements, iters=self.max_iters, callback_fn=lambda *_: None
        )
        #################

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
            ############ CHANGED TO MATCH PSEUDOCODE ############
            model = estimation.MirrorDescent().estimate(
                    data.domain, measurements, iters=self.max_iters, warm_start=potentials, callback_fn=lambda *_: None
            )
            #################
            
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
        ############ CHANGED TO MATCH PSEUDOCODE ############
        model = estimation.MirrorDescent().estimate(
            data.domain, measurements, iters=self.max_iters, warm_start=potentials
        )
        #################
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    return_mechanism=False,
    #################
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
            true_ates = mech.compute_all_ates(_to_frame(data))

        for b in range(pilot_samples):
            rows = num_synth_rows or data.records
            pilot_df = _to_frame(model.synthetic_data(rows=rows))
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    if return_mechanism:
        return selected_lambda, scores, final_model, final_synth, final_mech
    #################
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    params["max_candidate_arity"] = None
    #################
    # Selection mode: "marginal" (L1-based) or "ate" (ATE-based)
    ############ CHANGED TO MATCH PSEUDOCODE ############
    params["selection_mode"] = None  # ATE when queries are supplied, otherwise marginal
    #################
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    params["reference_ate_rho"] = None
    params["sigma_n"] = None  # FWL default spends 1% of rho on the count
    params["overlap_eta"] = 0.01
    params["n_min"] = 1
    params["sim_sample_size"] = 5000
    #################
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
    # Causal selection defaults to the full-query FWL path.
    ############ CHANGED TO MATCH PSEUDOCODE ############
    params["ate_method"] = "fwl"
    #################

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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    parser.add_argument(
        "--max_candidate_arity", type=int,
        help="maximum FWL candidate arity ell; default is the largest query size",
    )
    #################
    parser.add_argument("--save", type=str, help="path to save synthetic data")
    
    # Selection mode arguments
    ############ CHANGED TO MATCH PSEUDOCODE ############
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["marginal", "ate"],
        help="Selection mode: 'ate' when queries are supplied, otherwise 'marginal'"
    )
    #################
    
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    parser.add_argument(
        "--reference_ate_rho", type=float,
        help="Absolute zCDP budget for the FWL reference release",
    )
    parser.add_argument(
        "--sigma_n", type=float,
        help="Standard deviation for the FWL noisy record count"
    )
    parser.add_argument("--overlap_eta", type=float,
                        help="Floor for model treatment variance")
    parser.add_argument("--n_min", type=int,
                        help="Floor for reference ATE stratum counts")
    parser.add_argument("--sim_sample_size", type=int,
                        help="Number of model samples for adaptive lambda")
    parser.add_argument(
        "--ate_outcome_min",
        type=float,
        help="Minimum value the outcome column is clipped to before ATE computation"
    )
    #################
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    parser.add_argument(
        "--causal_graph",
        type=str,
        help="Path to GML file with causal graph (required for the DoWhy backend)"
    )
    #################
    parser.add_argument(
        "--marginal_weight",
        type=float,
        help="Weight for L1 marginal error in hybrid selection (0.0 = pure ATE, 1.0 = pure marginal)."
             " Only used when selection_mode='ate'. Default: 0.3"
    )
    ############ CHANGED TO MATCH PSEUDOCODE ############
    parser.add_argument(
        "--ate_method",
        type=str,
        choices=["dowhy", "fwl"],
        help="ATE estimation backend: 'dowhy' (CausalModel + backdoor) or 'fwl' "
             "(full-query FWL, the default). 'fwl' requires per-config 'bounds'."
    )
    #################

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    
    # Load ATE configs from JSON file if provided
    ate_configs_list = None
    if args.ate_configs is not None:
        with open(args.ate_configs, 'r') as f:
            ate_configs_list = json.load(f)
    ############ CHANGED TO MATCH PSEUDOCODE ############
    if args.selection_mode is None:
        args.selection_mode = "ate" if ate_configs_list else "marginal"
    #################

    # The new mbi has no Dataset.load: read the CSV and the domain JSON directly.
    _frame = pd.read_csv(args.dataset)
    with open(args.domain) as _f:
        _config = json.load(_f)
    data = Dataset({a: _frame[a].to_numpy() for a in _config},
                   Domain(list(_config), list(_config.values())))

    ############ CHANGED TO MATCH PSEUDOCODE ############
    if args.selection_mode == "ate" and args.ate_method == "fwl":
    #################
        # The FWL path builds every candidate up to max_candidate_arity.
        ############ CHANGED TO MATCH PSEUDOCODE ############
        workload = []
    else:
        workload = list(itertools.combinations(data.domain, args.degree))
        workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
        if args.num_marginals is not None:
            prng = np.random
            workload = [
                workload[i]
                for i in prng.choice(len(workload), args.num_marginals, replace=False)
            ]
        workload = [(cl, 1.0) for cl in workload]
        #################
    
    adaptive_lambda = args.adaptive_lambda and not args.fixed_lambda_from_list

    ############ CHANGED TO MATCH PSEUDOCODE ############
    claim_kwargs = dict(
        max_model_size=args.max_model_size,
        max_candidate_arity=args.max_candidate_arity,
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
        reference_ate_rho=args.reference_ate_rho,
        sigma_n=args.sigma_n,
        overlap_eta=args.overlap_eta,
        n_min=args.n_min,
        sim_sample_size=args.sim_sample_size,
        ate_outcome_range=(args.ate_outcome_min, args.ate_outcome_max),
    )
    #################

    selected_lambda = None
    lambda_scores = None
    if args.fixed_lambda_from_list:
        lambda_values = [float(x.strip()) for x in args.lambda_values.split(',') if x.strip()]
        ############ CHANGED TO MATCH PSEUDOCODE ############
        selected_lambda, lambda_scores, model, synth, mech = pilot_select_lambda(
            data,
            workload,
            args.epsilon,
            args.delta,
            lambda_values=lambda_values,
            pilot_fraction=args.pilot_fraction,
            pilot_samples=args.pilot_samples,
            return_mechanism=True,
            **claim_kwargs,
        )
        #################
    else:
        mech = CLAIM(args.epsilon, args.delta, **claim_kwargs)
        model, synth = mech.run(data, workload)

    if args.save is not None:
        _to_frame(synth).to_csv(args.save, index=False)

    # Print ATE comparison if in ATE mode
    if args.selection_mode == "ate":
        ############ CHANGED TO MATCH PSEUDOCODE ############
        if args.ate_method == "fwl":
            true_ates = mech.compute_all_ates(_to_frame(data))
        else:
            true_ates = mech._true_ates
        #################
        synth_ates = mech.compute_all_ates(_to_frame(synth))
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
    ############ CHANGED TO MATCH PSEUDOCODE ############
    if args.selection_mode == "ate" and args.ate_method == "fwl":
        evaluation_workload = [
            (cl, 1.0) for cl in itertools.combinations(data.domain, 2)
        ]
    else:
        evaluation_workload = workload
    # for proj, wgt in workload:
    for proj, wgt in evaluation_workload:
    #################
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        Z = model.project(proj).datavector()
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        synth_errors.append(e)
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Z / Z.sum(), 1)
        model_errors.append(e)
    print("Average Marginal Error: ", np.mean(model_errors), np.mean(synth_errors))


