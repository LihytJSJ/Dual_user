"""GOAC with true measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd




BASE_PARAMETERS: Dict[str, float] = {
    "K_I": 102.1,
    "K_IP": 41.5,
    "c_max": 15.0,
    "K_1": 0.46,
    "K_2": 0.18,
    "Y_c": 0.51,
    "Y_p": 0.61,
    "m_f": 0.04,
    "s_F": 600.0,
}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    profile: str
    x0: Tuple[float, float, float, float] = (5.0, 0.0, 50.0, 10.0)
    dpdt_noise_std: float = 0.05
    state_noise_std: Tuple[float, float, float] = (0.05, 0.02, 0.20)
    measurement_stride: int = 1
    measurement_control_stride: int | None = None
    measurement_delay_h: float = 0.0
    mu_mismatch_fraction: float = 0.0
    ksp_mismatch_fraction: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:

    profile_name: str = "paper"
    total_time_h: float = 300.0
    integration_step_h: float = 0.02
    measurement_interval_h: float = 0.1
    posterior_interval_h: float = 0.2
    control_interval_h: float = 0.2
    repeated_runs: int = 20
    base_seed: int = 20260817
    u_min_lph: float = 0.0
    u_max_lph: float = 3.0
    du_max_lph: float = 0.5
    volume_max_l: float = 30.0
    reserve_volume_for_ramp_down: bool = True
    kip_min: float = 10.0
    kip_max: float = 70.0
    n_kip_intervals: int = 60
    interval_diffusion_probability: float = 0.1
    interval_likelihood_std: float = 1.0
    interval_innovation_threshold_std: float = 4.0
    interval_innovation_persistence: int = 2
    interval_innovation_mixing_probability: float = 0.05
    confidence_level: float = 0.9
    substrate_target_min: float = 5.0
    substrate_target_max: float = 45.0
    n_target_candidates: int = 81
    target_kernel_std_gpl: float = 1.0
    target_credible_level: float = 0.9
    target_virtual_temperature: float = 4.0
    target_virtual_exploration_weight: float = 0.2
    target_product_increment_noise_multiplier: float = 1.0
    target_min_refinement_candidates: int = 5
    target_real_prior_mixing_probability: float = 0.02
    state_process_std: Tuple[float, float, float] = (0.03, 0.015, 0.12)
    proposed_q: float = 500.0
    proposed_r: float = 0.2
    proposed_regularization: float = 1e-08
    proposed_use_virtual_posterior: bool = True
    proposed_include_posterior_uncertainty: bool = True
    retain_output_target_cross_covariance: bool = False
    nominal_mu_max_per_h: float = 0.28
    nominal_k_sp_gpl: float = 12.2
    algorithms: Tuple[str, ...] = ("GOAC",)

    @property
    def interval_width_xi(self) -> float:
        return (self.kip_max - self.kip_min) / self.n_kip_intervals

    @property
    def integration_substeps(self) -> int:
        ratio = self.control_interval_h / self.integration_step_h
        rounded = int(round(ratio))
        if rounded < 1 or abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "control_interval_h must be an integer multiple of integration_step_h"
            )
        return rounded

    @property
    def measurement_substeps(self) -> int:
        ratio = self.measurement_interval_h / self.integration_step_h
        rounded = int(round(ratio))
        if rounded < 1 or abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "measurement_interval_h must be an integer multiple of integration_step_h"
            )
        return rounded

    @property
    def posterior_substeps(self) -> int:
        ratio = self.posterior_interval_h / self.integration_step_h
        rounded = int(round(ratio))
        if rounded < 1 or abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "posterior_interval_h must be an integer multiple of integration_step_h"
            )
        return rounded

    @property
    def posterior_measurement_steps(self) -> int:
        ratio = self.posterior_interval_h / self.measurement_interval_h
        rounded = int(round(ratio))
        if rounded < 1 or abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "posterior_interval_h must be an integer multiple of measurement_interval_h"
            )
        return rounded

    @property
    def control_measurement_steps(self) -> int:
        ratio = self.control_interval_h / self.measurement_interval_h
        rounded = int(round(ratio))
        if rounded < 1 or abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "control_interval_h must be an integer multiple of measurement_interval_h"
            )
        return rounded

    @property
    def n_measurement_steps(self) -> int:
        ratio = self.total_time_h / self.measurement_interval_h
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "total_time_h must be an integer multiple of measurement_interval_h"
            )
        return rounded

    @property
    def n_integration_steps(self) -> int:
        ratio = self.total_time_h / self.integration_step_h
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "total_time_h must be an integer multiple of integration_step_h"
            )
        return rounded

    @property
    def n_control_steps(self) -> int:
        ratio = self.total_time_h / self.control_interval_h
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1e-09:
            raise ValueError(
                "total_time_h must be an integer multiple of control_interval_h"
            )
        return rounded

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["interval_width_xi"] = self.interval_width_xi
        data["integration_substeps"] = self.integration_substeps
        data["measurement_substeps"] = self.measurement_substeps
        data["posterior_substeps"] = self.posterior_substeps
        data["posterior_measurement_steps"] = self.posterior_measurement_steps
        data["control_measurement_steps"] = self.control_measurement_steps
        data["goac_measurement_interval_h"] = self.measurement_interval_h
        data["n_measurement_steps"] = self.n_measurement_steps
        data["n_integration_steps"] = self.n_integration_steps
        data["n_control_steps"] = self.n_control_steps
        return data


def make_profile(name: str, repeated_runs: int | None = None) -> SimulationConfig:
    if name == "paper":
        config = SimulationConfig()
    elif name == "quick":
        config = SimulationConfig(
            profile_name="quick",
            total_time_h=30.0,
            integration_step_h=0.05,
            measurement_interval_h=0.1,
            posterior_interval_h=0.2,
            control_interval_h=0.5,
            repeated_runs=2,
            n_kip_intervals=24,
            n_target_candidates=41,
        )
    else:
        raise ValueError(f"Unknown profile: {name}")
    return (
        replace(config, repeated_runs=repeated_runs)
        if repeated_runs is not None
        else config
    )


SCENARIOS: Dict[str, ScenarioConfig] = {
    "piecewise_stress": ScenarioConfig(
        name="piecewise_stress",
        description="Severe 12-60-10 g/L K_IP step profile retained as a stress test.",
        profile="piecewise",
    ),
    "gradual_drift": ScenarioConfig(
        name="gradual_drift",
        description=(
            "Wide but smooth low-high-low K_IP drift representing gradual strain adaptation "
            "and loss of tolerance without idealized parameter jumps."
        ),
        profile="gradual_wide",
    ),
    "stochastic_drift": ScenarioConfig(
        name="stochastic_drift",
        description=(
            "Bounded mean-reverting K_IP variability around changing operating regimes, "
            "representing stochastic population-composition changes."
        ),
        profile="stochastic_regime",
    ),
    "multi_parameter_mismatch": ScenarioConfig(
        name="multi_parameter_mismatch",
        description="Correlated K_IP, mu_max, and K_sp variation plus nominal-model mismatch.",
        profile="correlated_multi_parameter",
        mu_mismatch_fraction=-0.10,
        ksp_mismatch_fraction=0.15,
    ),
    "mutation_contamination_challenge": ScenarioConfig(
        name="mutation_contamination_challenge",
        description=(
            "Strain-population shift and contamination-relapse challenge with low-high-low "
            "apparent product-inhibition tolerance regimes that demand rapid target reallocation."
        ),
        profile="mutation_contamination",
    ),
    "high_measurement_noise": ScenarioConfig(
        name="high_measurement_noise",
        description="Wide smooth K_IP drift with threefold state and product-rate measurement noise.",
        profile="gradual_wide",
        dpdt_noise_std=0.15,
        state_noise_std=(0.15, 0.06, 0.60),
    ),
    "alternate_initial_condition": ScenarioConfig(
        name="alternate_initial_condition",
        description=(
            "Wide smooth K_IP drift from a lower biomass, substrate, and liquid-volume "
            "initial state."
        ),
        profile="gradual_wide",
        x0=(3.0, 0.0, 20.0, 8.0),
    ),
    "delayed_sparse_measurements": ScenarioConfig(
        name="delayed_sparse_measurements",
        description=(
            "Abrupt low-high-low K_IP changes with sparse measurements and a one-hour "
            "nominal delay."
        ),
        profile="delayed_piecewise",
        measurement_control_stride=3,
        measurement_delay_h=1.0,
    ),
}


DEFAULT_SCENARIOS = tuple(SCENARIOS)


Array = np.ndarray




def intrinsic_rates(
    x: Array,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Tuple[float, float, float]:
    c, _, s, _ = np.asarray(x, dtype=float)
    c = max(float(c), 0.0)
    s = max(float(s), 0.0)
    growth_capacity = max(0.0, 1.0 - c / parameters["c_max"])
    mu = mu_max * growth_capacity / (1.0 + s / parameters["K_I"])
    r_c = mu * c
    saturation = s / (s + k_sp) if s > 0.0 else 0.0
    inhibition = 1.0 / (1.0 + s / parameters["K_IP"])
    r_p = parameters["K_1"] * r_c + parameters["K_2"] * saturation * inhibition * c
    r_s = -r_c / parameters["Y_c"] - r_p / parameters["Y_p"] - parameters["m_f"] * c
    return r_c, r_p, r_s


def fermentation_rhs(
    x: Array,
    u: float,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Array:
    c, p, s, volume = np.asarray(x, dtype=float)
    volume = max(float(volume), 1e-9)
    r_c, r_p, r_s = intrinsic_rates(x, mu_max, k_sp, parameters)
    return np.array(
        [
            r_c - u * c / volume,
            r_p - u * p / volume,
            r_s + u * (parameters["s_F"] - s) / volume,
            u,
        ],
        dtype=float,
    )


def product_mass_rate(
    x: Array,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> float:
    """d(pV)/dt = V r_p, with p a concentration and pV a mass."""
    return max(float(x[3]), 1e-9) * intrinsic_rates(x, mu_max, k_sp, parameters)[1]


def integrate_interval(
    x0: Array,
    u: float,
    duration_h: float,
    substeps: int,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Tuple[Array, float]:
    x = np.asarray(x0, dtype=float).copy()
    h = duration_h / substeps
    formed_mass = 0.0
    for _ in range(substeps):
        k1 = fermentation_rhs(x, u, mu_max, k_sp, parameters)
        q1 = product_mass_rate(x, mu_max, k_sp, parameters)
        x2 = x + 0.5 * h * k1
        k2 = fermentation_rhs(x2, u, mu_max, k_sp, parameters)
        q2 = product_mass_rate(x2, mu_max, k_sp, parameters)
        x3 = x + 0.5 * h * k2
        k3 = fermentation_rhs(x3, u, mu_max, k_sp, parameters)
        q3 = product_mass_rate(x3, mu_max, k_sp, parameters)
        x4 = x + h * k3
        k4 = fermentation_rhs(x4, u, mu_max, k_sp, parameters)
        q4 = product_mass_rate(x4, mu_max, k_sp, parameters)
        x += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        formed_mass += (h / 6.0) * (q1 + 2.0 * q2 + 2.0 * q3 + q4)
        x[:3] = np.maximum(x[:3], 0.0)
        x[3] = max(x[3], 1e-9)
    return x, formed_mass


def fermentation_rhs_parameter_support(
    states: Array,
    u: float,
    kip_support: Array,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Array:
    x = np.asarray(states, dtype=float)
    kip = np.asarray(kip_support, dtype=float)
    c = np.maximum(x[:, 0], 0.0)
    p = np.maximum(x[:, 1], 0.0)
    s = np.maximum(x[:, 2], 0.0)
    volume = np.maximum(x[:, 3], 1e-9)
    growth_capacity = np.maximum(0.0, 1.0 - c / parameters["c_max"])
    mu = mu_max * growth_capacity / (1.0 + s / parameters["K_I"])
    r_c = mu * c
    saturation = np.divide(s, s + k_sp, out=np.zeros_like(s), where=s > 0.0)
    inhibition = 1.0 / (1.0 + s / kip)
    r_p = parameters["K_1"] * r_c + parameters["K_2"] * saturation * inhibition * c
    r_s = -r_c / parameters["Y_c"] - r_p / parameters["Y_p"] - parameters["m_f"] * c
    feed = np.broadcast_to(np.asarray(u, dtype=float), c.shape)
    return np.column_stack(
        (
            r_c - feed * c / volume,
            r_p - feed * p / volume,
            r_s + feed * (parameters["s_F"] - s) / volume,
            feed,
        )
    )


def product_mass_rate_parameter_support(
    states: Array,
    kip_support: Array,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Array:
    x = np.asarray(states, dtype=float)
    intrinsic = fermentation_rhs_parameter_support(
        x,
        0.0,
        kip_support,
        mu_max,
        k_sp,
        parameters,
    )
    return np.maximum(x[:, 3], 1e-9) * intrinsic[:, 1]


def integrate_parameter_support(
    x0: Array,
    controls: Sequence[float],
    control_interval_h: float,
    integration_substeps: int,
    kip_support: Array,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Tuple[Array, Array]:
    kip = np.asarray(kip_support, dtype=float)
    states = np.repeat(np.asarray(x0, dtype=float)[None, :], len(kip), axis=0)
    formed_mass = np.zeros(len(kip), dtype=float)
    h = control_interval_h / integration_substeps
    for control in np.asarray(controls, dtype=float):
        for _ in range(integration_substeps):
            k1 = fermentation_rhs_parameter_support(
                states, control, kip, mu_max, k_sp, parameters
            )
            q1 = product_mass_rate_parameter_support(
                states, kip, mu_max, k_sp, parameters
            )
            x2 = states + 0.5 * h * k1
            k2 = fermentation_rhs_parameter_support(
                x2, control, kip, mu_max, k_sp, parameters
            )
            q2 = product_mass_rate_parameter_support(x2, kip, mu_max, k_sp, parameters)
            x3 = states + 0.5 * h * k2
            k3 = fermentation_rhs_parameter_support(
                x3, control, kip, mu_max, k_sp, parameters
            )
            q3 = product_mass_rate_parameter_support(x3, kip, mu_max, k_sp, parameters)
            x4 = states + h * k3
            k4 = fermentation_rhs_parameter_support(
                x4, control, kip, mu_max, k_sp, parameters
            )
            q4 = product_mass_rate_parameter_support(x4, kip, mu_max, k_sp, parameters)
            states += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            formed_mass += (h / 6.0) * (q1 + 2.0 * q2 + 2.0 * q3 + q4)
            states[:, :3] = np.maximum(states[:, :3], 0.0)
            states[:, 3] = np.maximum(states[:, 3], 1e-9)
    return states, formed_mass


def normalized_gaussian_likelihood(residual: Array, standard_deviation: float) -> Array:
    variance = max(float(standard_deviation), 1e-9) ** 2
    log_likelihood = -0.5 * np.square(residual) / variance
    log_likelihood -= np.max(log_likelihood)
    return np.exp(log_likelihood)


class IntervalBayesEstimator:

    def __init__(
        self,
        lower: float,
        upper: float,
        n_intervals: int,
        likelihood_std: float,
        diffusion_probability: float,
        innovation_threshold_std: float = math.inf,
        innovation_persistence: int = 1,
        innovation_mixing_probability: float = 0.0,
    ) -> None:
        self.edges = np.linspace(lower, upper, n_intervals + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.weights = np.full(n_intervals, 1.0 / n_intervals)
        self.likelihood_std = likelihood_std
        self.diffusion_probability = diffusion_probability
        self.innovation_threshold_std = innovation_threshold_std
        self.innovation_persistence = max(1, int(innovation_persistence))
        self.innovation_mixing_probability = float(
            np.clip(innovation_mixing_probability, 0.0, 1.0)
        )
        self.innovation_run_length = 0
        self.last_innovation_sigma = 0.0
        self.last_mixing_probability = 0.0
        self.last_prior_variance = self.variance
        self.last_information_gain = 0.0
        self.last_state_predictions = np.empty((0, 4), dtype=float)
        self.last_product_increment_predictions = np.empty(0, dtype=float)
        self.last_update_used_state_observation = False

    def _proposal(self) -> Array:
        move = self.diffusion_probability / 2.0
        stay = 1.0 - self.diffusion_probability
        proposal = np.zeros_like(self.weights)
        proposal[1:-1] = (
            stay * self.weights[1:-1]
            + move * self.weights[:-2]
            + move * self.weights[2:]
        )
        proposal[0] = (stay + move) * self.weights[0] + move * self.weights[1]
        proposal[-1] = (stay + move) * self.weights[-1] + move * self.weights[-2]
        return proposal

    def update_from_state_transition(
        self,
        previous_state: Array,
        measured_state: Array,
        controls: Sequence[float],
        mu_max: float,
        k_sp: float,
        nominal_parameters: Mapping[str, float],
        control_interval_h: float,
        integration_substeps: int,
        state_measurement_std: Array,
        state_process_std: Array,
    ) -> None:
        controls = np.asarray(controls, dtype=float)
        if controls.size == 0:
            self.last_update_used_state_observation = False
            return
        predictions, product_increments = integrate_parameter_support(
            previous_state,
            controls,
            control_interval_h,
            integration_substeps,
            self.centers,
            mu_max,
            k_sp,
            nominal_parameters,
        )
        prior = self._proposal()
        self.last_prior_variance = float(
            np.dot(np.square(self.centers - np.dot(prior, self.centers)), prior)
        )
        measurement_std = np.asarray(state_measurement_std, dtype=float)[:3]
        process_std = np.asarray(state_process_std, dtype=float)[:3]
        residual_std = np.sqrt(
            2.0 * np.square(measurement_std) + controls.size * np.square(process_std)
        )
        residual_std = np.maximum(self.likelihood_std * residual_std, 1e-6)
        posterior_prediction = np.einsum("i,ij->j", prior, predictions[:, :3])
        normalized_innovation = (
            np.asarray(measured_state, dtype=float)[:3] - posterior_prediction
        ) / residual_std
        self.last_innovation_sigma = float(
            np.sqrt(np.mean(np.square(normalized_innovation)))
        )
        if self.last_innovation_sigma > self.innovation_threshold_std:
            self.innovation_run_length += 1
        else:
            self.innovation_run_length = 0
        self.last_mixing_probability = 0.0
        if self.innovation_run_length >= self.innovation_persistence:
            self.last_mixing_probability = self.innovation_mixing_probability
            uniform = np.full_like(prior, 1.0 / len(prior))
            prior = (
                1.0 - self.last_mixing_probability
            ) * prior + self.last_mixing_probability * uniform
        residual = (
            np.asarray(measured_state, dtype=float)[None, :3] - predictions[:, :3]
        ) / residual_std[None, :]
        log_likelihood = -0.5 * np.sum(np.square(residual), axis=1)
        log_likelihood -= np.max(log_likelihood)
        posterior = prior * np.exp(log_likelihood)
        total = posterior.sum()
        self.weights = (
            posterior / total
            if total > 1e-300
            else np.full_like(posterior, 1.0 / len(posterior))
        )
        self.last_information_gain = float(
            np.sum(
                self.weights
                * np.log(np.maximum(self.weights, 1e-300) / np.maximum(prior, 1e-300))
            )
        )
        self.last_state_predictions = predictions
        self.last_product_increment_predictions = product_increments
        self.last_update_used_state_observation = True

    @property
    def mean(self) -> float:
        return float(np.dot(self.centers, self.weights))

    @property
    def variance(self) -> float:
        return float(np.dot(np.square(self.centers - self.mean), self.weights))

    @property
    def predictive_weights(self) -> Array:
        """Return the next-step bounded proposal without mutating the filter."""
        return self._proposal()

    def credible_interval(self, level: float) -> Tuple[float, float]:
        tail = 0.5 * (1.0 - level)
        cumulative = np.cumsum(self.weights)
        low_index = min(
            int(np.searchsorted(cumulative, tail, side="left")), len(self.weights) - 1
        )
        high_index = min(
            int(np.searchsorted(cumulative, 1.0 - tail, side="left")),
            len(self.weights) - 1,
        )
        return float(self.edges[low_index]), float(self.edges[high_index + 1])


def predicted_product_increment_surface(
    state: Array,
    kip_support: Array,
    substrate_support: Array,
    duration_h: float,
    config: SimulationConfig,
    parameters: Mapping[str, float],
) -> Array:
    c = max(float(np.asarray(state, dtype=float)[0]), 1e-9)
    volume = max(float(np.asarray(state, dtype=float)[3]), 1e-9)
    substrate = np.asarray(substrate_support, dtype=float)
    kip = np.asarray(kip_support, dtype=float)
    growth_capacity = max(0.0, 1.0 - c / parameters["c_max"])
    mu = (
        config.nominal_mu_max_per_h
        * growth_capacity
        / (1.0 + substrate[None, :] / parameters["K_I"])
    )
    r_c = mu * c
    saturation = substrate[None, :] / (substrate[None, :] + config.nominal_k_sp_gpl)
    inhibition = 1.0 / (1.0 + substrate[None, :] / kip[:, None])
    r_p = parameters["K_1"] * r_c + parameters["K_2"] * saturation * inhibition * c
    return duration_h * volume * r_p


class HierarchicalSubstrateLearner:
    """Update the real target posterior, then refine within its credible interval."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.edges = np.linspace(
            config.substrate_target_min,
            config.substrate_target_max,
            config.n_target_candidates + 1,
        )
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.weights = np.full(len(self.centers), 1.0 / len(self.centers))
        self.real_weights = self.weights.copy()
        self.virtual_prior = self.weights.copy()
        self.paired_targets = np.full(config.n_kip_intervals, np.mean(self.centers))
        self.real_paired_targets = self.paired_targets.copy()
        self.paired_target_variances = np.zeros(config.n_kip_intervals)
        self.paired_weights = np.full(
            config.n_kip_intervals, 1.0 / config.n_kip_intervals
        )
        self.last_product_likelihood = np.ones(config.n_kip_intervals)
        self._pending_product_likelihood = np.ones(config.n_kip_intervals)
        self._pending_product_evidence = False
        self.last_product_increment = math.nan
        self.last_product_increment_std = math.nan
        self.last_product_information_gain = 0.0
        self.last_real_interval = (
            config.substrate_target_min,
            config.substrate_target_max,
        )
        self.last_virtual_reward_variance = 0.0
        self.last_virtual_refinement_used = False
        self.last_update_used_product_increment = False

    def observe_product_increment(
        self,
        measured_increment: float,
        predicted_increments: Array,
        parameter_weights: Array,
        measurement_std: float,
    ) -> None:
        predictions = np.asarray(predicted_increments, dtype=float)
        prior = np.asarray(parameter_weights, dtype=float)
        if predictions.shape != prior.shape or predictions.size == 0:
            raise ValueError(
                "Product-increment predictions must match parameter support"
            )
        standard_deviation = max(float(measurement_std), 1e-9)
        likelihood = normalized_gaussian_likelihood(
            float(measured_increment) - predictions,
            standard_deviation,
        )
        posterior = prior * likelihood
        total = posterior.sum()
        posterior = posterior / total if total > 1e-300 else prior / prior.sum()
        self._pending_product_likelihood *= likelihood
        pending_total = self._pending_product_likelihood.sum()
        if pending_total > 1e-300:
            self._pending_product_likelihood /= pending_total
        else:
            self._pending_product_likelihood.fill(
                1.0 / len(self._pending_product_likelihood)
            )
        self.last_product_likelihood = self._pending_product_likelihood.copy()
        self.last_product_increment = float(measured_increment)
        self.last_product_increment_std = standard_deviation
        self.last_product_information_gain = float(
            np.sum(
                posterior
                * np.log(np.maximum(posterior, 1e-300) / np.maximum(prior, 1e-300))
            )
        )
        self.last_update_used_product_increment = True
        self._pending_product_evidence = True

    def _credible_mask(self, weights: Array) -> Array:
        cumulative = np.cumsum(weights)
        tail = 0.5 * (1.0 - self.config.target_credible_level)
        low = min(int(np.searchsorted(cumulative, tail, side="left")), len(weights) - 1)
        high = min(
            int(np.searchsorted(cumulative, 1.0 - tail, side="left")),
            len(weights) - 1,
        )
        minimum = min(self.config.target_min_refinement_candidates, len(weights))
        while high - low + 1 < minimum:
            if low > 0:
                low -= 1
            if high - low + 1 < minimum and high < len(weights) - 1:
                high += 1
            if low == 0 and high == len(weights) - 1:
                break
        mask = np.zeros(len(weights), dtype=bool)
        mask[low : high + 1] = True
        self.last_real_interval = (float(self.edges[low]), float(self.edges[high + 1]))
        return mask

    def refine(
        self,
        state: Array,
        estimator: IntervalBayesEstimator,
        parameters: Mapping[str, float],
    ) -> None:
        parameter_weights = np.asarray(estimator.weights, dtype=float)
        parameter_weights /= parameter_weights.sum()
        likelihood = np.asarray(self._pending_product_likelihood, dtype=float)
        if likelihood.shape != parameter_weights.shape:
            likelihood = np.ones_like(parameter_weights)
        branch_weights = (
            parameter_weights * likelihood
            if self._pending_product_evidence
            else parameter_weights.copy()
        )
        branch_total = branch_weights.sum()
        branch_weights = (
            branch_weights / branch_total
            if branch_total > 1e-300
            else parameter_weights
        )
        reward = predicted_product_increment_surface(
            state,
            estimator.centers,
            self.centers,
            self.config.control_interval_h,
            self.config,
            parameters,
        )
        best_indices = np.argmax(reward, axis=1)
        branch_optima = self.centers[best_indices]
        kernel_std = max(self.config.target_kernel_std_gpl, 1e-9)
        kernels = np.exp(
            -0.5
            * np.square((self.centers[None, :] - branch_optima[:, None]) / kernel_std)
        )
        kernels /= np.maximum(kernels.sum(axis=1, keepdims=True), 1e-300)
        target_mixing = float(
            np.clip(self.config.target_real_prior_mixing_probability, 0.0, 1.0)
        )
        target_prior = (1.0 - target_mixing) * self.real_weights + target_mixing / len(
            self.real_weights
        )
        real_evidence = np.einsum("i,ij->j", branch_weights, kernels)
        real_weights = target_prior * np.maximum(real_evidence, 1e-300)
        real_weights /= real_weights.sum()
        credible_mask = self._credible_mask(real_weights)

        real_conditionals = kernels * real_weights[None, :]
        real_conditionals[:, ~credible_mask] = 0.0
        real_conditional_total = real_conditionals.sum(axis=1, keepdims=True)
        real_conditionals = np.divide(
            real_conditionals,
            real_conditional_total,
            out=kernels.copy(),
            where=real_conditional_total > 1e-300,
        )
        virtual_prior = np.zeros_like(real_weights)
        virtual_prior[credible_mask] = 1.0 / int(np.count_nonzero(credible_mask))
        self.virtual_prior = virtual_prior
        conditionals = real_conditionals.copy()
        if self.config.proposed_use_virtual_posterior:
            expected_reward = np.einsum("i,ij->j", branch_weights, reward)
            reward_variance = np.maximum(
                np.einsum("i,ij->j", branch_weights, np.square(reward))
                - np.square(expected_reward),
                0.0,
            )
            influence = expected_reward + (
                self.config.target_virtual_exploration_weight * np.sqrt(reward_variance)
            )
            selected = influence[credible_mask]
            scale = max(float(np.ptp(selected)), 1e-12)
            normalized = (influence - float(np.min(selected))) / scale
            virtual_factor = np.zeros_like(normalized)
            virtual_factor[credible_mask] = np.exp(
                self.config.target_virtual_temperature
                * (normalized[credible_mask] - float(np.max(normalized[credible_mask])))
            )
            branch_scale = np.maximum(
                np.ptp(reward[:, credible_mask], axis=1, keepdims=True),
                1e-12,
            )
            branch_min = np.min(reward[:, credible_mask], axis=1, keepdims=True)
            branch_score = (reward - branch_min) / branch_scale
            conditionals = (
                virtual_prior[None, :]
                * kernels
                * virtual_factor[None, :]
                * np.exp(self.config.target_virtual_temperature * branch_score)
            )
            conditional_total = conditionals.sum(axis=1, keepdims=True)
            conditionals = np.divide(
                conditionals,
                conditional_total,
                out=kernels.copy(),
                where=conditional_total > 1e-300,
            )
            self.last_virtual_reward_variance = float(
                np.dot(real_weights, reward_variance)
            )
            self.last_virtual_refinement_used = True
        else:
            self.last_virtual_reward_variance = 0.0
            self.last_virtual_refinement_used = False

        self.real_weights = real_weights
        self.real_paired_targets = np.einsum("ij,j->i", real_conditionals, self.centers)
        self.paired_targets = np.einsum("ij,j->i", conditionals, self.centers)
        self.paired_target_variances = np.maximum(
            np.einsum("ij,j->i", conditionals, np.square(self.centers))
            - np.square(self.paired_targets),
            0.0,
        )
        self.paired_weights = branch_weights
        self.weights = np.einsum("i,ij->j", branch_weights, conditionals)
        self.weights /= self.weights.sum()
        self._pending_product_likelihood.fill(1.0)
        self._pending_product_evidence = False

    @property
    def mean(self) -> float:
        return float(np.dot(self.centers, self.weights))

    @property
    def variance(self) -> float:
        return float(np.dot(np.square(self.centers - self.mean), self.weights))

    @property
    def real_variance(self) -> float:
        real_mean = float(np.dot(self.centers, self.real_weights))
        return float(np.dot(np.square(self.centers - real_mean), self.real_weights))

    def credible_interval(self, level: float) -> Tuple[float, float]:
        tail = 0.5 * (1.0 - level)
        cumulative = np.cumsum(self.weights)
        low = min(
            int(np.searchsorted(cumulative, tail, side="left")), len(self.weights) - 1
        )
        high = min(
            int(np.searchsorted(cumulative, 1.0 - tail, side="left")),
            len(self.weights) - 1,
        )
        return float(self.edges[low]), float(self.edges[high + 1])


def braking_feed_volume_l(
    input_rate_lph: float,
    move_limit_lph: float,
    control_interval_h: float,
) -> float:
    """Minimum volume needed to ramp a nonnegative input down to zero."""
    input_rate_lph = max(float(input_rate_lph), 0.0)
    if input_rate_lph == 0.0:
        return 0.0
    if move_limit_lph <= 0.0:
        return math.inf
    n_positive = int(math.ceil(input_rate_lph / move_limit_lph))
    rate_sum = (
        n_positive * input_rate_lph
        - move_limit_lph * n_positive * (n_positive - 1) / 2.0
    )
    return float(control_interval_h * rate_sum)


def capacity_safe_input_upper(
    remaining_volume_l: float, config: SimulationConfig
) -> float:
    """Largest input that can still be ramped to zero before capacity is exhausted."""
    remaining_volume_l = max(float(remaining_volume_l), 0.0)
    immediate_upper = remaining_volume_l / config.control_interval_h
    if not config.reserve_volume_for_ramp_down:
        return min(config.u_max_lph, immediate_upper)
    rate_budget = remaining_volume_l / config.control_interval_h
    safe_upper = 0.0
    max_segments = int(math.ceil(config.u_max_lph / config.du_max_lph))
    for n_positive in range(1, max_segments + 1):
        segment_lower = (n_positive - 1) * config.du_max_lph
        segment_upper = min(n_positive * config.du_max_lph, config.u_max_lph)
        candidate = (
            rate_budget + config.du_max_lph * n_positive * (n_positive - 1) / 2.0
        ) / n_positive
        candidate = min(candidate, segment_upper)
        if candidate + 1e-12 >= segment_lower:
            safe_upper = max(safe_upper, candidate)
    return min(safe_upper, immediate_upper, config.u_max_lph)


def project_input(
    raw_u: float,
    previous_u: float,
    config: SimulationConfig,
    current_volume: float | None = None,
) -> float:
    lower, upper = admissible_input_bounds(previous_u, config, current_volume)
    return float(np.clip(raw_u, lower, upper))


def admissible_input_bounds(
    previous_u: float,
    config: SimulationConfig,
    current_volume: float | None = None,
) -> Tuple[float, float]:
    lower = max(config.u_min_lph, previous_u - config.du_max_lph)
    upper = min(config.u_max_lph, previous_u + config.du_max_lph)
    if current_volume is not None:
        capacity_upper = capacity_safe_input_upper(
            config.volume_max_l - current_volume, config
        )
        upper = min(upper, capacity_upper)
        lower = min(lower, upper)
    return float(lower), float(upper)


class ProposedDualController:

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.last_u = 0.0
        self.last_diagnostics: Dict[str, float] = {}

    def _closed_form_action(
        self,
        f_support: Array,
        g_support: Array,
        target_support: Array,
        weights: Array,
        lower: float,
        upper: float,
        include_uncertainty: bool,
        retain_cross_covariance: bool,
    ) -> float:
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        f_support = np.asarray(f_support, dtype=float)
        g_support = np.asarray(g_support, dtype=float)
        target_support = np.asarray(target_support, dtype=float)

        if include_uncertainty:
            quadratic = float(np.dot(weights, np.square(g_support)))
            if retain_cross_covariance:
                linear = float(
                    np.dot(weights, g_support * (f_support - target_support))
                )
            else:
                linear = float(
                    np.dot(weights, g_support * f_support)
                    - np.dot(weights, g_support) * np.dot(weights, target_support)
                )
        else:
            mean_f = float(np.dot(weights, f_support))
            mean_g = float(np.dot(weights, g_support))
            mean_target = float(np.dot(weights, target_support))
            quadratic = mean_g**2
            linear = mean_g * (mean_f - mean_target)

        denominator = (
            self.config.proposed_q * quadratic
            + self.config.proposed_r
            + self.config.proposed_regularization
        )
        numerator = self.config.proposed_q * linear
        unconstrained = (
            self.last_u if denominator <= 1e-15 else -numerator / denominator
        )
        return float(np.clip(unconstrained, lower, upper))

    def solve(
        self,
        x: Array,
        estimator: IntervalBayesEstimator,
        target_learner: HierarchicalSubstrateLearner,
        parameters: Mapping[str, float],
    ) -> Tuple[float, float]:
        state = np.asarray(x, dtype=float)
        previous_u = self.last_u
        lower, upper = admissible_input_bounds(
            previous_u,
            self.config,
            float(state[3]),
        )
        centers = np.asarray(estimator.centers, dtype=float)
        weights = np.asarray(target_learner.paired_weights, dtype=float)
        weights /= weights.sum()
        targets = np.asarray(target_learner.paired_targets, dtype=float)
        real_targets = np.asarray(target_learner.real_paired_targets, dtype=float)

        lower_states, _ = integrate_parameter_support(
            state,
            [lower],
            self.config.control_interval_h,
            self.config.integration_substeps,
            centers,
            self.config.nominal_mu_max_per_h,
            self.config.nominal_k_sp_gpl,
            parameters,
        )
        if upper - lower > 1e-12:
            upper_states, _ = integrate_parameter_support(
                state,
                [upper],
                self.config.control_interval_h,
                self.config.integration_substeps,
                centers,
                self.config.nominal_mu_max_per_h,
                self.config.nominal_k_sp_gpl,
                parameters,
            )
            g_support = (upper_states[:, 2] - lower_states[:, 2]) / (upper - lower)
            f_support = lower_states[:, 2] - g_support * lower
        else:
            g_support = np.zeros_like(centers)
            f_support = lower_states[:, 2]

        paired_action = self._closed_form_action(
            f_support,
            g_support,
            targets,
            weights,
            lower,
            upper,
            include_uncertainty=True,
            retain_cross_covariance=True,
        )
        diagonal_action = self._closed_form_action(
            f_support,
            g_support,
            targets,
            weights,
            lower,
            upper,
            include_uncertainty=True,
            retain_cross_covariance=False,
        )
        mean_only_action = self._closed_form_action(
            f_support,
            g_support,
            targets,
            weights,
            lower,
            upper,
            include_uncertainty=False,
            retain_cross_covariance=False,
        )
        real_posterior_action = self._closed_form_action(
            f_support,
            g_support,
            real_targets,
            weights,
            lower,
            upper,
            include_uncertainty=True,
            retain_cross_covariance=self.config.retain_output_target_cross_covariance,
        )
        distribution_action = (
            paired_action
            if self.config.retain_output_target_cross_covariance
            else diagonal_action
        )
        u = (
            distribution_action
            if self.config.proposed_include_posterior_uncertainty
            else mean_only_action
        )

        predicted_substrate = f_support + g_support * u
        mean_output = float(np.dot(weights, predicted_substrate))
        mean_target = float(np.dot(weights, targets))
        output_variance = float(
            np.dot(weights, np.square(predicted_substrate - mean_output))
        )
        between_target_variance = float(
            np.dot(weights, np.square(targets - mean_target))
        )
        conditional_target_variance = float(
            np.dot(weights, target_learner.paired_target_variances)
        )
        target_variance = between_target_variance + conditional_target_variance
        covariance = float(
            np.dot(
                weights,
                (predicted_substrate - mean_output) * (targets - mean_target),
            )
        )
        mean_tracking_error = (mean_output - mean_target) ** 2
        paired_uncertainty = max(
            output_variance + target_variance - 2.0 * covariance,
            0.0,
        )
        diagonal_uncertainty = output_variance + target_variance
        paired_tracking_cost = self.config.proposed_q * (
            mean_tracking_error + paired_uncertainty
        )
        diagonal_tracking_cost = self.config.proposed_q * (
            mean_tracking_error + diagonal_uncertainty
        )
        gain_mean = float(np.dot(weights, g_support))
        gain_variance = float(np.dot(weights, np.square(g_support - gain_mean)))
        quadratic = float(np.dot(weights, np.square(g_support)))
        linear = float(np.dot(weights, g_support * (f_support - targets)))
        if not self.config.retain_output_target_cross_covariance:
            linear = float(np.dot(weights, g_support * f_support) - gain_mean * mean_target)
        if not self.config.proposed_include_posterior_uncertainty:
            quadratic = gain_mean**2
            linear = gain_mean * (float(np.dot(weights, f_support)) - mean_target)

        self.last_u = u
        self.last_diagnostics = {
            "posterior_support_evaluations": float(len(centers)),
            "virtual_refinement_used": float(
                target_learner.last_virtual_refinement_used
            ),
            "parameter_update_used_state_observation": float(
                estimator.last_update_used_state_observation
            ),
            "target_update_used_product_increment": float(
                target_learner.last_update_used_product_increment
            ),
            "predicted_target_covariance": covariance,
            "output_variance": output_variance,
            "target_variance": target_variance,
            "cross_contribution": -2.0 * covariance,
            "covariance_ratio": abs(2.0 * covariance)
            / max(output_variance + target_variance, 1e-12),
            "paired_tracking_cost": paired_tracking_cost,
            "diagonal_tracking_cost": diagonal_tracking_cost,
            "uncertainty_cost": self.config.proposed_q
            * (
                paired_uncertainty
                if self.config.retain_output_target_cross_covariance
                else diagonal_uncertainty
            ),
            "expected_kip_variance": estimator.variance,
            "prior_kip_variance": estimator.last_prior_variance,
            "expected_target_variance": target_variance,
            "prior_target_variance": target_learner.real_variance,
            "virtual_refinement_reward_variance": (
                target_learner.last_virtual_reward_variance
            ),
            "support_input_gain_range": float(np.ptp(g_support)),
            "parameter_input_gain_variance": gain_variance,
            "uncertainty_action_shift": float(distribution_action - mean_only_action),
            "virtual_refinement_action_shift": float(
                distribution_action - real_posterior_action
            ),
            "paired_diagonal_action_difference": float(paired_action - diagonal_action),
            "analytic_quadratic_coefficient": (
                self.config.proposed_q * quadratic
                + self.config.proposed_r
                + self.config.proposed_regularization
            ),
            "analytic_linear_coefficient": self.config.proposed_q * linear,
        }
        return u, mean_target

    def set_applied_input(self, applied_u: float) -> None:
        self.last_u = float(applied_u)


@dataclass(frozen=True)
class Observation:
    state: Array
    dpdt: float
    u: float
    acquisition_step: int


def direct_control_state(observation: Observation, known_volume_l: float) -> Array:
    state = np.asarray(observation.state, dtype=float).copy()
    state[:3] = np.maximum(state[:3], 0.0)
    state[3] = max(float(known_volume_l), 1e-9)
    return state


class MeasurementChannel:

    def __init__(
        self, stride: int, delay_steps: int, initial_observation: Observation
    ) -> None:
        self.stride = stride
        self.delay_steps = delay_steps
        self.pending: deque[Tuple[int, Observation]] = deque()
        self.latest = initial_observation

    def submit(self, step: int, observation: Observation) -> Tuple[Observation, bool]:
        if step % self.stride == 0:
            self.pending.append((step + self.delay_steps, observation))
        released = False
        while self.pending and self.pending[0][0] <= step:
            _, self.latest = self.pending.popleft()
            released = True
        return self.latest, released


@dataclass
class SimulationResult:
    scenario: ScenarioConfig
    seed: int
    history: Dict[str, object]
    metrics: pd.DataFrame


def build_true_profiles(
    config: SimulationConfig,
    scenario: ScenarioConfig,
    rng: np.random.Generator,
    time_step_h: float | None = None,
) -> Dict[str, Array]:
    profile_step_h = (
        config.control_interval_h if time_step_h is None else float(time_step_h)
    )
    profile_steps = int(round(config.total_time_h / profile_step_h))
    if abs(profile_steps * profile_step_h - config.total_time_h) > 1e-9:
        raise ValueError("Profile time step must divide the total simulation time")
    times = np.linspace(0.0, config.total_time_h, profile_steps + 1)
    fraction = times / config.total_time_h
    if scenario.profile == "piecewise":
        kip = np.where(
            fraction < 1.0 / 3.0, 12.0, np.where(fraction < 2.0 / 3.0, 60.0, 10.0)
        )
    elif scenario.profile == "gradual_wide":
        knots = np.array([0.0, 0.28, 0.34, 0.62, 0.68, 1.0])
        values = np.array([10.0, 10.0, 70.0, 70.0, 10.0, 10.0])
        kip = np.empty_like(fraction)
        for segment in range(len(knots) - 1):
            mask = (fraction >= knots[segment]) & (fraction <= knots[segment + 1])
            local = (fraction[mask] - knots[segment]) / (
                knots[segment + 1] - knots[segment]
            )
            blend = 0.5 - 0.5 * np.cos(np.pi * local)
            kip[mask] = values[segment] + blend * (
                values[segment + 1] - values[segment]
            )
    elif scenario.profile == "stochastic_regime":
        kip = np.empty_like(times)
        kip[0] = 10.0
        reversion_per_h = 0.35
        diffusion = 1.5
        for index in range(1, len(times)):
            dt = profile_step_h
            local_fraction = fraction[index]
            mean_value = (
                10.0
                if local_fraction < 0.30
                else (70.0 if local_fraction < 0.65 else 10.0)
            )
            kip[index] = (
                kip[index - 1]
                + reversion_per_h * (mean_value - kip[index - 1]) * dt
                + diffusion * math.sqrt(dt) * rng.normal()
            )
        kip = np.clip(kip, config.kip_min, config.kip_max)
    elif scenario.profile == "correlated_multi_parameter":
        latent = np.sin(2.0 * np.pi * fraction) + 0.35 * np.sin(6.0 * np.pi * fraction)
        regime = np.where(fraction < 0.30, 12.0, np.where(fraction < 0.65, 66.0, 10.0))
        kip = np.clip(regime + 3.0 * latent, config.kip_min, config.kip_max)
    elif scenario.profile == "delayed_piecewise":
        kip = np.where(fraction < 0.30, 12.0, np.where(fraction < 0.67, 65.0, 10.0))
    elif scenario.profile == "mutation_contamination":
        kip = np.where(
            fraction < 1.0 / 3.0,
            10.0,
            np.where(fraction < 2.0 / 3.0, 70.0, 10.0),
        )
    else:
        raise ValueError(f"Unknown profile type: {scenario.profile}")

    latent = (
        np.sin(2.0 * np.pi * fraction)
        if scenario.profile != "correlated_multi_parameter"
        else (np.sin(2.0 * np.pi * fraction) + 0.35 * np.sin(6.0 * np.pi * fraction))
    )
    mu_max = config.nominal_mu_max_per_h * (1.0 + scenario.mu_mismatch_fraction)
    k_sp = config.nominal_k_sp_gpl * (1.0 + scenario.ksp_mismatch_fraction)
    mu_profile = np.full_like(times, mu_max)
    ksp_profile = np.full_like(times, k_sp)
    if scenario.profile == "correlated_multi_parameter":
        mu_profile *= 1.0 + 0.15 * latent
        ksp_profile *= 1.0 + 0.20 * latent
    return {"time_h": times, "K_IP": kip, "mu_max": mu_profile, "K_sp": ksp_profile}


def _initial_product_rate(
    x0: Array,
    u0: float,
    profile: Mapping[str, Array],
    parameters: Mapping[str, float],
) -> float:
    true_parameters = dict(parameters)
    true_parameters["K_IP"] = float(profile["K_IP"][0])
    return float(
        fermentation_rhs(
            x0,
            u0,
            float(profile["mu_max"][0]),
            float(profile["K_sp"][0]),
            true_parameters,
        )[1]
    )


def _adaptation_lags(
    estimate: Array,
    truth: Array,
    config: SimulationConfig,
    scenario: ScenarioConfig,
) -> Dict[str, float]:
    result = {
        "adaptation_lag_h": math.nan,
        "adaptation_lag_up_h": math.nan,
        "adaptation_lag_down_h": math.nan,
        "adaptation_transitions_converged": math.nan,
        "adaptation_transition_count": math.nan,
    }
    if scenario.profile not in {
        "piecewise",
        "delayed_piecewise",
        "mutation_contamination",
    }:
        return result
    transition_indices = (np.flatnonzero(~np.isclose(np.diff(truth), 0.0)) + 1).tolist()
    lags: list[float] = []
    directions: list[str] = []
    for transition_number, transition in enumerate(transition_indices):
        end = (
            transition_indices[transition_number + 1]
            if transition_number + 1 < len(transition_indices)
            else len(estimate)
        )
        jump = abs(truth[transition] - truth[transition - 1])
        tolerance = max(2.0, 0.10 * jump)
        found = math.nan
        for index in range(transition, max(transition, end - 2)):
            if np.all(
                np.abs(estimate[index : index + 3] - truth[index : index + 3])
                <= tolerance
            ):
                found = (index - transition) * config.control_interval_h
                break
        lags.append(found)
        directions.append("up" if truth[transition] > truth[transition - 1] else "down")
    for direction, lag in zip(directions, lags):
        result[f"adaptation_lag_{direction}_h"] = lag
    finite = np.asarray(lags, dtype=float)[np.isfinite(lags)]
    result["adaptation_transitions_converged"] = float(finite.size)
    result["adaptation_transition_count"] = float(len(lags))
    if len(lags) > 0 and finite.size == len(lags):
        result["adaptation_lag_h"] = float(np.mean(finite))
    return result




def run_simulation(
    config: SimulationConfig, scenario: ScenarioConfig, seed: int
) -> SimulationResult:
    seed_sequence = np.random.SeedSequence(seed)
    # Preserve the benchmark's child streams for exact seed matching.
    profile_seed, noise_seed, _unused_seed = seed_sequence.spawn(3)
    profile_rng = np.random.default_rng(profile_seed)
    noise_rng = np.random.default_rng(noise_seed)
    parameters = dict(BASE_PARAMETERS)
    event_profile = build_true_profiles(
        config, scenario, profile_rng, time_step_h=config.measurement_interval_h
    )
    n_events = config.n_measurement_steps
    n_steps = config.n_control_steps
    control_stride = config.control_measurement_steps
    posterior_stride = config.posterior_measurement_steps
    if n_events != n_steps * control_stride:
        raise ValueError("Control and measurement clocks do not cover the same horizon")
    if tuple(config.algorithms) != ("GOAC",):
        raise ValueError("This standalone implementation supports GOAC only")
    algorithms = ("GOAC",)
    feedback_methods = algorithms
    plants: Dict[str, Array] = {
        name: np.asarray(scenario.x0, dtype=float).copy() for name in algorithms
    }
    controls: Dict[str, float] = {name: 0.0 for name in algorithms}
    state_noise = noise_rng.normal(size=(n_events + 1, 3))
    dpdt_noise = noise_rng.normal(size=n_events + 1)
    interval_likelihood_std = (
        config.interval_likelihood_std * scenario.dpdt_noise_std / 0.05
    )
    interval_estimator = IntervalBayesEstimator(
        config.kip_min,
        config.kip_max,
        config.n_kip_intervals,
        interval_likelihood_std,
        config.interval_diffusion_probability,
        config.interval_innovation_threshold_std,
        config.interval_innovation_persistence,
        config.interval_innovation_mixing_probability,
    )
    target_learner = HierarchicalSubstrateLearner(config)
    proposed = ProposedDualController(config)
    controller_objects = {"GOAC": proposed}
    inventory_volumes = {name: float(scenario.x0[3]) for name in algorithms}
    acquisition_strides: Dict[str, int] = {}
    for name in algorithms:
        if scenario.measurement_control_stride is not None:
            stride = int(scenario.measurement_control_stride) * control_stride
        elif name == "GOAC":
            stride = int(scenario.measurement_stride)
        else:
            stride = int(scenario.measurement_stride) * control_stride
        acquisition_strides[name] = max(1, stride)
    delay_ratio = scenario.measurement_delay_h / config.measurement_interval_h
    delay_events = int(round(delay_ratio))
    if abs(delay_ratio - delay_events) > 1e-09:
        raise ValueError(
            "measurement_delay_h must be an integer multiple of measurement_interval_h"
        )
    channels: Dict[str, MeasurementChannel] = {}
    latest_observations: Dict[str, Observation] = {}
    for name in algorithms:
        initial_rate = _initial_product_rate(
            plants[name], controls[name], event_profile, parameters
        )
        initial_observation = Observation(
            plants[name].copy(), initial_rate, controls[name], 0
        )
        channels[name] = MeasurementChannel(
            acquisition_strides[name], delay_events, initial_observation
        )
        latest_observations[name] = initial_observation
    previous_goac_learning_observation = (
        channels["GOAC"].latest if "GOAC" in channels else None
    )
    pending_goac_observations: deque[Observation] = deque()
    goac_event_controls = np.zeros(n_events, dtype=float)
    released_since_control = {name: 0 for name in algorithms}
    goac_posterior_events_since_control = 0
    goac_parameter_updates_since_control = 0
    goac_target_updates_since_control = 0
    goac_virtual_updates_since_control = 0
    goac_learning_time_since_control = 0.0
    latest_posterior_time_h = math.nan
    sampled_profile = {
        key: np.asarray(value)[::control_stride] for key, value in event_profile.items()
    }
    history: Dict[str, object] = {
        "time_h": sampled_profile["time_h"],
        "true_K_IP": sampled_profile["K_IP"],
        "true_mu_max": sampled_profile["mu_max"],
        "true_K_sp": sampled_profile["K_sp"],
        "measurement_event_counts": {name: 0 for name in algorithms},
        "measurement_acquisition_strides": acquisition_strides.copy(),
        "states": {name: np.zeros((n_steps + 1, 4)) for name in algorithms},
        "controller_states": {
            name: np.full((n_steps + 1, 4), np.nan) for name in feedback_methods
        },
        "controls": {name: np.zeros(n_steps) for name in algorithms},
        "unfiltered_controls": {name: np.zeros(n_steps) for name in algorithms},
        "safety_interventions": {
            name: np.zeros(n_steps, dtype=bool) for name in algorithms
        },
        "targets": {name: np.full(n_steps, np.nan) for name in algorithms},
        "controller_time_s": {name: np.zeros(n_steps) for name in algorithms},
        "solver_success": {name: np.ones(n_steps, dtype=bool) for name in algorithms},
        "solver_status": {name: np.zeros(n_steps, dtype=int) for name in algorithms},
        "solver_iterations": {
            name: np.zeros(n_steps, dtype=int) for name in algorithms
        },
        "solver_messages": {
            name: np.full(n_steps, "not applicable", dtype=object)
            for name in algorithms
        },
        "kip_mean": {"GOAC": np.full(n_steps, np.nan)},
        "kip_lower": {"GOAC": np.full(n_steps, np.nan)},
        "kip_upper": {"GOAC": np.full(n_steps, np.nan)},
        "proposed_posterior_support_evaluations": np.full(n_steps, np.nan),
        "proposed_target_covariance": np.full(n_steps, np.nan),
        "proposed_output_variance": np.full(n_steps, np.nan),
        "proposed_target_variance": np.full(n_steps, np.nan),
        "proposed_cross_contribution": np.full(n_steps, np.nan),
        "proposed_covariance_ratio": np.full(n_steps, np.nan),
        "proposed_paired_tracking_cost": np.full(n_steps, np.nan),
        "proposed_diagonal_tracking_cost": np.full(n_steps, np.nan),
        "proposed_expected_kip_variance": np.full(n_steps, np.nan),
        "proposed_prior_kip_variance": np.full(n_steps, np.nan),
        "proposed_prior_target_variance": np.full(n_steps, np.nan),
        "proposed_virtual_reward_variance": np.full(n_steps, np.nan),
        "proposed_uncertainty_cost": np.full(n_steps, np.nan),
        "proposed_uncertainty_action_shift": np.full(n_steps, np.nan),
        "proposed_virtual_refinement_action_shift": np.full(n_steps, np.nan),
        "proposed_paired_diagonal_action_difference": np.full(n_steps, np.nan),
        "proposed_innovation_sigma": np.full(n_steps, np.nan),
        "proposed_adaptive_mixing": np.zeros(n_steps),
        "proposed_parameter_state_update": np.zeros(n_steps, dtype=bool),
        "proposed_target_product_update": np.zeros(n_steps, dtype=bool),
        "proposed_virtual_refinement_used": np.zeros(n_steps, dtype=bool),
        "proposed_parameter_information_gain": np.full(n_steps, np.nan),
        "proposed_target_information_gain": np.full(n_steps, np.nan),
        "proposed_support_input_gain_range": np.full(n_steps, np.nan),
        "proposed_target_lower": np.full(n_steps, np.nan),
        "proposed_target_upper": np.full(n_steps, np.nan),
        "proposed_measurement_events": np.zeros(n_steps, dtype=int),
        "proposed_posterior_events": np.zeros(n_steps, dtype=int),
        "proposed_parameter_update_count": np.zeros(n_steps, dtype=int),
        "proposed_target_update_count": np.zeros(n_steps, dtype=int),
        "proposed_posterior_age_h": np.full(n_steps, np.nan),
        "proposed_virtual_prior_uniform_error": np.full(n_steps, np.nan),
    }
    formed_product_mass = {name: 0.0 for name in algorithms}
    event_process_std = np.asarray(config.state_process_std) * math.sqrt(
        config.measurement_interval_h / config.control_interval_h
    )
    for event_step in range(n_events):
        current_time_h = event_step * config.measurement_interval_h
        true_parameters = dict(parameters)
        true_parameters["K_IP"] = float(event_profile["K_IP"][event_step])
        observations: Dict[str, Tuple[Observation, bool]] = {}
        for name in algorithms:
            state = plants[name]
            dpdt = fermentation_rhs(
                state,
                controls[name],
                float(event_profile["mu_max"][event_step]),
                float(event_profile["K_sp"][event_step]),
                true_parameters,
            )[1]
            measured_state = state.copy()
            measured_state[:3] += state_noise[event_step] * np.asarray(
                scenario.state_noise_std
            )
            measured_state[:3] = np.maximum(measured_state[:3], 0.0)
            candidate_observation = Observation(
                measured_state,
                float(dpdt + dpdt_noise[event_step] * scenario.dpdt_noise_std),
                controls[name],
                event_step,
            )
            observation, is_new = channels[name].submit(
                event_step, candidate_observation
            )
            observations[name] = (observation, is_new)
            latest_observations[name] = observation
            if is_new:
                history["measurement_event_counts"][name] += 1
                released_since_control[name] += 1
                if name == "GOAC":
                    pending_goac_observations.append(observation)
        posterior_due = event_step % posterior_stride == 0
        if "GOAC" in algorithms and posterior_due and pending_goac_observations:
            posterior_start = time.perf_counter_ns()
            interval_estimator.last_update_used_state_observation = False
            target_learner.last_update_used_product_increment = False
            parameter_updates_this_event = 0
            target_updates_this_event = 0
            while pending_goac_observations:
                observation = pending_goac_observations.popleft()
                if (
                    previous_goac_learning_observation is not None
                    and observation.acquisition_step
                    > previous_goac_learning_observation.acquisition_step
                ):
                    first = previous_goac_learning_observation.acquisition_step
                    last = observation.acquisition_step
                    applied_controls = goac_event_controls[first:last]
                    interval_estimator.update_from_state_transition(
                        previous_goac_learning_observation.state,
                        observation.state,
                        applied_controls,
                        config.nominal_mu_max_per_h,
                        config.nominal_k_sp_gpl,
                        parameters,
                        config.measurement_interval_h,
                        config.measurement_substeps,
                        np.asarray(scenario.state_noise_std),
                        event_process_std,
                    )
                    if interval_estimator.last_update_used_state_observation:
                        parameter_updates_this_event += 1
                    measured_product_increment = float(
                        observation.state[1] * observation.state[3]
                        - previous_goac_learning_observation.state[1]
                        * previous_goac_learning_observation.state[3]
                    )
                    elapsed_h = len(applied_controls) * config.measurement_interval_h
                    average_volume = 0.5 * (
                        observation.state[3]
                        + previous_goac_learning_observation.state[3]
                    )
                    product_increment_std = (
                        config.target_product_increment_noise_multiplier
                        * math.sqrt(
                            (
                                scenario.state_noise_std[1]
                                * previous_goac_learning_observation.state[3]
                            )
                            ** 2
                            + (scenario.state_noise_std[1] * observation.state[3]) ** 2
                            + (scenario.dpdt_noise_std * elapsed_h * average_volume)
                            ** 2
                        )
                    )
                    target_learner.observe_product_increment(
                        measured_product_increment,
                        interval_estimator.last_product_increment_predictions,
                        interval_estimator.weights,
                        product_increment_std,
                    )
                    target_updates_this_event += 1
                previous_goac_learning_observation = observation
            target_learner.refine(
                direct_control_state(
                    latest_observations["GOAC"], inventory_volumes["GOAC"]
                ),
                interval_estimator,
                parameters,
            )
            latest_posterior_time_h = current_time_h
            goac_posterior_events_since_control += 1
            goac_parameter_updates_since_control += parameter_updates_this_event
            goac_target_updates_since_control += target_updates_this_event
            goac_virtual_updates_since_control += int(
                target_learner.last_virtual_refinement_used
            )
            goac_learning_time_since_control += (
                time.perf_counter_ns() - posterior_start
            ) * 1e-09
        control_due = event_step % control_stride == 0
        if control_due:
            step = event_step // control_stride
            previous_controls = controls.copy()
            for name in algorithms:
                history["states"][name][step] = plants[name]
            control_states = {
                name: direct_control_state(
                    latest_observations[name], inventory_volumes[name]
                )
                for name in feedback_methods
            }
            for name, state in control_states.items():
                history["controller_states"][name][step] = state
            if "GOAC" in algorithms:
                decision_start = time.perf_counter_ns()
                interval_estimator.last_update_used_state_observation = (
                    goac_parameter_updates_since_control > 0
                )
                target_learner.last_update_used_product_increment = (
                    goac_target_updates_since_control > 0
                )
                controls["GOAC"], target = proposed.solve(
                    control_states["GOAC"],
                    interval_estimator,
                    target_learner,
                    parameters,
                )
                history["controller_time_s"]["GOAC"][step] = (
                    goac_learning_time_since_control
                    + (time.perf_counter_ns() - decision_start) * 1e-09
                )
                history["targets"]["GOAC"][step] = target
                history["kip_mean"]["GOAC"][step] = interval_estimator.mean
                low, high = interval_estimator.credible_interval(
                    config.confidence_level
                )
                history["kip_lower"]["GOAC"][step] = low
                history["kip_upper"]["GOAC"][step] = high
                target_low, target_high = target_learner.credible_interval(
                    config.confidence_level
                )
                history["proposed_target_lower"][step] = target_low
                history["proposed_target_upper"][step] = target_high
                diagnostics = proposed.last_diagnostics
                history["proposed_posterior_support_evaluations"][step] = diagnostics[
                    "posterior_support_evaluations"
                ]
                history["proposed_target_covariance"][step] = diagnostics[
                    "predicted_target_covariance"
                ]
                history["proposed_output_variance"][step] = diagnostics[
                    "output_variance"
                ]
                history["proposed_target_variance"][step] = diagnostics[
                    "target_variance"
                ]
                history["proposed_cross_contribution"][step] = diagnostics[
                    "cross_contribution"
                ]
                history["proposed_covariance_ratio"][step] = diagnostics[
                    "covariance_ratio"
                ]
                history["proposed_paired_tracking_cost"][step] = diagnostics[
                    "paired_tracking_cost"
                ]
                history["proposed_diagonal_tracking_cost"][step] = diagnostics[
                    "diagonal_tracking_cost"
                ]
                for history_key, diagnostic_key in (
                    ("proposed_expected_kip_variance", "expected_kip_variance"),
                    ("proposed_prior_kip_variance", "prior_kip_variance"),
                    ("proposed_prior_target_variance", "prior_target_variance"),
                    (
                        "proposed_virtual_reward_variance",
                        "virtual_refinement_reward_variance",
                    ),
                    ("proposed_uncertainty_cost", "uncertainty_cost"),
                    ("proposed_uncertainty_action_shift", "uncertainty_action_shift"),
                    (
                        "proposed_virtual_refinement_action_shift",
                        "virtual_refinement_action_shift",
                    ),
                    (
                        "proposed_paired_diagonal_action_difference",
                        "paired_diagonal_action_difference",
                    ),
                ):
                    history[history_key][step] = diagnostics[diagnostic_key]
                history["proposed_innovation_sigma"][
                    step
                ] = interval_estimator.last_innovation_sigma
                history["proposed_adaptive_mixing"][
                    step
                ] = interval_estimator.last_mixing_probability
                history["proposed_parameter_state_update"][step] = (
                    goac_parameter_updates_since_control > 0
                )
                history["proposed_target_product_update"][step] = (
                    goac_target_updates_since_control > 0
                )
                history["proposed_virtual_refinement_used"][step] = (
                    goac_virtual_updates_since_control > 0
                )
                history["proposed_parameter_information_gain"][
                    step
                ] = interval_estimator.last_information_gain
                history["proposed_target_information_gain"][
                    step
                ] = target_learner.last_product_information_gain
                history["proposed_support_input_gain_range"][step] = diagnostics[
                    "support_input_gain_range"
                ]
                history["proposed_measurement_events"][step] = released_since_control[
                    "GOAC"
                ]
                history["proposed_posterior_events"][
                    step
                ] = goac_posterior_events_since_control
                history["proposed_parameter_update_count"][
                    step
                ] = goac_parameter_updates_since_control
                history["proposed_target_update_count"][
                    step
                ] = goac_target_updates_since_control
                if math.isfinite(latest_posterior_time_h):
                    history["proposed_posterior_age_h"][step] = (
                        current_time_h - latest_posterior_time_h
                    )
                virtual_prior = np.asarray(target_learner.virtual_prior)
                positive = virtual_prior > 0.0
                if np.any(positive):
                    expected_mass = 1.0 / int(np.count_nonzero(positive))
                    history["proposed_virtual_prior_uniform_error"][step] = max(
                        float(np.max(np.abs(virtual_prior[positive] - expected_mass))),
                        float(np.sum(virtual_prior[~positive])),
                    )
            for name in algorithms:
                candidate = controls[name]
                applied = project_input(
                    candidate,
                    previous_controls[name],
                    config,
                    current_volume=inventory_volumes[name],
                )
                history["unfiltered_controls"][name][step] = candidate
                history["safety_interventions"][name][step] = (
                    abs(applied - candidate) > 1e-12
                )
                controls[name] = applied
                controller_objects[name].set_applied_input(applied)
                history["controls"][name][step] = applied
            released_since_control = {name: 0 for name in algorithms}
            goac_posterior_events_since_control = 0
            goac_parameter_updates_since_control = 0
            goac_target_updates_since_control = 0
            goac_virtual_updates_since_control = 0
            goac_learning_time_since_control = 0.0
        if "GOAC" in algorithms:
            goac_event_controls[event_step] = controls["GOAC"]
        for name in algorithms:
            plants[name], formed_increment = integrate_interval(
                plants[name],
                controls[name],
                config.measurement_interval_h,
                config.measurement_substeps,
                float(event_profile["mu_max"][event_step]),
                float(event_profile["K_sp"][event_step]),
                true_parameters,
            )
            inventory_volumes[name] += controls[name] * config.measurement_interval_h
            formed_product_mass[name] += formed_increment
    for name in algorithms:
        history["states"][name][-1] = plants[name]
    for name in feedback_methods:
        history["controller_states"][name][-1] = direct_control_state(
            latest_observations[name], inventory_volumes[name]
        )
    metrics = compute_metrics(config, scenario, seed, history, formed_product_mass)
    return SimulationResult(
        scenario=scenario, seed=seed, history=history, metrics=metrics
    )


def compute_metrics(
    config: SimulationConfig,
    scenario: ScenarioConfig,
    seed: int,
    history: Mapping[str, object],
    formed_product_mass: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    initial = np.asarray(scenario.x0)
    initial_product_mass = initial[1] * initial[3]
    initial_substrate_mass = initial[2] * initial[3]
    truth = np.asarray(history["true_K_IP"][:-1])
    for algorithm in config.algorithms:
        states = np.asarray(history["states"][algorithm])
        u = np.asarray(history["controls"][algorithm])
        final = states[-1]
        product_mass = final[1] * final[3]
        feed_volume = float(np.sum(u) * config.control_interval_h)
        substrate_consumed = (
            initial_substrate_mass
            + feed_volume * BASE_PARAMETERS["s_F"]
            - final[2] * final[3]
        )
        yield_value = (
            (product_mass - initial_product_mass) / substrate_consumed
            if substrate_consumed > 0
            else math.nan
        )
        computation = np.asarray(history["controller_time_s"][algorithm])
        constraint_violations = int(
            np.count_nonzero(
                (u < config.u_min_lph - 1e-09) | (u > config.u_max_lph + 1e-09)
            )
            + np.count_nonzero(
                np.abs(np.diff(np.r_[0.0, u])) > config.du_max_lph + 1e-09
            )
            + np.count_nonzero(states[:, 3] > config.volume_max_l + 1e-08)
        )
        max_volume_excess = float(
            np.max(np.maximum(states[:, 3] - config.volume_max_l, 0.0))
        )
        solver_success = np.asarray(history["solver_success"][algorithm], dtype=bool)
        solver_status = np.asarray(history["solver_status"][algorithm], dtype=int)
        solver_iterations = np.asarray(
            history["solver_iterations"][algorithm], dtype=int
        )
        solver_messages = np.asarray(
            history["solver_messages"][algorithm], dtype=object
        )
        failed_status_counts = Counter(solver_status[~solver_success].tolist())
        failed_message_counts = Counter(
            (str(message) for message in solver_messages[~solver_success])
        )
        row = {
            "scenario": scenario.name,
            "seed": seed,
            "algorithm": algorithm,
            "state_estimator": "None",
            "state_source": "Direct released measurement with sample-and-hold",
            "final_product_concentration_gpl": final[1],
            "final_product_mass_g": product_mass,
            "productivity_g_per_h": (product_mass - initial_product_mass)
            / config.total_time_h,
            "substrate_yield_g_per_g": yield_value,
            "feed_volume_l": feed_volume,
            "input_total_variation_lph": float(np.sum(np.abs(np.diff(np.r_[0.0, u])))),
            "constraint_violations": constraint_violations,
            "max_volume_excess_l": max_volume_excess,
            "feasible_run": int(constraint_violations == 0),
            "safety_interventions": int(
                np.count_nonzero(history["safety_interventions"][algorithm])
            ),
            "controller_time_total_s": float(np.sum(computation)),
            "controller_time_mean_ms": float(np.mean(computation) * 1000.0),
            "controller_time_p95_ms": float(np.percentile(computation, 95) * 1000.0),
            "solver_success_fraction": float(np.mean(solver_success)),
            "solver_failure_count": int(np.count_nonzero(~solver_success)),
            "solver_iterations_mean": float(np.mean(solver_iterations)),
            "solver_iterations_p95": float(np.percentile(solver_iterations, 95)),
            "solver_failure_status_counts": "; ".join(
                (
                    f"{status}:{count}"
                    for status, count in sorted(failed_status_counts.items())
                )
            ),
            "solver_failure_message_counts": " | ".join(
                (
                    f"{message}:{count}"
                    for message, count in sorted(failed_message_counts.items())
                )
            ),
            "product_mass_balance_error_g": float(
                product_mass - initial_product_mass - formed_product_mass[algorithm]
            ),
            "kip_rmse_gpl": math.nan,
            "kip_mae_gpl": math.nan,
            "kip_bias_gpl": math.nan,
            "kip_90pct_coverage": math.nan,
            "kip_90pct_mean_width_gpl": math.nan,
            "adaptation_lag_h": math.nan,
            "adaptation_lag_up_h": math.nan,
            "adaptation_lag_down_h": math.nan,
            "adaptation_transitions_converged": math.nan,
            "adaptation_transition_count": math.nan,
            "adaptive_mixing_events": math.nan,
            "measurement_event_count": int(
                history.get("measurement_event_counts", {}).get(algorithm, 0)
            ),
            "posterior_event_count": math.nan,
            "parameter_state_update_count": math.nan,
            "target_product_update_count": math.nan,
            "posterior_age_mean_h": math.nan,
            "posterior_age_max_h": math.nan,
            "virtual_prior_uniform_error_max": math.nan,
            "virtual_refinement_usage_fraction": math.nan,
            "parameter_information_gain_mean": math.nan,
            "target_information_gain_mean": math.nan,
            "state_c_rmse_gpl": math.nan,
            "state_p_rmse_gpl": math.nan,
            "state_s_rmse_gpl": math.nan,
            "state_joint_normalized_rmse": math.nan,
            "output_target_covariance_mean": math.nan,
            "output_target_covariance_abs_mean": math.nan,
            "output_target_covariance_abs_max": math.nan,
            "cross_to_marginal_variance_ratio_mean": math.nan,
            "paired_minus_diagonal_tracking_cost_mean": math.nan,
            "expected_kip_variance_mean": math.nan,
            "kip_variance_reduction_mean": math.nan,
            "expected_target_variance_mean": math.nan,
            "target_variance_reduction_mean": math.nan,
            "virtual_reward_variance_mean": math.nan,
            "support_input_gain_range_mean": math.nan,
            "uncertainty_cost_mean": math.nan,
            "uncertainty_action_shift_abs_mean": math.nan,
            "virtual_refinement_action_shift_abs_mean": math.nan,
            "paired_diagonal_action_difference_abs_max": math.nan,
        }
        if algorithm in history["controller_states"]:
            controller_states = np.asarray(history["controller_states"][algorithm])[
                :-1, :3
            ]
            state_error = controller_states - states[:-1, :3]
            component_rmse = np.sqrt(np.mean(np.square(state_error), axis=0))
            normalization = np.maximum(np.ptp(states[:-1, :3], axis=0), 1.0)
            row.update(
                {
                    "state_c_rmse_gpl": float(component_rmse[0]),
                    "state_p_rmse_gpl": float(component_rmse[1]),
                    "state_s_rmse_gpl": float(component_rmse[2]),
                    "state_joint_normalized_rmse": float(
                        np.sqrt(np.mean(np.square(state_error / normalization)))
                    ),
                }
            )
        if algorithm in ("GOAC",):
            estimate = np.asarray(history["kip_mean"][algorithm])
            lower = np.asarray(history["kip_lower"][algorithm])
            upper = np.asarray(history["kip_upper"][algorithm])
            error = estimate - truth
            row.update(
                {
                    "kip_rmse_gpl": float(np.sqrt(np.mean(np.square(error)))),
                    "kip_mae_gpl": float(np.mean(np.abs(error))),
                    "kip_bias_gpl": float(np.mean(error)),
                    "kip_90pct_coverage": float(
                        np.mean((truth >= lower) & (truth <= upper))
                    ),
                    "kip_90pct_mean_width_gpl": float(np.mean(upper - lower)),
                }
            )
            row.update(_adaptation_lags(estimate, truth, config, scenario))
            if algorithm == "GOAC":
                row["adaptive_mixing_events"] = int(
                    np.count_nonzero(
                        np.asarray(history["proposed_adaptive_mixing"]) > 0.0
                    )
                )
                row["posterior_event_count"] = int(
                    np.sum(history["proposed_posterior_events"])
                )
                row["parameter_state_update_count"] = int(
                    np.sum(history["proposed_parameter_update_count"])
                )
                row["target_product_update_count"] = int(
                    np.sum(history["proposed_target_update_count"])
                )
                posterior_age = np.asarray(history["proposed_posterior_age_h"])
                row["posterior_age_mean_h"] = float(np.nanmean(posterior_age))
                row["posterior_age_max_h"] = float(np.nanmax(posterior_age))
                uniform_error = np.asarray(
                    history["proposed_virtual_prior_uniform_error"]
                )
                row["virtual_prior_uniform_error_max"] = float(np.nanmax(uniform_error))
                row["virtual_refinement_usage_fraction"] = float(
                    np.mean(history["proposed_virtual_refinement_used"])
                )
                row["parameter_information_gain_mean"] = float(
                    np.mean(history["proposed_parameter_information_gain"])
                )
                row["target_information_gain_mean"] = float(
                    np.mean(history["proposed_target_information_gain"])
                )
                covariance_values = np.asarray(history["proposed_target_covariance"])
                ratio_values = np.asarray(history["proposed_covariance_ratio"])
                paired_cost = np.asarray(history["proposed_paired_tracking_cost"])
                diagonal_cost = np.asarray(history["proposed_diagonal_tracking_cost"])
                expected_kip_variance = np.asarray(
                    history["proposed_expected_kip_variance"]
                )
                prior_kip_variance = np.asarray(history["proposed_prior_kip_variance"])
                expected_target_variance = np.asarray(
                    history["proposed_target_variance"]
                )
                prior_target_variance = np.asarray(
                    history["proposed_prior_target_variance"]
                )
                virtual_reward_variance = np.asarray(
                    history["proposed_virtual_reward_variance"]
                )
                support_input_gain_range = np.asarray(
                    history["proposed_support_input_gain_range"]
                )
                uncertainty_cost = np.asarray(history["proposed_uncertainty_cost"])
                uncertainty_action_shift = np.asarray(
                    history["proposed_uncertainty_action_shift"]
                )
                virtual_refinement_action_shift = np.asarray(
                    history["proposed_virtual_refinement_action_shift"]
                )
                paired_diagonal_action_difference = np.asarray(
                    history["proposed_paired_diagonal_action_difference"]
                )
                row["output_target_covariance_mean"] = float(
                    np.nanmean(covariance_values)
                )
                row["output_target_covariance_abs_mean"] = float(
                    np.nanmean(np.abs(covariance_values))
                )
                row["output_target_covariance_abs_max"] = float(
                    np.nanmax(np.abs(covariance_values))
                )
                row["cross_to_marginal_variance_ratio_mean"] = float(
                    np.nanmean(ratio_values)
                )
                row["paired_minus_diagonal_tracking_cost_mean"] = float(
                    np.nanmean(paired_cost - diagonal_cost)
                )
                row["expected_kip_variance_mean"] = float(
                    np.nanmean(expected_kip_variance)
                )
                row["kip_variance_reduction_mean"] = float(
                    np.nanmean(prior_kip_variance - expected_kip_variance)
                )
                row["expected_target_variance_mean"] = float(
                    np.nanmean(expected_target_variance)
                )
                row["target_variance_reduction_mean"] = float(
                    np.nanmean(prior_target_variance - expected_target_variance)
                )
                row["virtual_reward_variance_mean"] = float(
                    np.nanmean(virtual_reward_variance)
                )
                row["support_input_gain_range_mean"] = float(
                    np.nanmean(support_input_gain_range)
                )
                row["uncertainty_cost_mean"] = float(np.nanmean(uncertainty_cost))
                row["uncertainty_action_shift_abs_mean"] = float(
                    np.nanmean(np.abs(uncertainty_action_shift))
                )
                row["virtual_refinement_action_shift_abs_mean"] = float(
                    np.nanmean(np.abs(virtual_refinement_action_shift))
                )
                row["paired_diagonal_action_difference_abs_max"] = float(
                    np.nanmax(np.abs(paired_diagonal_action_difference))
                )
        rows.append(row)
    return pd.DataFrame(rows)


def history_to_frame(result: SimulationResult) -> pd.DataFrame:
    history = result.history
    n_rows = len(history["time_h"])
    data: MutableMapping[str, Array] = {
        "time_h": np.asarray(history["time_h"]),
        "true_K_IP_gpl": np.asarray(history["true_K_IP"]),
        "true_mu_max_per_h": np.asarray(history["true_mu_max"]),
        "true_K_sp_gpl": np.asarray(history["true_K_sp"]),
    }
    for algorithm, states in history["states"].items():
        safe_name = algorithm.replace("-", "_").replace(" ", "_")
        states = np.asarray(states)
        data[f"c_{safe_name}_gpl"] = states[:, 0]
        data[f"p_{safe_name}_gpl"] = states[:, 1]
        data[f"s_{safe_name}_gpl"] = states[:, 2]
        data[f"V_{safe_name}_l"] = states[:, 3]
        data[f"P_{safe_name}_g"] = states[:, 1] * states[:, 3]
        for prefix, source in (
            ("u", history["controls"]),
            ("u_unfiltered", history["unfiltered_controls"]),
            ("safety_intervention", history["safety_interventions"]),
            ("s_target", history["targets"]),
            ("controller_time_s", history["controller_time_s"]),
            ("solver_success", history["solver_success"]),
            ("solver_status", history["solver_status"]),
            ("solver_iterations", history["solver_iterations"]),
        ):
            values = np.full(n_rows, np.nan)
            values[:-1] = np.asarray(source[algorithm])
            data[f"{prefix}_{safe_name}"] = values
        if algorithm in history["controller_states"]:
            controller_state = np.asarray(history["controller_states"][algorithm])
            data[f"c_measured_{safe_name}_gpl"] = controller_state[:, 0]
            data[f"p_measured_{safe_name}_gpl"] = controller_state[:, 1]
            data[f"s_measured_{safe_name}_gpl"] = controller_state[:, 2]
    for label in (
        "proposed_posterior_support_evaluations",
        "proposed_target_covariance",
        "proposed_output_variance",
        "proposed_target_variance",
        "proposed_cross_contribution",
        "proposed_covariance_ratio",
        "proposed_paired_tracking_cost",
        "proposed_diagonal_tracking_cost",
        "proposed_expected_kip_variance",
        "proposed_prior_kip_variance",
        "proposed_prior_target_variance",
        "proposed_virtual_reward_variance",
        "proposed_uncertainty_cost",
        "proposed_uncertainty_action_shift",
        "proposed_virtual_refinement_action_shift",
        "proposed_paired_diagonal_action_difference",
        "proposed_innovation_sigma",
        "proposed_adaptive_mixing",
        "proposed_parameter_state_update",
        "proposed_target_product_update",
        "proposed_virtual_refinement_used",
        "proposed_parameter_information_gain",
        "proposed_target_information_gain",
        "proposed_support_input_gain_range",
        "proposed_target_lower",
        "proposed_target_upper",
        "proposed_measurement_events",
        "proposed_posterior_events",
        "proposed_parameter_update_count",
        "proposed_target_update_count",
        "proposed_posterior_age_h",
        "proposed_virtual_prior_uniform_error",
    ):
        values = np.full(n_rows, np.nan)
        values[:-1] = np.asarray(history[label])
        data[label] = values
    for algorithm in ("GOAC",):
        safe_name = algorithm.replace("-", "_")
        for label, source in (
            ("K_IP_mean", history["kip_mean"]),
            ("K_IP_90pct_lower", history["kip_lower"]),
            ("K_IP_90pct_upper", history["kip_upper"]),
        ):
            values = np.full(n_rows, np.nan)
            values[:-1] = np.asarray(source[algorithm])
            data[f"{label}_{safe_name}"] = values
    return pd.DataFrame(data)


def aggregate_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in run_metrics.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    grouped = run_metrics.groupby(["scenario", "algorithm"], sort=False)[
        numeric_columns
    ]
    mean = grouped.mean().add_suffix("_mean")
    standard_deviation = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    return pd.concat([mean, standard_deviation], axis=1).reset_index()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GOAC with high-frequency measurements and no state observer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260817,
        help="First seed; subsequent repeats increment it by one.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scenarios", nargs="+", choices=DEFAULT_SCENARIOS)
    selection.add_argument("--all-scenarios", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory; existing directories are not overwritten.",
    )
    parser.add_argument("--save-all-trajectories", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.scenarios and len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios must not contain duplicate names")
    return args


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_arguments()
    if args.list_scenarios:
        for name, scenario in SCENARIOS.items():
            print(f"{name}: {scenario.description}")
        return 0
    config = replace(make_profile(args.profile, args.repeats), base_seed=args.seed)
    selected = (
        DEFAULT_SCENARIOS
        if args.all_scenarios
        else tuple(args.scenarios or ("piecewise_stress",))
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        args.output_dir or Path(__file__).resolve().parent / "results" / f"goac_{stamp}"
    )
    if output.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "simulation_config.json", config.to_dict())
    write_json(
        output / "scenario_config.json", {n: asdict(SCENARIOS[n]) for n in selected}
    )
    write_json(
        output / "environment.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "state_source": "Direct released measurements; volume from feed inventory",
            "timing_scope": "Parameter/target learning and control solve; excludes plant integration",
        },
    )
    records = []
    total = len(selected) * args.repeats
    print(f"Output directory: {output.resolve()}", flush=True)
    for scenario_name in selected:
        for repeat in range(args.repeats):
            seed = config.base_seed + repeat
            print(
                f"[{len(records) + 1}/{total}] {scenario_name}, seed={seed}", flush=True
            )
            result = run_simulation(config, SCENARIOS[scenario_name], seed)
            records.append(result.metrics)
            runs = pd.concat(records, ignore_index=True)
            runs.to_csv(output / "run_metrics.csv", index=False)
            if repeat == 0 or args.save_all_trajectories:
                history_to_frame(result).to_csv(
                    output / f"trajectory_{scenario_name}_seed_{seed}.csv", index=False
                )
            print(
                f"  Final product mass: {result.metrics.iloc[0]['final_product_mass_g']:.6f} g",
                flush=True,
            )
    summary = aggregate_metrics(runs)
    counts = (
        runs.groupby(["scenario", "algorithm"], sort=False).size().rename("run_count")
    )
    summary = summary.merge(
        counts.reset_index(), on=["scenario", "algorithm"], validate="one_to_one"
    )
    # Sample SD is undefined for one run; do not present it as zero variability.
    sd_columns = [c for c in summary if c.endswith("_std")]
    summary.loc[summary.run_count < 2, sd_columns] = np.nan
    summary.to_csv(output / "aggregate_metrics.csv", index=False)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "completed_runs": len(records),
            "expected_runs": total,
            "scenarios": list(selected),
            "repeats": args.repeats,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("All requested GOAC simulations completed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
