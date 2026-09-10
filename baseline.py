"""PF-NMPC, ESC, and Bang-Bang controllers."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from goac import (
    Array,
    BASE_PARAMETERS,
    ScenarioConfig,
    SimulationResult,
    SimulationConfig as GOACConfig,
    Observation,
    MeasurementChannel,
    fermentation_rhs,
    integrate_interval,
    intrinsic_rates,
    product_mass_rate,
    normalized_gaussian_likelihood,
    project_input,
    admissible_input_bounds,
    capacity_safe_input_upper,
    braking_feed_volume_l,
    build_true_profiles,
    _initial_product_rate,
    _adaptation_lags,
)


@dataclass(frozen=True)
class SimulationConfig(GOACConfig):

    pf_particles: int = 120
    pf_process_std: float = 0.10
    pf_likelihood_std: float = 0.05
    pf_resample_ess_fraction: float = 0.50
    state_pf_particles: int = 240
    state_pf_resample_ess_fraction: float = 0.50
    mpc_prediction_horizon: int = 15
    mpc_control_horizon: int = 3
    mpc_q: float = 1.0
    mpc_r: float = 0.01
    mpc_r_delta: float = 20.0
    solver_method: str = "SLSQP"
    solver_ftol: float = 1e-7
    solver_max_iterations: int = 100
    esc_update_period_h: float = 10.0
    esc_initial_u_lph: float = 0.20
    esc_step_lph: float = 0.10
    bang_bang_half_width_gpl: float = 2.0
    algorithms: Tuple[str, ...] = ("PF-NMPC", "ESC", "Bang-Bang")


def make_profile(name: str, repeated_runs: int | None = None) -> SimulationConfig:
    if name == "paper":
        config = SimulationConfig()
    elif name == "quick":
        config = SimulationConfig(
            profile_name="quick",
            total_time_h=30.0,
            integration_step_h=0.05,
            measurement_interval_h=0.10,
            posterior_interval_h=0.20,
            control_interval_h=0.50,
            repeated_runs=2,
            n_kip_intervals=24,
            n_target_candidates=41,
            pf_particles=48,
            state_pf_particles=80,
            mpc_prediction_horizon=6,
            mpc_control_horizon=2,
            solver_max_iterations=50,
            esc_update_period_h=3.0,
        )
    else:
        raise ValueError(f"Unknown profile: {name}")
    return (
        replace(config, repeated_runs=repeated_runs)
        if repeated_runs is not None
        else config
    )


def weighted_quantile(
    values: Array, weights: Array, probabilities: Sequence[float]
) -> Array:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(np.asarray(probabilities), cumulative, sorted_values)


class ParticleParameterEstimator:

    def __init__(
        self,
        lower: float,
        upper: float,
        n_particles: int,
        process_std: float,
        likelihood_std: float,
        ess_fraction: float,
        rng: np.random.Generator,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.n_particles = n_particles
        self.process_std = process_std
        self.likelihood_std = likelihood_std
        self.ess_fraction = ess_fraction
        self.rng = rng
        self.particles = rng.uniform(lower, upper, n_particles)
        self.weights = np.full(n_particles, 1.0 / n_particles)

    def update(
        self,
        measured_state: Array,
        u_at_measurement: float,
        measured_dpdt: float,
        mu_max: float,
        k_sp: float,
        nominal_parameters: Mapping[str, float],
    ) -> None:
        self.particles += self.rng.normal(0.0, self.process_std, self.n_particles)
        self.particles = np.clip(self.particles, self.lower, self.upper)
        predictions = product_concentration_rate_candidates(
            measured_state,
            u_at_measurement,
            self.particles,
            mu_max,
            k_sp,
            nominal_parameters,
        )
        self.weights *= normalized_gaussian_likelihood(
            measured_dpdt - predictions, self.likelihood_std
        )
        total = self.weights.sum()
        self.weights = (
            self.weights / total
            if total > 1e-300
            else np.full(self.n_particles, 1.0 / self.n_particles)
        )
        effective_sample_size = 1.0 / np.sum(np.square(self.weights))
        if effective_sample_size < self.ess_fraction * self.n_particles:
            positions = (
                self.rng.random() + np.arange(self.n_particles)
            ) / self.n_particles
            indices = np.searchsorted(np.cumsum(self.weights), positions, side="right")
            self.particles = self.particles[np.minimum(indices, self.n_particles - 1)]
            self.weights.fill(1.0 / self.n_particles)

    @property
    def mean(self) -> float:
        return float(np.dot(self.particles, self.weights))

    def credible_interval(self, level: float) -> Tuple[float, float]:
        tail = 0.5 * (1.0 - level)
        low, high = weighted_quantile(self.particles, self.weights, [tail, 1.0 - tail])
        return float(low), float(high)


def product_concentration_rate_candidates(
    x: Array,
    u: float,
    kip_candidates: Array,
    mu_max: float,
    k_sp: float,
    parameters: Mapping[str, float],
) -> Array:
    c, p, s, volume = np.asarray(x, dtype=float)
    c = max(c, 0.0)
    p = max(p, 0.0)
    s = max(s, 0.0)
    volume = max(volume, 1e-9)
    growth_capacity = max(0.0, 1.0 - c / parameters["c_max"])
    mu = mu_max * growth_capacity / (1.0 + s / parameters["K_I"])
    r_c = mu * c
    saturation = s / (s + k_sp) if s > 0.0 else 0.0
    inhibition = 1.0 / (1.0 + s / np.asarray(kip_candidates))
    r_p = parameters["K_1"] * r_c + parameters["K_2"] * saturation * inhibition * c
    return r_p - u * p / volume


def optimal_substrate_targets(
    biomass_concentration: float,
    kip_candidates: Array,
    config: SimulationConfig,
    parameters: Mapping[str, float],
) -> Array:
    s_grid = np.linspace(
        config.substrate_target_min,
        config.substrate_target_max,
        config.n_target_candidates,
    )
    c = max(float(biomass_concentration), 1e-9)
    growth_capacity = max(0.0, 1.0 - c / parameters["c_max"])
    mu = (
        config.nominal_mu_max_per_h
        * growth_capacity
        / (1.0 + s_grid[:, None] / parameters["K_I"])
    )
    r_c = mu * c
    saturation = s_grid[:, None] / (s_grid[:, None] + config.nominal_k_sp_gpl)
    inhibition = 1.0 / (1.0 + s_grid[:, None] / np.asarray(kip_candidates)[None, :])
    r_p = parameters["K_1"] * r_c + parameters["K_2"] * saturation * inhibition * c
    best_indices = np.argmax(r_p, axis=0)
    return s_grid[best_indices]


class BootstrapStateParticleFilter:

    def __init__(
        self,
        x0: Array,
        config: SimulationConfig,
        measurement_variances: Array,
        rng: np.random.Generator,
        n_particles: int | None = None,
    ) -> None:
        self.config = config
        self.rng = rng
        self.n_particles = int(n_particles or config.state_pf_particles)
        self.measurement_variances = np.maximum(
            np.asarray(measurement_variances, dtype=float), 1e-12
        )
        self.measurement_std = np.sqrt(self.measurement_variances)
        self.process_std = np.asarray(config.state_process_std, dtype=float)
        initial_std = np.maximum(self.measurement_std, 2.0 * self.process_std)
        self.particles = np.tile(np.asarray(x0, dtype=float), (self.n_particles, 1))
        self.particles[:, :3] += rng.normal(0.0, initial_std, (self.n_particles, 3))
        self._clip_particles()
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.ess = float(self.n_particles)

    def _clip_particles(self) -> None:
        self.particles[:, :3] = np.maximum(self.particles[:, :3], 0.0)
        self.particles[:, 0] = np.minimum(
            self.particles[:, 0], 1.25 * BASE_PARAMETERS["c_max"]
        )
        self.particles[:, 2] = np.minimum(self.particles[:, 2], BASE_PARAMETERS["s_F"])
        self.particles[:, 3] = np.maximum(self.particles[:, 3], 1e-9)

    def _systematic_indices(self, weights: Array, count: int) -> Array:
        positions = (self.rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(weights)
        return np.minimum(
            np.searchsorted(cumulative, positions, side="right"), len(weights) - 1
        )

    def _measurement_update(self, measurement: Array) -> None:
        residual = self.particles[:, :3] - np.asarray(measurement, dtype=float)[:3]
        log_likelihood = -0.5 * np.sum(
            np.square(residual) / self.measurement_variances, axis=1
        )
        log_posterior = np.log(np.maximum(self.weights, 1e-300)) + log_likelihood
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        total = posterior.sum()
        self.weights = (
            posterior / total
            if total > 1e-300
            else np.full(self.n_particles, 1.0 / self.n_particles)
        )
        self.ess = float(1.0 / np.sum(np.square(self.weights)))

    def predict(
        self,
        u: float,
        kip_estimate: float,
        parameters: Mapping[str, float],
        duration_h: float | None = None,
    ) -> None:
        duration = (
            self.config.control_interval_h if duration_h is None else float(duration_h)
        )
        model_parameters = dict(parameters)
        model_parameters["K_IP"] = float(kip_estimate)
        propagated = np.empty_like(self.particles)
        for index, particle in enumerate(self.particles):
            propagated[index], _ = integrate_interval(
                particle,
                u,
                duration,
                1,
                self.config.nominal_mu_max_per_h,
                self.config.nominal_k_sp_gpl,
                model_parameters,
            )
        noise_scale = math.sqrt(
            max(duration, 0.0) / max(self.config.control_interval_h, 1e-12)
        )
        propagated[:, :3] += self.rng.normal(
            0.0, noise_scale * self.process_std, (self.n_particles, 3)
        )
        self.particles = propagated
        self._clip_particles()

    def update(self, measurement: Array) -> None:
        self._measurement_update(measurement)
        if self.ess < self.config.state_pf_resample_ess_fraction * self.n_particles:
            indices = self._systematic_indices(self.weights, self.n_particles)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.n_particles)
            self.ess = float(self.n_particles)

    @property
    def x(self) -> Array:
        return np.asarray(self.weights @ self.particles, dtype=float)

    @property
    def covariance(self) -> Array:
        centered = self.particles - self.x
        return (centered.T * self.weights) @ centered

    def set_known_volume(self, volume_l: float) -> None:
        self.particles[:, 3] = max(float(volume_l), 1e-9)


class NMPCController:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.last_u = 0.0
        self.last_success = True
        self.last_iterations = 0
        self.last_status = 0
        self.last_message = "not run"
        self.warm_start_expanded = np.zeros(config.mpc_prediction_horizon)

    def _expanded_controls(self, decision: Array) -> Array:
        indices = np.minimum(
            np.arange(self.config.mpc_prediction_horizon)
            * self.config.mpc_control_horizon
            // self.config.mpc_prediction_horizon,
            self.config.mpc_control_horizon - 1,
        )
        return np.asarray(decision)[indices]

    def _objective(
        self,
        decision: Array,
        x0: Array,
        kip: float,
        substrate_target: float,
        parameters: Mapping[str, float],
    ) -> float:
        model_parameters = dict(parameters)
        model_parameters["K_IP"] = kip
        x = np.asarray(x0, dtype=float).copy()
        tracking_cost = 0.0
        for u in self._expanded_controls(decision):
            x, _ = integrate_interval(
                x,
                float(u),
                self.config.control_interval_h,
                1,
                self.config.nominal_mu_max_per_h,
                self.config.nominal_k_sp_gpl,
                model_parameters,
            )
            tracking_cost += self.config.mpc_q * (x[2] - substrate_target) ** 2
        previous = np.r_[self.last_u, np.asarray(decision)[:-1]]
        control_cost = self.config.mpc_r * float(np.dot(decision, decision))
        move_cost = self.config.mpc_r_delta * float(
            np.sum(np.square(np.asarray(decision) - previous))
        )
        return float(tracking_cost + control_cost + move_cost)

    def _warm_start(self, expanded_indices: Array, current_volume: float) -> Array:
        shifted = np.r_[self.warm_start_expanded[1:], self.warm_start_expanded[-1]]
        initial = np.array(
            [
                float(np.mean(shifted[expanded_indices == index]))
                for index in range(self.config.mpc_control_horizon)
            ]
        )
        previous = self.last_u
        for index in range(len(initial)):
            initial[index] = np.clip(
                initial[index],
                max(self.config.u_min_lph, previous - self.config.du_max_lph),
                min(self.config.u_max_lph, previous + self.config.du_max_lph),
            )
            previous = initial[index]
        cumulative_feed = self.config.control_interval_h * np.cumsum(
            initial[expanded_indices]
        )
        remaining_volume = max(0.0, self.config.volume_max_l - float(current_volume))
        if np.any(cumulative_feed > remaining_volume + 1e-10):
            initial = np.maximum(
                self.config.u_min_lph,
                self.last_u - self.config.du_max_lph * np.arange(1, len(initial) + 1),
            )
        return initial

    def solve(
        self,
        x: Array,
        kip: float,
        substrate_target: float,
        parameters: Mapping[str, float],
    ) -> float:
        horizon = self.config.mpc_control_horizon
        difference_matrix = np.eye(horizon)
        for index in range(1, horizon):
            difference_matrix[index, index - 1] = -1.0
        lower_moves = np.full(horizon, -self.config.du_max_lph)
        upper_moves = np.full(horizon, self.config.du_max_lph)
        lower_moves[0] += self.last_u
        upper_moves[0] += self.last_u
        move_constraint = LinearConstraint(difference_matrix, lower_moves, upper_moves)
        expansion = np.zeros((self.config.mpc_prediction_horizon, horizon))
        expanded_indices = np.minimum(
            np.arange(self.config.mpc_prediction_horizon)
            * horizon
            // self.config.mpc_prediction_horizon,
            horizon - 1,
        )
        expansion[np.arange(self.config.mpc_prediction_horizon), expanded_indices] = 1.0
        cumulative_feed = self.config.control_interval_h * np.cumsum(expansion, axis=0)
        volume_constraint = LinearConstraint(
            cumulative_feed,
            np.full(self.config.mpc_prediction_horizon, -np.inf),
            np.full(
                self.config.mpc_prediction_horizon,
                self.config.volume_max_l - float(x[3]),
            ),
        )
        initial = self._warm_start(expanded_indices, float(x[3]))
        result = minimize(
            self._objective,
            initial,
            args=(x, kip, substrate_target, parameters),
            method=self.config.solver_method,
            bounds=Bounds(
                np.full(horizon, self.config.u_min_lph),
                np.full(horizon, self.config.u_max_lph),
            ),
            constraints=(move_constraint, volume_constraint),
            options={
                "ftol": self.config.solver_ftol,
                "maxiter": self.config.solver_max_iterations,
                "disp": False,
            },
        )
        self.last_success = bool(result.success and np.all(np.isfinite(result.x)))
        self.last_iterations = int(getattr(result, "nit", 0))
        self.last_status = int(getattr(result, "status", -1))
        self.last_message = str(getattr(result, "message", "not reported"))
        solution = (
            np.asarray(result.x, dtype=float)
            if np.all(np.isfinite(result.x))
            else initial
        )
        self.warm_start_expanded = self._expanded_controls(solution)
        candidate = float(result.x[0]) if self.last_success else self.last_u
        self.last_u = project_input(
            candidate, self.last_u, self.config, current_volume=float(x[3])
        )
        return self.last_u

    def set_applied_input(self, applied_u: float) -> None:
        self.last_u = float(applied_u)


class PerturbAndObserveESC:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.current_u = config.esc_initial_u_lph
        self.direction = 1.0
        self.last_product_mass: float | None = None
        self.last_observation_step: int | None = None
        self.last_rate: float | None = None
        self.next_update_step = 0
        self.period_steps = max(
            1, int(round(config.esc_update_period_h / config.control_interval_h))
        )

    def solve(
        self,
        step: int,
        measured_product_mass: float,
        measured_volume: float,
        new_measurement: bool,
    ) -> float:
        if not new_measurement or step < self.next_update_step:
            self.current_u = project_input(
                self.current_u,
                self.current_u,
                self.config,
                current_volume=measured_volume,
            )
            return self.current_u
        if (
            self.last_product_mass is not None
            and self.last_observation_step is not None
        ):
            elapsed = (
                step - self.last_observation_step
            ) * self.config.control_interval_h
            rate = (measured_product_mass - self.last_product_mass) / max(elapsed, 1e-9)
            if self.last_rate is not None and rate < self.last_rate:
                self.direction *= -1.0
            self.last_rate = rate
            proposed = self.current_u + self.direction * self.config.esc_step_lph
            self.current_u = project_input(
                proposed, self.current_u, self.config, current_volume=measured_volume
            )
        self.last_product_mass = measured_product_mass
        self.last_observation_step = step
        self.next_update_step = step + self.period_steps
        return self.current_u

    def set_applied_input(self, applied_u: float) -> None:
        self.current_u = float(applied_u)


class BangBangController:
    def __init__(self, target: float, config: SimulationConfig) -> None:
        self.target = target
        self.lower = target - config.bang_bang_half_width_gpl
        self.upper = target + config.bang_bang_half_width_gpl
        self.config = config
        self.last_u = 0.0

    def solve(self, substrate_measurement: float, measured_volume: float) -> float:
        raw_u = self.last_u
        if substrate_measurement < self.lower:
            raw_u = self.config.u_max_lph
        elif substrate_measurement > self.upper:
            raw_u = self.config.u_min_lph
        self.last_u = project_input(
            raw_u, self.last_u, self.config, current_volume=measured_volume
        )
        return self.last_u

    def set_applied_input(self, applied_u: float) -> None:
        self.last_u = float(applied_u)


def run_simulation(
    config: SimulationConfig, scenario: ScenarioConfig, seed: int
) -> SimulationResult:
    seed_sequence = np.random.SeedSequence(seed)
    profile_seed, noise_seed, pf_seed, interval_state_seed, state_pf_seed = (
        seed_sequence.spawn(5)
    )
    profile_rng = np.random.default_rng(profile_seed)
    noise_rng = np.random.default_rng(noise_seed)
    pf_rng = np.random.default_rng(pf_seed)
    state_pf_rng = np.random.default_rng(state_pf_seed)
    parameters = dict(BASE_PARAMETERS)
    event_profile = build_true_profiles(
        config, scenario, profile_rng, time_step_h=config.measurement_interval_h
    )
    n_events = config.n_measurement_steps
    n_steps = config.n_control_steps
    control_stride = config.control_measurement_steps
    if n_events != n_steps * control_stride:
        raise ValueError("Control and measurement clocks do not cover the same horizon")
    algorithms = tuple(config.algorithms)
    if not algorithms or not set(algorithms).issubset({"PF-NMPC", "ESC", "Bang-Bang"}):
        raise ValueError("Baseline algorithms must be PF-NMPC, ESC, or Bang-Bang")
    model_based = tuple((name for name in algorithms if name in ("PF-NMPC",)))
    plants: Dict[str, Array] = {
        name: np.asarray(scenario.x0, dtype=float).copy() for name in algorithms
    }
    controls: Dict[str, float] = {
        name: config.esc_initial_u_lph if name == "ESC" else 0.0 for name in algorithms
    }
    state_noise = noise_rng.normal(size=(n_events + 1, 3))
    dpdt_noise = noise_rng.normal(size=n_events + 1)
    measurement_variances = np.square(np.asarray(scenario.state_noise_std))
    observers: Dict[str, object] = {}
    if "PF-NMPC" in model_based:
        observers["PF-NMPC"] = BootstrapStateParticleFilter(
            plants["PF-NMPC"], config, measurement_variances, state_pf_rng
        )
    pf_likelihood_std = config.pf_likelihood_std * scenario.dpdt_noise_std / 0.05
    particle_estimator = ParticleParameterEstimator(
        config.kip_min,
        config.kip_max,
        config.pf_particles,
        config.pf_process_std,
        pf_likelihood_std,
        config.pf_resample_ess_fraction,
        pf_rng,
    )
    adaptive_mpc = NMPCController(config)
    nominal_target = float(
        optimal_substrate_targets(
            scenario.x0[0], np.array([parameters["K_IP"]]), config, parameters
        )[0]
    )
    esc = PerturbAndObserveESC(config)
    bang_bang = BangBangController(nominal_target, config)
    controller_objects = {"PF-NMPC": adaptive_mpc, "ESC": esc, "Bang-Bang": bang_bang}
    inventory_volumes = {name: float(scenario.x0[3]) for name in algorithms}
    acquisition_strides: Dict[str, int] = {}
    for name in algorithms:
        if scenario.measurement_control_stride is not None:
            stride = int(scenario.measurement_control_stride) * control_stride
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
    released_since_control = {name: 0 for name in algorithms}
    ampc_learning_time_since_control = 0.0
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
        "estimated_states": {
            name: np.full((n_steps + 1, 4), np.nan) for name in model_based
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
        "kip_mean": {"PF-NMPC": np.full(n_steps, np.nan)},
        "kip_lower": {"PF-NMPC": np.full(n_steps, np.nan)},
        "kip_upper": {"PF-NMPC": np.full(n_steps, np.nan)},
    }
    formed_product_mass = {name: 0.0 for name in algorithms}
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
                if name in observers:
                    observers[name].update(observation.state[:3])
                if name == "PF-NMPC":
                    update_start = time.perf_counter_ns()
                    particle_estimator.update(
                        observation.state,
                        observation.u,
                        observation.dpdt,
                        config.nominal_mu_max_per_h,
                        config.nominal_k_sp_gpl,
                        parameters,
                    )
                    ampc_learning_time_since_control += (
                        time.perf_counter_ns() - update_start
                    ) * 1e-09
            if name in observers:
                observers[name].set_known_volume(inventory_volumes[name])
        control_due = event_step % control_stride == 0
        if control_due:
            step = event_step // control_stride
            previous_controls = controls.copy()
            for name in algorithms:
                history["states"][name][step] = plants[name]
            for name in model_based:
                history["estimated_states"][name][step] = observers[name].x
            if "PF-NMPC" in algorithms:
                decision_start = time.perf_counter_ns()
                target = float(
                    optimal_substrate_targets(
                        observers["PF-NMPC"].x[0],
                        np.array([particle_estimator.mean]),
                        config,
                        parameters,
                    )[0]
                )
                controls["PF-NMPC"] = adaptive_mpc.solve(
                    observers["PF-NMPC"].x, particle_estimator.mean, target, parameters
                )
                history["controller_time_s"]["PF-NMPC"][step] = (
                    ampc_learning_time_since_control
                    + (time.perf_counter_ns() - decision_start) * 1e-09
                )
                history["targets"]["PF-NMPC"][step] = target
                history["solver_success"]["PF-NMPC"][step] = adaptive_mpc.last_success
                history["solver_status"]["PF-NMPC"][step] = adaptive_mpc.last_status
                history["solver_iterations"]["PF-NMPC"][
                    step
                ] = adaptive_mpc.last_iterations
                history["solver_messages"]["PF-NMPC"][step] = adaptive_mpc.last_message
                history["kip_mean"]["PF-NMPC"][step] = particle_estimator.mean
                low, high = particle_estimator.credible_interval(
                    config.confidence_level
                )
                history["kip_lower"]["PF-NMPC"][step] = low
                history["kip_upper"]["PF-NMPC"][step] = high
            if "ESC" in algorithms:
                decision_start = time.perf_counter_ns()
                observation = latest_observations["ESC"]
                measured_product_mass = observation.state[1] * observation.state[3]
                controls["ESC"] = esc.solve(
                    step,
                    measured_product_mass,
                    plants["ESC"][3],
                    released_since_control["ESC"] > 0,
                )
                history["controller_time_s"]["ESC"][step] = (
                    time.perf_counter_ns() - decision_start
                ) * 1e-09
            if "Bang-Bang" in algorithms:
                decision_start = time.perf_counter_ns()
                observation = latest_observations["Bang-Bang"]
                controls["Bang-Bang"] = bang_bang.solve(
                    observation.state[2], plants["Bang-Bang"][3]
                )
                history["controller_time_s"]["Bang-Bang"][step] = (
                    time.perf_counter_ns() - decision_start
                ) * 1e-09
                history["targets"]["Bang-Bang"][step] = bang_bang.target
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
            ampc_learning_time_since_control = 0.0
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
        if "PF-NMPC" in observers:
            observers["PF-NMPC"].predict(
                controls["PF-NMPC"],
                particle_estimator.mean,
                parameters,
                config.measurement_interval_h,
            )
        for name in model_based:
            observers[name].set_known_volume(inventory_volumes[name])
    for name in algorithms:
        history["states"][name][-1] = plants[name]
    for name in model_based:
        history["estimated_states"][name][-1] = observers[name].x
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
            "state_estimator": {
                "PF-NMPC": "Bootstrap Particle Filter",
                "ESC": "Direct Measurement",
                "Bang-Bang": "Direct Measurement",
            }[algorithm],
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
        if algorithm in history["estimated_states"]:
            estimates = np.asarray(history["estimated_states"][algorithm])[:-1, :3]
            state_error = estimates - states[:-1, :3]
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
        if algorithm in ("PF-NMPC",):
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
        if algorithm in history["estimated_states"]:
            estimated = np.asarray(history["estimated_states"][algorithm])
            data[f"c_est_{safe_name}_gpl"] = estimated[:, 0]
            data[f"p_est_{safe_name}_gpl"] = estimated[:, 1]
            data[f"s_est_{safe_name}_gpl"] = estimated[:, 2]
    for label in ():
        values = np.full(n_rows, np.nan)
        values[:-1] = np.asarray(history[label])
        data[label] = values
    for algorithm in ("PF-NMPC",):
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
