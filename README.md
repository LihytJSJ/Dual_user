# GOAC high-observation benchmark

This repository contains the standalone fed-batch fermentation benchmark used
to evaluate GOAC with a higher observation frequency than the legacy baseline
controllers. It includes the simulator, five control stacks, eight scenarios,
repeated-seed experiment orchestration, plotting, and automated checks.

This is an intentionally asymmetric observation study. GOAC receives more
physical measurements than the baselines in scenarios S1-S7. Results from this
variant quantify the combined effect of the GOAC method and its higher-rate
information interface; they must not be described as a same-information
controller comparison. Scenario S8 overrides this policy and gives every
method the same sparse, delayed measurements.

## Control stacks

- **GOAC** uses a dynamic interval particle state filter, a bounded interval
  posterior for `K_IP`, an independent measured-product-increment target
  posterior, equal-weight virtual refinement inside the real credible target
  interval, and a posterior-support closed-form control action followed by the
  common safety projection.
- **A-MPC** uses a bootstrap particle state filter, a particle estimator for
  `K_IP`, and a 15-step prediction/3-step control NMPC solved by SLSQP.
- **Std-NMPC** uses a finite-difference EKF and fixed nominal kinetics with the
  same NMPC horizon and solver settings as A-MPC.
- **ESC** is a perturb-and-observe extremum-seeking reference controller.
- **Bang-Bang** is a substrate-deadband switching reference controller.

Every method has the same plant equations, actuator limits, vessel-capacity
projection, and standardized measurement-noise process for a given seed.
