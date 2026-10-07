# GOAC framework

This repository contains the standalone fed-batch fermentation benchmark used
to evaluate GOAC. It includes the simulator, other control stacks, eight scenarios,
repeated-seed experiment orchestration.

## Control stacks

- **GOAC** uses a dynamic interval particle  filter, a bounded interval
  posterior for `K_IP`, an independent measured-product-increment target
  posterior, equal-weight virtual refinement inside the real credible target
  interval, and a posterior-support closed-form control action followed by the
  common safety projection.
