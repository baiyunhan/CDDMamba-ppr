# CDDMamba

Model-architecture source for CDDMamba, including CDFM, DP-MPC, DGR-CSMamba, multi-scale feature fusion and the detail decoder.

## Source files

- `models/cddmamba.py`: network assembly and structural forward path.
- `models/cdfm.py`: channel-dependent feature restoration and Runge–Kutta integration.
- `models/dp_mpc.py`: dual-path multi-axis polynomial convolution.
- `models/dgr_csmamba.py`: difference-guided scanning and global–local fusion.
- `models/layers.py`: shared network layers.

This is a model-source-only release. Experiment settings are not supplied. Constructor arguments expose configurable quantities without supplying the study's experiment configuration. The encoder is supplied as a module by the caller. The released structural forward path excludes auxiliary objectives.

The source uses PyTorch and mamba-ssm. This repository does not provide a complete experiment reproduction package.
