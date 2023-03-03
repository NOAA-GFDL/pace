========
Overview
========

Execution
---------

A few environment variable change the behavior of the model:
- FV3_DACEMODE=Python[Build|BuildAndRun|Run] controls the full program optimizer behavior
  - Python: default, use stencil only, no full program optmization
  - Build: will build the program then exit. This _build no matter what_. (backend must be `dace:gpu` or `dace:cpu`)
  - BuildAndRun: same as above but after build the program will keep executing (backend must be `dace:gpu` or `dace:cpu`)
  - Run: load pre-compiled program and execute, fail if the .so is not present (_no hashs check!_) (backend must be `dace:gpu` or `dace:cpu`)
- PACE_FLOAT_PRECISION=64 control the floating point precision throughout the program.
