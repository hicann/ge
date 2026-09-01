# Python Offline Custom Operator

This sample uses Python decorators to register the `AnnotatedAddCustom` prototype, `infer_meta`, `compile`, and
`declare_launch_args`. It generates AIR/OM and runs the model on an NPU through ACL. The declaration callback runs
only during ATC compilation; offline execution consumes the generated task description. The Python callback compiles
and serializes the Ascend C kernel binary.

## Directory

```text
offline/python
├── CMakeLists.txt                 # Build the custom OPP library and ES wheel
├── run.sh                         # Kernel, ES, AIR, ATC, and ACL validation entry point
├── proto/                         # C++ graph prototype consumed by gen_esb
└── src/
    ├── build_graph.py             # Build the Python graph and save AIR
    ├── run_model.py               # Two-round ACL execution on an NPU
    └── ge/annotated_add_custom.py # Python prototype and infer_meta
```

## Requirements and usage

Install and configure CANN, ATC, CMake, Python 3/pip, numpy, and an available NPU.

The C++ `REG_OP` in `proto/add_custom.h` is used only by `gen_esb` to generate the
`ge.es.custom.AnnotatedAddCustom` graph-building interface. The runtime prototype, `infer_meta`, `compile`, and
`declare_launch_args` are supplied by the Python decorators in `src/ge/annotated_add_custom.py`.

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/annotated_args_refresh_add_custom/offline/python
bash run.sh
```

The script accepts `SOC_VERSION` (default `Ascend910B1`) and `DEVICE_ID` (default `0`).

## Execution flow

1. Build the custom OPP and ES wheel; `build_graph.py` generates AIR.
2. ATC imports the Python module and invokes `infer_meta`, `compile`, and `declare_launch_args` to write the Ascend C
   binary and address layout into `build/annotated_add.om`.
3. `run_model.py` loads the OM with ACL, creates independent datasets for two rounds, and validates address refresh.

`NPU_TWO_ROUND_VALIDATION=PASS` in the runtime log indicates that both rounds passed. The ATC log should contain
the Python module-load, `infer_meta`, compile, and address-declaration markers. These
compile-time markers must not appear during OM execution.

The Add kernel is written with Ascend C in `offline/cpp/ge/add_custom_kernel.cpp`. Use the CANN Ascend C
documentation when developing kernels and ACL RTC when delivering their binaries.

## Callback constraints

Use the flattened input/output index of the current node with `append_input` and `append_output`.
`AnnotatedArgsContext`, tensors, workspace, and the argument builder are borrowed objects valid only during the
callback and must not escape. The argument builder is consumed by `add_launch` and cannot be reused.
