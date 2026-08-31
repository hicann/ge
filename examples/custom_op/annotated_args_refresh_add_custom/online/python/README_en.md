# Python Online Custom Operator Performance Comparison

This sample uses Python decorators to register the prototypes and `infer_meta` functions of `AnnotatedAddCustom` and `NoRefreshAddCustom`. Two online GE
graphs compare declarative address refresh with a path that does not declare
refreshable addresses. Both paths use the same Ascend C Add kernel. The script validates results and measures
execution without generating AIR/OM.

## Directory

```text
online/python
├── CMakeLists.txt                         # Build the custom OPP library and ES wheel
├── run.sh                                 # Build and run the online comparison
├── proto/                                 # C++ graph prototypes consumed by gen_esb
└── src/
    ├── run.py                             # Validate and benchmark the two online graphs
    └── ge/annotated_add_custom.py         # Python prototypes and infer_meta
```

## Requirements and usage

Install and configure CANN, CMake, Python 3/pip, and an available NPU.

The C++ `REG_OP` declarations in `proto/add_custom.h` are used only by `gen_esb` to generate the graph APIs.
The runtime prototypes and `infer_meta` functions of both operators are supplied by Python decorators. Python `register_op_impl`
supplies `compile`, `declare_launch_args`, and `execute` for both operators. They reuse the Ascend C binary generated
from `cpp/add_custom_kernel/add_custom.asc`.

This directory does not compile `online/cpp/ge/custom_op.cpp`; that file is provided only as the C++ comparison sample.
All operator execution in this Python sample is implemented by Python callbacks.

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/annotated_args_refresh_add_custom/online/python
bash run.sh
```

Use `DEVICE_ID` (default `0`) to select the NPU:

```bash
DEVICE_ID=1 bash run.sh
```

## Execution flow

1. The custom OPP and ES wheel are built to generate both Python graph APIs.
2. The Python `compile` callback compiles and retains the Ascend C Add kernel binary while GE compiles the graph.
3. The Python `declare_launch_args` callback declares the input/output address slots and kernel launch, allowing GE to refresh addresses during
   repeated execution.
4. Python `execute` allocates the schema output and launches the same Ascend C kernel through the ACL Python API as the
   no-declaration baseline.
5. `run.py` validates both graphs, performs five warm-up iterations, and measures 100 iterations.

The log reports total and average times for both operators plus `Annotated speedup`.
`NPU_EXECUTION=PASS` indicates that both paths passed validation and execution.

The Add kernel is written with Ascend C in `online/cpp/add_custom_kernel/add_custom.asc`. Use the CANN Ascend C
documentation when developing kernels for the target hardware.
