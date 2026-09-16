# Python Declarative Address Refresh Add Custom Operator Offline Sample

## Sample Overview

This offline sample demonstrates the minimal chain of registering a custom operator prototype with Python
decorators, running `infer_meta`, `compile`, and `declare_launch_args`, generating AIR/OM, and executing
the model on an NPU through ACL. `declare_launch_args` runs only during ATC compilation; offline execution
consumes the generated task description and never imports the Python module again.

Core path:

```text
Register prototype and infer_meta with Python decorators
  -> build_graph.py generates AIR
  -> ATC imports the Python module and invokes the compile callback (bisheng compiles the Ascend C kernel)
  -> DeclareLaunchArgs declares kernel/bin/block_dim and address slots
  -> ATC generates TaskDef.kernel().args and context.args_format
  -> GE dispatches the custom kernel per TaskDef during OM loading
  -> INPUT/OUTPUT address slots are refreshed before execution
```

Main implementation points:

- The operator type is `AnnotatedAddCustom`. The prototype, `infer_meta`, `compile`, and
  `declare_launch_args` are all supplied by the Python decorators in `src/ge/annotated_add_custom.py`.
- The offline model path does not depend on the Python `execute` callback; no Python environment is
  required at OM runtime.
- `declare_launch_args` uses `append_input`/`append_output` to annotate two input address slots and one
  output address slot; GE refreshes addresses according to the args layout.
- The kernel binary is produced by the Python `compile` callback invoking `bisheng` on
  `add_custom_kernel.cpp` and is written into the OM together with the address layout.

## Prerequisites

- The CANN environment is installed and configured, for example by running `source ${ASCEND_HOME_PATH}/set_env.sh`.
- The environment provides the required `ACL`, `GE`, and `Graph` headers and libraries.
- Refer to the [Installation Guide](../../../../../docs/en/quick_install.md) to install the toolkit and ops packages.
- ATC, CMake, Python 3/pip, and an available NPU are required.
- `bisheng` and `llvm-objcopy` (shipped with the CANN toolchain, invoked by the Python `compile` callback).

## Quick Run

Run the following command in the `examples/custom_op/annotated_args_refresh_add_custom/offline/python` directory:

```bash
bash run.sh
```

The script performs the following steps:

1. Builds the custom OPP registration library and the Python ES wheel, then installs it.
2. Runs `build_graph.py` to generate `build/annotated_add.air`.
3. Invokes `atc`, which imports the Python module, executes `infer_meta`, `compile`, and
   `declare_launch_args`, and generates `build/annotated_add.om`.
4. Runs `run_model.py` to load the OM through ACL and execute two rounds with independent datasets.

On success the terminal prints:

```text
ROUND_1_FIRST=3
ROUND_2_FIRST=9
NPU_TWO_ROUND_VALIDATION=PASS
```

The script accepts `SOC_VERSION` (default `Ascend910B1`) and `DEVICE_ID` (default `0`).

## Key Files

```text
annotated_args_refresh_add_custom
└── offline/python
    ├── CMakeLists.txt                  # Build the custom OPP library and ES wheel
    ├── README.md
    ├── README_en.md
    ├── run.sh                          # ES, AIR, ATC, and ACL validation entry point
    ├── proto
    │   └── add_custom.h                # C++ graph prototype consumed by gen_esb
    └── src
        ├── build_graph.py              # Build the Python graph and save AIR
        ├── run_model.py                # Two-round ACL execution on an NPU
        └── ge
            └── annotated_add_custom.py # Python prototype, infer_meta, and compile callback
```

- The C++ `REG_OP` in `proto/add_custom.h` is used only by `gen_esb` to generate the
  `ge.es.custom.AnnotatedAddCustom` graph-building interface.
- `src/ge/annotated_add_custom.py` supplies the runtime prototype, `infer_meta`, `compile`, and
  `declare_launch_args`.
- `src/build_graph.py` builds the `AnnotatedAddCustom` graph and exports AIR.
- `src/run_model.py` loads the OM and validates AnnotatedArgs address refresh over two rounds.

## Declarative Address Refresh Implementation

```text
ATC compile phase:
  infer_meta / compile / declare_launch_args
    ├─ compile: bisheng compiles add_custom_kernel.cpp and extracts .aicore_binary
    ├─ KernelArgs: InputAddr{0}, InputAddr{1}, OutputAddr{0}
    ├─ AnnotatedKernelLaunchInfo { kernel_name, kernel_bin, block_dim }
    └─ The binary and address layout are written into annotated_add.om

OM execution phase:
  ACL loads the OM
    ├─ GE dispatches the custom kernel per TaskDef
    └─ INPUT/OUTPUT address slots are refreshed before execution
    (The Python module is not imported at runtime)
```

Callback constraints:

- `append_input` and `append_output` use the flattened input/output index of the current node.
- `AnnotatedArgsContext`, tensors, workspace, and the argument builder are borrowed objects valid only
  during the callback and must not escape. The argument builder is consumed by `add_launch` and cannot
  be reused.

## Build Artifacts

- `build/annotated_add.air`
  The AIR intermediate graph generated by `build_graph.py`.
- `build/annotated_add.om`
  The offline model compiled by ATC. It embeds the kernel binary and the AnnotatedArgs address layout, so no Python environment is required at runtime.
- `build/opp/`
  The custom operator OPP registration library (`op_proto/custom/libcust_opapi.so` plus platform copies), exposed to GE/ATC through `ASCEND_CUSTOM_OPP_PATH`.
- `build/es_output/`
  ES graph-building deliverables: `lib64/libes_custom.so` and `whl/es_custom-1.0.0-py3-none-any.whl` (installed into `build/whl_package/`).

## Notes

- The ATC log should contain the Python module-load, `infer_meta`, compile, and address-declaration
  markers. These compile-time markers must not appear during OM execution; `run.sh` verifies this
  automatically.
- `run.sh` unsets `ASCEND_CUSTOM_OPP_PATH` after the ATC phase to prove that OM execution does not
  depend on the Python plugin path.
- The OM embeds the kernel binary and address layout and can be distributed for execution outside this
  sample directory.
