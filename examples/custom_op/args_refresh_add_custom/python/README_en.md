# ArgsUpdater Add Custom Python Implementation Sample

This directory is the minimal Python custom operator execution sample at `examples/custom_op/args_refresh_add_custom/python`. It verifies that GE can load a Python `EagerExecuteOp` custom operator and build and execute a graph through the Python ES API.

## Scope

- The custom operator prototype still uses the C++ `REG_OP` registration: `AddPythonCustomOp`
- The operator implementation uses Python `EagerExecuteOp.execute(ctx)`
- The `ge.es.custom.AddPythonCustomOp` Python ES API is generated through `gen_esb`, and the Python `GraphBuilder` and `Session.run_graph` are used for graph construction and execution
- The Ascend C kernel reuses `../cpp/add_custom_kernel/add_custom.asc`
- `run.sh` precompiles the Ascend C kernel into a host object through `bisheng` and extracts the AI Core device binary from the `.aicore_binary` section
- In `execute`, the output is allocated, the extracted device binary is loaded through the ACL Python runtime, the x/y/z address parameters are prepared using `kernel_args_*`, and `acl.rt.launch_kernel_with_config` is called
- The ArgsUpdater address refresh optimization, performance comparison, and accuracy verification are not performed

## Directory

```text
args_refresh_add_custom
├── cpp
│   ├── add_custom_kernel
│   │   └── add_custom.asc             # Ascend C kernel reused in this sample
└── python
    ├── CMakeLists.txt                 # Generates the es_custom wheel through gen_esb
    ├── run.sh
    ├── proto
    │   ├── add_custom.h               # REG_OP prototype definition for gen_esb to generate the Python ES API
    │   └── add_custom.cc              # Custom operator proto compilation entry
    └── src
        ├── run.py                     # Python graph construction and execution entry
        └── ge
            └── add_custom.py          # Python EagerExecuteOp implementation
```

## Prerequisites

- Follow the [Installation Guide](../../../../docs/en/quick_install.md) to install the `toolkit` and `ops` packages.
- Configure the environment variables. The following example assumes that the packages are installed in `/usr/local/Ascend/`:
```
source /usr/local/Ascend/cann/set_env.sh
```
- **The Python version used to compile the run package** matches the Python version used to run this sample. The current Python custom operator loading path does not support cross-Python-version compatibility
- The current Python environment can import `ge.custom_op` and `acl`

## Conda Environment Sample (Python 3.11)

If you do not have a matching environment on the local machine, you can create one as follows:

```bash
conda create -n ge-custom-op-py311 python=3.11 -y
conda activate ge-custom-op-py311
python -m pip install --upgrade pip
python -m pip install attrs decorator sympy numpy psutil scipy
```

After creating the environment, confirm the following:

- The Python version in this environment matches the Python version used to compile the run package
- Run `source ${ASCEND_HOME_PATH}/set_env.sh` to configure the CANN environment variables
- Follow the "Running" section in this document to run the sample

## Running

`run.sh` first compiles `build/add_custom.host.o` through `bisheng`, then extracts `build/add_custom.aicore.o` through `llvm-objcopy`, and then generates the `es_custom` wheel based on `proto/add_custom.h` so that the current Python process can import `ge.es.custom.AddPythonCustomOp`.
`ADD_CUSTOM_NPU_ARCH` corresponds to `xxxx` in the Bisheng `--npu-arch=dav-xxxx` option, and the default value is `2201`. You can query the [mapping between AI processor models and `__NPU_ARCH__`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/compiler/BishengCompiler/atlas_bisheng_10_0005.html#ZH-CN_TOPIC_0000002594810544__table547713114286) in the official documentation and specify the value at runtime, for example, `ADD_CUSTOM_NPU_ARCH=2202 bash run.sh`.

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd examples/custom_op/args_refresh_add_custom/python
bash run.sh
```

Upon success, you can see output similar to the following:

```text
[Sample] graph added, graph_id=0
[PythonCustomOp] loaded kernel binary=.../build/add_custom.aicore.o, kernel=add_custom
[PythonCustomOp] AddPythonCustomOp.execute called
[PythonCustomOp] x shape=[1024], dtype=0, addr=0x...
[PythonCustomOp] y shape=[1024], dtype=0, addr=0x...
[PythonCustomOp] z shape=[1024], dtype=0, addr=0x...
[PythonCustomOp] stream=0x...
[PythonCustomOp] kernel args handle=0x..., num_blocks=1
[PythonCustomOp] acl.rt.launch_kernel_with_config ret=0
[Sample] run_graph finished, outputs=1
[Sample] output shape=[1024], dtype=0, format=2
```

## Description

`src/run.py` uses `GraphBuilder` to create two `Data` inputs, builds the graph through the generated `ge.es.custom.AddPythonCustomOp`, explicitly sets the output shape, data type, and format, and then calls `Session.run_graph` for execution.
`run.sh` first compiles the Ascend C source code into `add_custom.host.o` through Bisheng, and then extracts `add_custom.aicore.o` through `llvm-objcopy --only-section=.aicore_binary`. `AddPythonCustomOp` uses Python to perform host-side scheduling: it reads the input and output addresses and loads the extracted `add_custom.aicore.o` through `acl.rt.binary_load_from_file`. Currently, `binary_load_from_file` does not support `ACL_RT_LOAD_BINARY_OPT_MAGIC`, so an empty list is passed for the loading options.
The kernel parameters are appended in the x/y/z order through `acl.rt.kernel_args_init`, `acl.rt.kernel_args_append`, and `acl.rt.kernel_args_finalize`, and the `add_custom` kernel is launched through `acl.rt.launch_kernel_with_config` of the ACL Python runtime. An empty list is passed as the `cfg` parameter of `launch_kernel_with_config` to use the default runtime configuration.
At the current stage, the C++ code only provides the `REG_OP` prototype declaration and the reused Ascend C kernel source code. The actual operator execution entry is in `execute(ctx)` of `src/ge/add_custom.py`.
