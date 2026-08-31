# ONNX Python Plugin Graph Compilation and Execution Sample

This sample demonstrates the complete path: export a PyTorch custom operator to ONNX,
register a Python parser plugin with `onnx_plugin`, decompose `ThresholdedRelu` into
the existing `Threshold` and `Mul` operators, compile the graph to an OM model with
ATC, and execute it through the ACL Python API.

## Directory layout

```text
onnx_plugin/
├── plugin/
│   └── thresholded_relu_plugin.py  # GE ONNX Python plugin
├── export_onnx.py                  # PyTorch custom operator -> ONNX
├── run_model.py                    # Load and execute OM with ACL
└── run.sh                          # Export, compile, and execute
```

The `plugin/` directory is intentionally separate from the other scripts. GE scans
one level of Python files under `ASCEND_CUSTOM_OPP_PATH`; the exporter and runner
depend on PyTorch, NumPy, or ACL and must not be imported as plugins during ATC
initialization.

## Requirements

- CANN installed and its matching `set_env.sh` sourced;
- `atc` available in `PATH` and an Ascend device;
- PyTorch, ONNX, and NumPy compatible with the CANN Python environment.

## Run the sample

Source the CANN environment and set the SoC model for the target device:

```bash
source /path/to/cann/set_env.sh
SOC_VERSION=Ascend910B1 ./run.sh
```

The default `SOC_VERSION` is `Ascend910B1`; override it for the target device, for
example:

```bash
SOC_VERSION=Ascend910B2 ./run.sh
SOC_VERSION=Ascend910_9362 ./run.sh
```

Verified end to end (export, ATC compile, ACL execution, and result comparison) on:

```text
SoC: Ascend910_9362 (Atlas A3)   SOC_VERSION=Ascend910_9362
```

The script performs these steps:

1. `export_onnx.py` writes `output/thresholded_relu.onnx`;
2. `ASCEND_CUSTOM_OPP_PATH` is set so ATC discovers the plugin file;
3. ATC compiles the ONNX model to `output/thresholded_relu.om`;
4. `run_model.py` loads and executes the OM model with ACL and checks the output.

The input is:

```text
[[-1.0,  0.5, 1.5],
 [ 2.0, -2.0, 3.0]]
```

The expected output keeps elements greater than `alpha=1.0` and replaces the rest
with `0`:

```text
[[0.0, 0.0, 1.5],
 [2.0, 0.0, 3.0]]
```

## Run step by step

To inspect export, compilation, or execution separately:

```bash
python3 export_onnx.py --output output/thresholded_relu.onnx
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
atc --model=output/thresholded_relu.onnx \
    --framework=5 \
    --output=output/thresholded_relu \
    --soc_version="${SOC_VERSION:-Ascend910B1}"
python3 run_model.py --model output/thresholded_relu.om
```

The sample uses the `decompose` callback to build a graph from existing GE/ES
operators and does not provide a new device kernel. It therefore exercises Python
ONNX plugin registration, parsing, subgraph expansion, graph compilation, and graph
execution.

The `parse_node` and `decompose` callbacks cooperate in this sample:

- `parse_node` runs when the ONNX node is converted to the target operator.
  It relays the `alpha` attribute onto the target operator and registers the
  ports of the dynamic-IO target `PartitionedCall` (the parser needs port
  names to wire the graph).
- The `source` received by `decompose` is the operator produced by
  `parse_node`; the `alpha` it reads was relayed by that callback, so
  `decompose` depends on the attribute relay and port registration done in
  `parse_node`.
