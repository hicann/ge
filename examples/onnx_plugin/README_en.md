# ONNX Python Plugin Graph Compilation and Execution Sample

## 1. Function Description

This sample demonstrates the complete path: export PyTorch custom operators to an
ONNX model, register Python parser plugins with GE through `onnx_plugin`, bring
the custom operators into the GE graph, compile the graph into an OM model with
ATC, and then load and execute the OM model through the ACL Python API with
result verification.

The sample contains two plugins, so one run covers all three callbacks:

| Custom Operator | Plugin | Callbacks Used | What It Does |
| --------------- | ------ | -------------- | ------------ |
| ThresholdedRelu | `thresholded_relu_plugin.py` | `parse_node` + `decompose` | GE has no existing counterpart, so it is decomposed into the existing Threshold and Mul (no kernel needed) |
| MyElu | `my_elu_plugin.py` | `parse_operator` | GE has the existing Elu operator, so only attributes are relayed |

For parameter parsing, `parse_node` is preferred; `parse_operator` is used only
in a few cases. See Chapter 4.

## 2. Directory Structure

```text
onnx_plugin/
├── plugin/
│   ├── thresholded_relu_plugin.py  // Plugin 1: parse_node + decompose
│   └── my_elu_plugin.py            // Plugin 2: parse_operator
├── export_onnx.py                  // Export PyTorch custom operators to ONNX
├── run_model.py                    // Load and execute the OM model with ACL
└── run.sh                          // One-shot script: export, compile, execute
```

The `plugin/` directory is intentionally separate from the other scripts: GE scans
one level of Python files under `ASCEND_CUSTOM_OPP_PATH`, while the exporter and
runner depend on PyTorch, NumPy, or ACL and must not be loaded as plugins during
ATC initialization. Note that if **any** Python file in the directory fails to
load (for example, due to a syntax error), the entire compilation is aborted; use
debug logging (see Section 3.6) to locate the offending file.

## 3. Usage

### 3.1 Preparing the CANN Package

- Refer to [Environment Preparation](../../docs/en/quick_install.md#1-environment-preparation), section "Method 3: Manual Package Installation > Scenario 1: Experience master version capabilities or develop based on master version", and install the `toolkit` and `ops` packages properly.
- Set environment variables (assuming the packages are installed in /usr/local/Ascend/):

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### 3.2 Preparing Python Dependencies

| Dependency | Usage | Version Requirement |
| ---------- | ----- | ------------------- |
| PyTorch | Export the ONNX model (`torch.onnx.export`) | 2.7 - 2.8, both versions verified |
| onnx | Used internally by the `torch.onnx.export` export process | 1.21.0 verified |
| NumPy | Input construction and result comparison | No special requirement |

The `acl` Python API is provided by the CANN toolkit. Installation command:

```bash
pip3 install torch numpy onnx
```

Verified end to end (export, ATC compilation, ACL execution, and result
comparison all passed) on:

- Python 3.12. Temporary requirement: the Python version used to build the run
  package must match the Python version used to run this sample.
- onnx 1.21.0

### 3.3 One-Shot Run

Run the following command in the `examples/onnx_plugin` directory:

```bash
SOC_VERSION=Ascend910B1 bash run.sh
```

`SOC_VERSION` is the SoC model of the target device. The default value is
`Ascend910B1`; change it to match the actual device, for example:

```bash
SOC_VERSION=Ascend910B2 bash run.sh
SOC_VERSION=Ascend910_9362 bash run.sh
```

`run.sh` creates the `output/` directory automatically and performs these steps:

1. Runs `export_onnx.py` to generate `output/thresholded_relu.onnx`, which
   contains the two custom operators ThresholdedRelu and MyElu;
2. Sets `ASCEND_CUSTOM_OPP_PATH` to the `plugin/` directory so that ATC
   discovers both plugins;
3. Runs ATC to compile the ONNX model into `output/thresholded_relu.om`
   (the two plugins handle the two custom operators respectively);
4. Runs `run_model.py` to load and execute the OM model with ACL and verify
   both outputs.

See Section 3.5 for the expected output.

### 3.4 Step-by-Step Run

To inspect export, compilation, or execution separately, run the commands below.
Note: the `output/` directory is not created automatically (the one-shot
`run.sh` creates it); create it manually first when running step by step:

```bash
# 0. Create the output directory (required for step-by-step run; run.sh creates it automatically)
mkdir -p output
# 1. Export the ONNX model
python3 export_onnx.py --output output/thresholded_relu.onnx
# 2. Set the plugin path so that ATC discovers the Python plugins in plugin/
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
# 3. Compile the ONNX model into an OM model
atc --model=output/thresholded_relu.onnx \
    --framework=5 \
    --output=output/thresholded_relu \
    --soc_version="${SOC_VERSION:-Ascend910B1}"
# 4. Load and execute the OM model
python3 run_model.py --model output/thresholded_relu.om
```

### 3.5 Result Verification

On success you will see:

```text
Input:
[[-1.   0.5  1.5]
 [ 2.  -2.   3. ]]
Output (ThresholdedRelu, decomposed into Threshold + Mul):
[[-0.   0.  1.5]
 [ 2.  -0.   3. ]]
Output (MyElu, mapped to Elu by parse_operator):
[[-0.6323242  0.5        1.5      ]
 [ 2.        -0.8647461  3.       ]]
[Success] GE graph compiled and executed; output matches PyTorch reference.
```

### 3.6 Verifying the Operator Results

GE supports graph dump. After compilation, the graph files are saved in the
`graph_dump/pid_*/` directory (the directory name varies by environment), where
`ge_onnx_*.pbtxt` are ONNX-format graph files that can be opened to inspect the
graph structure at each compilation stage:

```bash
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin:${ASCEND_CUSTOM_OPP_PATH:-}"
mkdir -p graph_dump
DUMP_GE_GRAPH=3 DUMP_GRAPH_PATH="$(pwd)/graph_dump" atc \
    --model=output/thresholded_relu.onnx \
    --framework=5 \
    --output=output/thresholded_relu \
    --soc_version="${SOC_VERSION:-Ascend910B1}"
```

The integration results of the two plugins can be confirmed in the pbtxt graph
files:

- ThresholdedRelu: the decomposed graph contains Threshold and Mul nodes, and
  the Threshold node carries the `/ThresholdedRelu` original-type marker, so
  the decomposition origin can be traced;
- MyElu: one-to-one mapping; the graph contains an Elu node with the `alpha`
  attribute relayed.

To inspect compilation logs, add `--log=info` (or a higher log level) to the
atc command above and set the screen-printing environment variable
`export ASCEND_SLOG_PRINT_TO_STDOUT=1` before running. See the
[atc --log parameter description (Chinese)](../../docs/zh/user_guides/atc_tools/CLI_options/--log.md)
for details. When a plugin callback raises a Python exception or a plugin file
fails to load, the default error output (E19999) already carries the
Python-side information (exception type, failing statement, plugin file, and
line number); use the log level above only when the full compilation log is
needed.

## 4. Plugin and Callback Writing

Integrating a custom operator involves two kinds of work:

1. Parameter parsing: move the attributes on the ONNX node onto the GE
   operator - bind either `parse_node` or `parse_operator`, not both;
2. Operator decomposition: when GE has no existing counterpart, replace the
   operator with a subgraph built from existing operators - use `decompose`.

| Callback | What It Does | Example |
| -------- | ------------ | ------- |
| `parse_node` | Read attributes by name (preferred) | `thresholded_relu_plugin.py` |
| `parse_operator` | Parse attributes as a whole (few cases, see 4.2) | `my_elu_plugin.py` |
| `decompose` | Replace the node with a subgraph of existing operators | `thresholded_relu_plugin.py` |

Only one plugin can be registered for a given `source`, and the `opsets` of
different plugins must not overlap (for example, `opsets=(1,)` together with
`opsets=(1, 2)` is rejected) - a conflict aborts compilation with an explicit
error at startup.

### 4.1 parse_node: Read Attributes by Name

`parse_node` is the preferred choice. The callback receives two objects:

- `node`: the ONNX node. Attributes are read by name with
  `node.attrs["attr_name"]`, and the values are plain Python values. The
  attribute type is decided by the suffix used at torch export time:

  | torch export | node.attrs gives |
  | ----------- | ---------------- |
  | `alpha_f=1.0` | `1.0` (float) |
  | `axis_i=1` | `1` (int) |
  | `name_s="x"` | `"x"` (str) |
  | `dims_i=[1,2]` | `[1,2]` (int list) |

- `target`: the target operator. Write attributes with
  `target.set_attr("attr_name", value)`.

Supported attribute types: int, float, string, and homogeneous lists of them.
If the node carries tensor or subgraph attributes, `node.attrs` raises an
error directly; use `parse_operator` instead (see 4.2).

Input and output registration: no registration is needed when the target
operator has a fixed number of inputs and outputs; when the number is
variable, each one must be registered with `register_input`,
`register_output`, `register_optional_input`, `register_dynamic_input`, or
`register_dynamic_output`.

First, what "wiring" is: an ONNX model is a dataflow graph. A node does not
hold data itself - it only names data: what its inputs are called and what its
outputs are called. For example, a model may contain two nodes:

```text
node 1 (Conv):  input=["image", "weights"]   output="conv_out"
node 2 (Add):   input=["conv_out", "bias"]   output="out"
```

The first input of node 2, "conv_out", is just a name; the node that actually
produces it is node 1. Wiring is the step GE performs after reading all nodes:
for each node input name, find the node that produces that data, and connect
the producer's output to the consumer's input. Only after everything is wired
can data flow between nodes.

The difference between the two input registration methods (outputs have no
optional variant):

- `register_input`: registers a required input. It states that every node
  using this operator has data wired to this position - ONNX models are
  exported under exactly this convention;
- `register_optional_input`: registers an optional input. It states that
  whether this position is wired is decided by each node in the model. When a
  node leaves it unwired, the input simply carries no data - the framework
  fills in no value, and what "unwired" means is defined by the target
  operator itself. For example, the `value` input of ThresholdV2 is optional;
  when unwired, that operator treats it as 0 - its own rule, not a framework
  default. It is perfectly legal for the operator on node A to wire the input
  while the same operator on node B does not; an unwired node only produces a
  warning during wiring, not an error.

Wiring does not match by name; it matches by position: every ONNX operator
type has its input list and order defined in the ONNX standard (for example,
Conv takes X, W, then B, with B optional), and each node fills in data names
in that order - the "i-th input" is the `i`-th entry of the node input list
("conv_out" is the 0th in the example above). GE connects the node's `i`-th
input to the `i`-th input of the target operator; on the target side, this
order is fixed in the operator definition for fixed-count operators, and is
the registration order otherwise (the first one registered is the 0th). Two
hard requirements follow:

- Fill every position: register every position, in the input order of the
  ONNX operator (required ones with `register_input`, optional ones with
  `register_optional_input`). Do not register only the positions you expect
  to be used - as soon as any node may wire data at position `k`, input `k`
  must be registered, or wiring fails with
  `E19999: Resolve operator IO name failed`;
- Keep the order right: the registration order must match the input order of
  the ONNX operator. Counter-example: the operator takes X, W, then optional
  B, but the plugin registers B first - the data of X then flows into the B
  input; compilation succeeds, and the mismatch only surfaces later during
  shape inference or execution.

Example: `plugin/thresholded_relu_plugin.py` - reads `alpha` onto the target
operator and registers its inputs and outputs.

### 4.2 parse_operator: Parse Attributes as a Whole

Use `parse_operator` in the following cases:

1. The node carries tensor or subgraph attributes - `parse_node` raises an
   error on them (see 4.1), while the JSON obtained by `parse_operator`
   contains everything;
2. You want to fetch all attributes of the node at once for wholesale relay,
   without knowing each attribute name in advance.

The callback receives two operators:

- `source`: the operator converted from the ONNX node (read-only). All
  attributes of the node are packed into one JSON string, stored in the
  `attribute` attribute of `source`; retrieve it with
  `source.get_attr("attribute")` (`source` also carries framework-filled
  information such as the node name and the operator type, but `attribute` is
  all you need for parsing node attributes). The structure of the JSON string:
  the outermost object has a single `attribute` key whose value is an array -
  one object per attribute on the node; each object describes one attribute
  with `name` (the attribute name), `type` (the type code), and the key that
  holds the value - which key holds the value depends on the attribute type.
  For example, if `alpha_f=1.5` was written at export time, `source` gives:

```json
{
  "attribute": [
    { "f": "1.5",            "name": "alpha",  "type": 1 },
    { "i": 3,                "name": "level",  "type": 2 },
    { "s": "x",              "name": "label",  "type": 3 },
    { "floats": [1.5, 0.5],  "name": "coeffs", "type": 6 },
    { "ints": [2, 3],        "name": "dims",   "type": 7 },
    { "strings": ["a", "b"], "name": "tags",   "type": 8 }
  ]
}
```

  The type codes are the ONNX-standard attribute type numbers: 1 = float,
  2 = int, 3 = string, 4 = tensor, 5 = subgraph, 6 = float list, 7 = int list,
  8 = string list. Scalar
  attributes (`f`/`i`/`s`) hold their values as strings, so convert them when
  using; list attributes (`floats`/`ints`/`strings`) come as plain JSON
  arrays; tensor (`t`) and subgraph (`g`) attributes appear as complete
  structure dictionaries - a tensor attribute means the attribute value itself
  is a tensor (for example, the `value` attribute of the Constant operator),
  and a subgraph attribute means the attribute value itself is a subgraph (for
  example, `then_branch`/`else_branch` of the control-flow operator If, or
  `body` of Loop); you normally do not need to parse their fields field by
  field.

- `target`: the target operator, likewise written with `set_attr`.

Example: `plugin/my_elu_plugin.py` - parses `alpha` out of the JSON and relays
it to Elu. This example has a single float attribute that `parse_node` could
handle as well; it is used here to demonstrate how to read the JSON.

For the complete description of every value field of `attribute` (including
composite types such as tensor and subgraph), see the
[parse_operator interface doc (Chinese)](../../docs/zh/api/graph_engine_api/python/ge/onnx_plugin/OnnxPlugin/parse_operator.md).

### 4.3 decompose: Replace the Node with a Subgraph of Existing Operators

Use `decompose` when GE has no existing counterpart. The callback receives
`source` (the operator produced by parameter parsing; the attributes it reads
are exactly the values written by `parse_node` or `parse_operator`) and
returns a subgraph built from existing operators; GE then replaces the
original node with this subgraph, so no device kernel is needed.

The subgraph is built with the GE ES graph-building API: `GraphBuilder` builds
the graph (creating inputs, setting outputs, building), and existing operators
(Threshold, Mul, SplitD, and so on) are directly callable functions. For the
full operator list and parameter descriptions, see the
[ES Python API document](../../docs/en/user_guides/es_graph/api/es_python.md).

Notes when writing `decompose`:

- the subgraph outputs must correspond to the outputs registered during
  parameter parsing;
- the subgraph input does not need an explicit dtype or shape; GE aligns them
  automatically with the input of the original node.

Example: `plugin/thresholded_relu_plugin.py` - after `alpha` is relayed, a
Threshold x Mul subgraph is built.
