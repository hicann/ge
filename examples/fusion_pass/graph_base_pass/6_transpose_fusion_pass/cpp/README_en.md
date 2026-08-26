# transpose_fusion_pass - GraphNodeSettedFormatPass Example

## Description

This example demonstrates how to implement **config-driven format modification and redundant Transpose elimination**
by inheriting `FusionBasePass`.

`GraphNodeSettedFormatPass` runs at the `kAfterOriginGraphOptimize` stage. It reads a configuration file
to batch-modify the input/output format and shape of specified operators in the graph, validates the changes
via `CheckOpSupported`, and then automatically detects and removes Transpose nodes that have become redundant
due to the format changes.

### Typical Scenario

On Ascend AI processors, some operators (e.g., Conv2D) prefer NHWC format. If the model contains
`Data(NCHW) -> Transpose(NCHW->NHWC) -> Conv2D(NHWC)`, the user can configure the Data node's output format
to NHWC. This makes the Transpose redundant (both sides now NHWC), and the pass automatically removes it
and connects Data directly to the downstream operator.

```
Before:                              After:

Data(NCHW)                           Data(NHWC)
    |                                    |
 Transpose                             Conv2D
 (perm=[0,2,3,1])                     (NHWC)
    |
 Conv2D(NHWC)
```

### Configuration File Format

The configuration file path is specified via the `ASCEND_CUSTOM_FORMATS_CFG` environment variable.
The format is plain text:

```ini
# [NodeName] starts a section; NodeName must match the node name in the graph
[conv1]
input.0=FORMAT_NCHW
output.0=FORMAT_NHWC

[matmul_custom]
input.0=FORMAT_NCHW
input.1=FORMAT_NCHW
output.0=FORMAT_NCHW
```

- `[NodeName]` starts a section; `NodeName` must match the node name in the graph
- `input.<index>=<format>` or `output.<index>=<format>` configures the format of a specific port
- Format values are limited to: `FORMAT_NCHW`, `FORMAT_NHWC`
- Blank lines and lines starting with `#` are ignored

### Pass Execution Flow

```
Run()
  │
  ├─ 1. ParseConfigFile()         Read config file path from env var, parse node_name → FormatConfig mapping
  ├─ 2. Backup original graph     Graph origin_graph = *graph (for full rollback)
  ├─ 3. Iterate all nodes in graph:
  │      └─ ApplyFormatAndCheck()  For nodes matching config:
  │           ├─ Step0: Validate original format of ports to be modified is within supported range (FORMAT_NCHW / FORMAT_NHWC)
  │           │           └─ Unsupported → log and return false (node fails)
  │           ├─ Step1: Backup current input/output format + shape
  │           ├─ Step2: Modify input/output format + shape (with shape dimension reordering)
  │           ├─ Step3: Propagate output format to directly connected NetOutput nodes
  │           ├─ Step4: CheckOpSupported validation (skipped for Data nodes)
  │           │           └─ On failure → node-level rollback (RollbackFormats + RollbackNetOutput)
   │           └─ Step5: Detect and remove redundant Transpose nodes
   │                       ├─ Collection: IsTransposeNode → IsTransposePermConst → HasNoControlEdge → IsTransposeRedundant
   │                       │    ├─ Non-Const perm or Const with input edges → log and skip (not added to removal list)
   │                       │    └─ Transpose has control input or output edges → log and skip (not added to removal list)
   │                       └─ On failure → return false (triggers full graph rollback)
  │      Successfully configured nodes are recorded into configured_nodes set
  └─ 4. If any node failed:
         └─ Full rollback *graph = origin_graph, return FAILED
  └─ 5. CheckFormatContinuity()    Only check format continuity for nodes in configured_nodes
         ├─ Log nodes not participating in check (not found in graph / not successfully configured) at entry
         └─ On failure → Full rollback *graph = origin_graph, return FAILED
```

### Rollback Mechanism

| Level | Trigger | Mechanism |
|-------|---------|-----------|
| **Node-level rollback** | CheckOpSupported failed / format modification failed | `RollbackFormats` + `RollbackNetOutput` restore the current node and associated NetOutput format and shape |
| **Full graph rollback** | Any node's `ApplyFormatAndCheck` returned false (including Transpose removal failure), or `CheckFormatContinuity` check failed | `*graph = origin_graph` restores all modifications |

### Known Limitations

- **Non-Const perm or Const with input edges not removed**: A Transpose node is only removed if its perm input (input.1) is a Const node with no input edges. If perm is provided by a non-Const node (e.g., computed at runtime), or the perm Const node itself has data/control input edges, removing the Transpose would leave the perm node orphaned and could cause semantic errors. Such Transpose nodes are filtered out during the collection phase and not removed.
- **Transpose with control edges not removed**: If a Transpose node has control input edges (`GetInControlNodes`) or control output edges (`GetOutControlNodes`), removing it would lose control dependencies. The collection phase checks via `HasNoControlEdge`; Transpose nodes with control edges are not added to the removal list.
- **Subgraph Transpose not removed**: Transpose nodes in subgraphs may have associations with nodes in the root graph. Removing subgraph Transpose nodes can cause unexpected errors. The pass does not currently support removing Transpose nodes in subgraphs.

## Directory Structure

```
├── src
│   └── graph_node_setted_format_pass.cpp     // GraphNodeSettedFormatPass implementation
├── CMakeLists.txt                             // build script
├── data
│   ├── custom_formats.cfg                     // example configuration file
│   ├── gen_onnx.py                            // ONNX model export script
│   ├── verify_format.sh                       // format verification script
│   ├── run_atc.sh                             // ATC conversion script (with DUMP)
│   ├── format_analysis.md                     // format analysis findings
│   └── format_analysis_en.md                  // format analysis findings (English)
└── gen_es_api
    └── CMakeLists.txt                          // ES API generation script
```

## Environment Requirements

- Compiler: GCC >= 7.3.x
- Python >= 3.9, onnx library
- Completed [environment preparation](../../../../../docs/en/build.md#1-environment-preparation).

## Implementation Steps

1. Define `GraphNodeSettedFormatPass` inheriting `FusionBasePass`.
2. Override the `Run` method with the following logic:
   - Parse the configuration file specified by the `ASCEND_CUSTOM_FORMATS_CFG` environment variable.
   - Back up the original graph, then iterate over all nodes and execute `ApplyFormatAndCheck` for nodes matching the configuration:
     - Validate that the original format of ports to be modified is within the supported range (`FORMAT_NCHW` / `FORMAT_NHWC`); if not, log and skip the node.
     - Modify input/output format and shape, synchronously reordering shape dimensions when modifying format (e.g., NCHW→NHWC: `[N,C,H,W]`→`[N,H,W,C]`).
     - If an output is directly connected to a NetOutput node, propagate the format change to the corresponding NetOutput input port.
     - Validate the modified format combination via `GeUtils::CheckNodeSupportOnAicore` (Data nodes skip validation).
     - On validation failure, roll back the current node and associated NetOutput changes; on Transpose removal failure, trigger full graph rollback.
     - After successful validation, detect and remove Transpose nodes that have become redundant due to the format change.
     - Record successfully configured nodes into the `configured_nodes` set.
   - After all nodes are processed, call `CheckFormatContinuity` to only check format continuity for nodes in `configured_nodes`; log nodes not participating in check (not found in graph / not successfully configured) at entry; rollback the entire graph if discontinuity is detected.
3. Register `GraphNodeSettedFormatPass` as a custom fusion pass at the `kAfterOriginGraphOptimize` stage.

## Compilation

1. Set up environment:

   ```bash
   source ${ASCEND_HOME_PATH}/set_env.sh
   ```

2. Build:

   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc) data_transpose_fusion_pass
   make install
   ```

3. If build directory is deleted:

   ```bash
   cp build/es_output/lib64/libes_all.so ${ASCEND_PATH}/opp/vendors/${PASS_SO_DIR}/custom_fusion_passes/
   ```

## Running

### Verify Format (without installing pass)

```bash
cd data
bash verify_format.sh Ascend910_9362
```

### ATC Offline Inference (with pass installed)

1. Set environment variables:

   ```bash
   export DUMP_GE_GRAPH=2
   export ASCEND_CUSTOM_FORMATS_CFG=/path/to/custom_formats.cfg
   ```

2. Generate model and run ATC:

   ```bash
   cd data
   python3 gen_onnx.py
   atc --model=./model.onnx --framework=5 --soc_version=xxx --output=./model
   ```

   Or use the script directly:

   ```bash
   bash run_atc.sh Ascend910_9362
   ```

3. Expected log output:

   ```
   GraphNodeSettedFormatPass is starting
   [GraphNodeSettedFormatPass] Parsed 4 op config(s) from /path/to/custom_formats.cfg
   [GraphNodeSettedFormatPass] Processing node[conv1] (type=Conv2D)
   [GraphNodeSettedFormatPass] Node[conv1]: format applied and CheckOpSupported passed
   [GraphNodeSettedFormatPass] Removed redundant Transpose[transpose_0]
   GraphNodeSettedFormatPass completed successfully
   ```

4. Compare dump graphs to verify Transpose removal and format changes.

## Cleanup

```bash
make clean_custom_pass
```
