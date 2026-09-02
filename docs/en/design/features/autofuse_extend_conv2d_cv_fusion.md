# Autofuse Adaptation for ExtendConv2D: Key GE Repository Flows

This document describes the compilation/runtime pipeline changes made in the GE repository to support Autofuse automatic fusion for ExtendConv2D (Conv2DV2 was already supported).
The entry point is `Autofuser::Fuse` (`compiler/graph/optimize/autofuse/autofuse/autofuser.cpp`).

Scope: changes in the `ge` repository. Kernel codegen, `scaleGM`, and the tiling-key template dimension are in the `graph-autofusion` repository, which consumes the AscGraph / `Conv2DAttr` / `conv_subgraph` produced here.

For how to run LLTs, inspect coverage, and determine whether GE UTs depend on the autofuse repository, see [autofuse_extend_conv2d_llt.md](./autofuse_extend_conv2d_llt.md).

---

## 0. Objects Beginners Should Understand First

Autofuse does not directly concatenate GE nodes into a piece of C++. It passes through representations at several different levels:

| Object                         | How to understand it                                                                                         | Storage location                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| `ComputeGraph / Node / OpDesc` | Original GE compute graph; nodes are operators such as Conv, Add, and Relu, while `OpDesc` stores I/O descriptions and attributes | GE graph                                              |
| `SymbolicDescAttr`             | A "computable expression" version of tensor shape; a static dimension 64 is represented as `Symbol(64)`, and dynamic dimensions can also be represented by expressions | attrs group of `GeTensorDesc`                         |
| `LoopVar / LoopOp`             | Loop semantics translated from one GE operator, such as load, broadcast, reduce, and store                    | Temporary lowering objects                            |
| `KernelBox`                    | A group of LoopOps bounded by an output; a candidate kernel that can continue to merge with neighbors         | `LoweringResultAttrs` corresponding to an output `OutDataAnchor` |
| `AscGraph / ASCIR`             | A graph closer to AscendC codegen; nodes have become ASCIR types such as `Add`, `Reduce`, and `ExtendConv2DScale` | `AutoFuseAttrs::asc_graph`                            |
| `AscBackend`                   | A node in the GE graph representing that a group of original operators will generate one fused kernel         | Remains in the GE graph after replacing original nodes |
| `AutoFuseAttrs`                | The accompanying record of an AscBackend: internal AscGraph, original GE nodes, original inputs/outputs, fusion type, and so on | `OpDesc` of AscBackend                                |
| `conv_subgraph`                | A minimal copy of the original convolution used only for secondary ops-nn tiling; it is not executed again     | ExtAttr of the AscBackend `OpDesc`                    |
| `OpRunInfoV2`                  | Kernel launch parameters: tiling key/data, workspace, `block_dim`, and so on                                  | Output of the tiling stage                            |

The main process can be understood as:

```text
GE operator graph
  → Translate each operator into a LoopOp and attach it to the output to form a KernelBox
  → Merge adjacent KernelBoxes and realize them into an AscGraph
  → Replace this group of GE operators with one AscBackend node
  → can_fuse attempts to continue merging the AscBackend with more neighbors
  → Lifting checks whether the fusion is worth retaining; restore it if not, retain it if so
  → codegen generates a fused kernel from the AscGraph
  → tiling generates kernel launch parameters from the actual shape
```

This framework is not limited to convolution:

- **Pointwise**: Add, Mul, Relu, and similar operators are generally the easiest to fuse with one another.
- **Reduction**: ReduceSum/Max and similar operators require additional reduction-axis and scheduling information.
- **View**: Reshape/Transpose and similar operators may only change indexing and do not necessarily produce real computation.
- **Cube**: Conv/MatMul run on AIC and are often CV-fused with an AIV vector tail.
- **Fallback/Extern**: When no lowering is registered, the shape does not meet the conditions, or a feature switch is disabled, the operator remains external instead of being forcibly fused.

---

## 1. End-to-End Sequence

```text
[Compile time, before Autofuse]
  InferSymbolShape          Attach SymbolicDescAttr to tensors

[Autofuser::Fuse]
  PatternFusion             Graph-level patterns (weakly related to this change)
  Lowering                  GE operator → Loop IR / ASCIR → synthesize AscBackend
  can_fuse                  Continue vector fusion on lowered nodes
  Lifting                   Most fusions can be split back; cube operators such as convolution/MatMul are not,
                            and instead the original convolution is copied into an ExtAttr of the fused node (conv_subgraph)
  PostProcess               AscBackend post-processing

[Late compilation / runtime]
  AicoreRtParseAndTiling    Read conv_subgraph, run convolution tiling first, then fused-node tiling
```

Corresponding code sequence:

```text
autofuser.cpp:150  patter_fusion.RunAllPatternFusion(graph);
autofuser.cpp:157  lowerer.Lowering(graph);
                    ├─ asc_ir_lowerer.cpp:200 CompleteStaticSymbolicShapeForConv
                    ├─ asc_ir_lowerer.cpp:215 LoweringManager::LoweringGraph
                    └─ asc_ir_lowerer.cpp:216 FusedLoopToAscBackendOp
autofuser.cpp:170  FuseSubgraphsAndRootGraph(...)        // can_fuse
autofuser.cpp:177  lowerer.Lifting(graph);
                    └─ liftings.cpp:423 IsSkipLifting
                       └─ liftings.cpp:174 IsCubeSkipLifting
                          └─ liftings.cpp:61 CreateConvSubgraphAttr
autofuser.cpp:183  post_processor.Do(graph);
```

Runtime processing is not inside `Autofuser`; it is triggered by `AicoreRtParseAndTiling` (`base/common/op_tiling/op_tiling_rt2.cc`) when tiling is required.

> Line numbers are provided for quickly locating code in the current version. As the code evolves, search by function name first.

---

## 2. InferSymbolShape (Before Lowering)

**File**: `compiler/graph/optimize/symbolic/infer_symbolic_shape/infer/conv2d.cc`

**Why this change is needed**: Autofuse lowering does not obtain shapes through `GeTensorDesc::GetShape()`; it only uses `GetBufferShape()` → `SymbolicDescAttr`. If ExtendConv2D does not register inference, its output (and subsequent nodes that depend on it) may have no symbolic shape, causing lowering to fail immediately.

**Inputs**:

- An `ExtendConv2D` node in the GE graph
- Symbolic shapes of inputs x / filter
- Attributes: `strides` / `pads` / `dilations` / `groups` / `pad_mode`, and so on

**Differences from Conv2DV2**:

- Registration: `IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ExtendConv2D).InferSymbolShape(InferShape4Conv2D)`
- Attribute index of `pad_mode`: 6 for Conv2D/Conv2DV2; 7 for ExtendConv2D because `round_mode` precedes it
- groups and SAME/VALID padding use the same branches as Conv2DV2 (`is_conv2dv2 == true`)

**Output**: `SymbolicDescAttr` on the output tensor (`OriginSymbolShape` is a list of Expressions).

**In-code data flow**:

```text
conv2d.cc:552  context->GetInputSymbolShape(0/1)
      ↓
conv2d.cc:563  GetConv2DXShapeDim / GetConv2DWShapeDim
      ↓
conv2d.cc:565  CheckConv2DGroups
      ↓
conv2d.cc:567  GetConv2DStrides / Dilations / Pads
      ↓
conv2d.cc:573  Construct oh / ow expressions
      ↓
conv2d.cc:594  SetConv2DYShape
      ↓
SymbolicDescAttr of the output GeTensorDesc
```

This result is not used only by the convolution itself. When lowering subsequent Relu/Add/Reduce operators reads the shape of an input producer, it also reads this symbolic shape along the edge.

---

## 3. Lowering

Entry point: `AscIrLowerer::Lowering`. It contains three internal steps.

### 3.1 Complete Static Symbolic Shapes: `CompleteStaticSymbolicShapeForConv`

**File**: `lowering/asc_ir_lowerer.cpp`

Inference covers only operators that have an `InferSymbolShape` implementation. The **optional inputs** (bias / offset_w / scale0) of ExtendConv2D are often `Const` / `FileConstant`: their `GeShape` is already static, such as `[64]`, but they have no `SymbolicDescAttr`.

When `GetBufferShape()` in lowering finds `sym_attr == nullptr`, it returns failure and the entire CV fusion is discarded.

This is a **temporary workaround**: its logic is the same as the original whole-graph completion, but it processes only `Conv2D` / `Conv2DV2` / `ExtendConv2D` nodes and skips all other nodes.

Rules:

- Non-convolution node: skip
- `SymbolicDescAttr` already exists: skip
- Origin/current shape is unknown: skip (it must not be fabricated)
- Otherwise, write static dimensions as `Symbol(dim)` and attach them

**Example**:

```text
Data(x) [1,64,56,56]     ← Usually already has SymbolicDescAttr
Const(filter)            ← Usually already has it
Const(scale0) [64]       ← Has only GeShape, no SymbolicDescAttr
        │
        ▼
   ExtendConv2D → Relu
```

`StoreConv2D` calls `GetBufferShape` for every connected input. If the scale0 Const has no completed symbolic shape, lowering is skipped.

**Input**: Output descriptions of convolution nodes in the graph.
**Output**: For static outputs of convolution nodes that lack symbolic information, add `SymbolicDescAttr = [Symbol(d0), ...]`.

**Production and consumption locations**:

```text
Production:
  asc_ir_lowerer.cpp CompleteStaticSymbolicShapeForConv
    Conv2D / Conv2DV2 / ExtendConv2D only
      → GetOrCreateAttrsGroup<SymbolicDescAttr>
      → MutableOriginSymbolShape().MutableDims()

Consumption:
  loop_common.cpp:57-66 GetBufferShape
    Obtain SymbolicDescAttr from the output desc and return Expression dimensions

Common callers:
  lowering_impl.cpp GetConv2DDims
  loop_api.cpp StoreConv2D reads the output shape
```

### 3.2 Single-Operator Lowering: `InnerLowerConv2D`

**Files**:

- Registration: `lowering/op_lowering_impl/lowering_impl.cpp` (`REGISTER_LOWERING_WITH_EXISTED(ExtendConv2D, LowerExtendConv2D)`)
- Attribute structure: `Conv2DAttr` in `lowering/op_helper/cube.h`
- ASCIR construction: `StoreConv2D` in `lowering/asc_lowerer/asc_overrides.h`

`Conv2D` / `Conv2DV2` use `LowerConv2D` (`is_extend_conv2d=false`); `ExtendConv2D` uses `LowerExtendConv2D` (`true`), and both share `InnerLowerConv2D`.

**Input**: A GE `ExtendConv2D` node (x, filter, and possibly unconnected bias / offset_w / scale0).

**Intermediate result** `Conv2DAttr`:

| Field                                                                                 | Meaning                                                                                  |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| strides / pads / dilations / groups / data_format / pad_mode / enable_hf32 / offset_x | Shared with Conv2DV2                                                                     |
| round_mode / enable_relu0 / fixed_shift_value                                         | ExtendConv2D extensions (`round_mode` must be read from `op_desc`; the default `"rint"` alone is insufficient) |
| has_bias / has_offset_w / has_scale0                                                  | Whether the corresponding input anchor is connected to a producer; **not** a proto field |
| is_extend_conv2d                                                                      | Determines which ASCIR is subsequently constructed                                       |

**Output**: A `KernelBox` (Loop IR) on the node's output anchor, plus an ASCIR variant node.

Strictly speaking, these two results are not produced at the same time:

1. `InnerLowerConv2D` first collects inputs and attributes, then calls `loop::StoreConv2D`.
2. `loop_api.cpp:577-608` creates a `StoreConv2DOp` and writes it through `SetLoopKernel` into the `LoweringResultAttrs` corresponding to the output anchor, forming and realizing a `KernelBox`.
3. Only at `LoweringManager::BuildOpDescForKernelBox` (`lowerings.cpp:423`) is `kernel_box.Realize<AscOverrides>` called.
4. During Realize, execution enters `StoreConv2D` at `asc_overrides.h:411` and creates a concrete ASCIR variant according to `Conv2DAttr`.

Data chain:

```text
lowering_impl.cpp:1296 BuildConv2DAttr
lowering_impl.cpp:1326 SetConv2DBiasAndOffsetW
lowering_impl.cpp:1344 SetExtendConv2DFixpipeParams
lowering_impl.cpp:1355 CollectConv2DInputs
      │  Produces connected inputs + Conv2DAttr (including has_bias / has_scale0)
      ▼
lowering_impl.cpp:1417 loop::StoreConv2D
      ▼
loop_api.cpp:606 StoreConv2DOp
      ▼
loop_api.cpp:608 SetLoopKernel(...).Realize()
      │  Writes into LoweringResultAttrs of the OpDesc containing the output anchor
      ▼
lowerings.cpp:431 kernel_box.Realize<AscOverrides>
      ▼
asc_overrides.h:411 StoreConv2D
      │  SET_EXTEND_CONV2D_ATTRS writes attributes into ASCIR ir_attr
      ▼
ExtendConv2D / Bias / Scale / BiasScale ASCIR nodes
```

Variant mapping (counting only **connected** inputs):

| Number of connected inputs | Condition       | ASCIR type                  |
| -------------------------- | --------------- | --------------------------- |
| 2                          | —               | `ExtendConv2D`              |
| 3                          | has_bias        | `ExtendConv2DBias`          |
| 3                          | has_scale0      | `ExtendConv2DScale`         |
| 4                          | bias + scale0   | `ExtendConv2DBiasScale`     |

Conv2DV2 continues to map bias / offset_w combinations to `Conv2D` / `Conv2DBias` / `Conv2DOffset` / `Conv2DOffsetBias`.

Backend secondary tiling (`build_conv_args` in `graph-autofusion`) **no longer** accepts `nullptr_inputs_index`. ExtendConv2D has 10 fixed logical proto input slots:

```text
0 x, 1 filter, 2 bias(has_bias), 3 offset_w always empty,
4 scale0(has_scale0), 5 relu_weight0 / 6 clip_value0 / 7 scale1 / 8 relu_weight1 / 9 clip_value1 always empty
```

`has_bias` / `has_scale0` are passed to Python through ASCIR ir_attr; missing slots are filled with `None`. Conv2DV2 still packs connected inputs compactly.

Related handling:

- `IsStaticShape` (`loop_api.cpp`): `dst` / `inputs` are already the non-null edges of this lowering, so only the origin shapes of these edges themselves are checked (the output desc of `dst`, and the input desc at `input->GetIdx()`). It no longer calls `AreDescriptorsStatic` to scan every anchor of the convolution node, so unused `y1` and unconnected optional inputs do not cause a static graph to be classified as dynamic.
- Unused `y1`: `InnerLowerConv2D` calls `StoreIgnoredOutput` for an unconnected output (a non-Extern, unrealized placeholder). `GetNodeKernelBoxes` / `GetRealizedKernelBoxes` still scan all output anchors; dead outputs are not skipped in the common path. The placeholder does not become a second AscBackend because it has no compute and is not realized.

ASCIR type definitions are in `temporary_dependencies/ascir/ascir_ops.h` (generated from REG_ASC_IR in graph-autofusion and then synchronized here).

### 3.3 Synthesizing the Fused Node: `FusedLoopToAscBackendOp`

Adjacent KernelBoxes (for example, ExtendConv2D + a subsequent Relu/Add) are merged into **one GE** `AscBackend` **node**.

**Input**: Multiple KernelBoxes.
**Outputs**:

- `AscBackend` in the graph
- `AutoFuseAttrs`: original GE node list, internal AscGraph, `is_fuse_from_lowering`, and so on

At this point, an independent ExtendConv2D is no longer visible in the graph; only the fused node remains. The original convolution information is still present in the origin/AscGraph attributes.

Key code locations:

```text
lowerings.cpp:423 BuildOpDescForKernelBox
  lowerings.cpp:431  KernelBox → AscGraph
  lowerings.cpp:451  fuse_attrs->SetAscGraph(...)
  lowerings.cpp:453  SetOriginOutputBuffers(...)
  lowerings.cpp:454  SetOriginNodes(...)
  lowerings.cpp:458  is_fuse_from_lowering = true

lowerings.cpp:529 FusedLoopToAscBackendOp
  Traverse the root graph and subgraphs, replacing fusible KernelBoxes with AscBackend
```

`SetOriginNodes` is important: Lifting relies on it to determine which GE nodes originally formed this AscBackend. Data dumps, troubleshooting, and construction of `conv_subgraph` also depend on it.

---

## 4. can_fuse (After Lowering, Before Lifting)

**File**: `CreateNewNodeInputDescAttr` / `CreateNewNodeOutputDescAttr` in `can_fuse/backend/backend_utils.cpp`

On a lowered AscBackend, the strategy solver may continue to fuse neighboring vector operations. It **creates** a new fused node and copies the I/O descriptions.

The old logic copied only shape / dtype / format, losing extended attributes (especially `SymbolicDescAttr` and format details needed by later tiling). The new logic is:

```text
*dst_desc = *src_desc;          // Copy the complete description
*dst_SymbolicDescAttr = *src;  // Then explicitly copy the symbolic shape
```

**Input**: An existing fused node and its neighbors.
**Output**: A larger fused node whose I/O retains complete format and symbolic-shape information.

**Code locations and data chain**:

```text
asc_backend_fusion_decider.cpp:528
  CreateNewNodeInputDescAttr(new_node, node1, node2, input_maps...)
asc_backend_fusion_decider.cpp:530
  CreateNewNodeOutputDescAttr(new_node, node1, node2, output_maps...)
      ↓
backend_utils.cpp:1826 / 1867
  Find the source tensor desc according to the input/output map
      ↓
  *dst_desc = *src_desc
  Explicitly copy SymbolicDescAttr
      ↓
I/O desc of the new AscBackend
```

The input/output map answers: "For external input i of the new fused node, which input of node1 or node2 does it come from?" The same mechanism is used for non-convolution fusions such as Add/Reduce/Transpose. This change ensures that their complete tensor descriptions are not lost when a fused node is reconstructed.

---

## 5. Lifting: Why Cube Is Not "Split Back" but Gets a `conv_subgraph`

### 5.1 What Lifting Originally Does

Lowering combines many small operators into an `AscBackend`. Some combinations are poor (too few operators, or no further growth through can_fuse). Lifting **splits the fused node back** into the original GE operators to avoid a poor fused kernel.

The decision is made in `IsSkipLifting` (`lowering/liftings.cpp`):

1. Has `_disable_lifting`: skip splitting back
2. Fusion type is **cube** (MatMul / Conv): use `IsCubeSkipLifting` (see below)
3. Comes from can_fuse and is not a split type: skip splitting back
4. Otherwise: it may actually be split back

### 5.2 Why Convolution Cannot Be Split Back

The goal of CV fusion is to place the **convolution (AIC) and subsequent vector operations (AIV) in the same kernel**.

If Lifting splits `AscBackend(ExtendConv2D + Relu)` back into `ExtendConv2D` and `Relu`:

- The graph again contains two single operators
- At runtime, each runs its own kernel
- The compile-time fusion work is wasted

Therefore, for a cube fused node, `IsCubeSkipLifting` **returns true (skip splitting back)** and the fused node remains in the graph.

This introduces a second problem: runtime tiling cannot invoke only Autofuse's own vector tiling. The convolution's tiling key, L1/L0 partitioning, and workspace must use the **ops-nn Conv2DV2 / ExtendConv2D tiling**. The fused node itself is no longer of type ExtendConv2D, so the tiling entry point cannot find the original convolution.

Therefore, while avoiding the split-back, **a small convolution graph usable for secondary tiling** is attached to the fused node. This is what it means to "attach `conv_subgraph`."

### 5.3 What `IsCubeSkipLifting` Does

**Inputs**: The current `AscBackend` node and its `AutoFuseAttrs`.

Steps:

1. Extract compute nodes from the origin nodes (excluding reshape and similar nodes).
2. If there are too few compute nodes and this fusion comes only from lowering without being enlarged by can_fuse, return false to allow splitting back (the fusion benefit is insufficient).
3. Inspect the cube node type in the internal AscGraph:
   - Conv family: `Conv2D` / `Conv2DBias` / `Conv2DOffset` / `Conv2DOffsetBias` / `ExtendConv2D` / `ExtendConv2DBias` / `ExtendConv2DScale` / `ExtendConv2DBiasScale`
   - Otherwise, treat it as MatMul
4. Convolution: `CreateConvSubgraphAttr`; MatMul: `CreateMMSubgraphAttr`
5. **return true**: the caller does not perform the actual lifting split-back.

`IsCubeNodeType` includes the four ExtendConv2D ASCIR types in the cube set (`autofuse_utils.h/.cpp`). Otherwise, ExtendConv2D would be misclassified here as MatMul and receive a `matmul_subgraph`.

### 5.4 `CreateConvSubgraphAttr`: What Exactly Is in the Subgraph

**Inputs**:

- Fused node `node` (AscBackend)
- Compute nodes in origin (including the original GE `ExtendConv2D` or `Conv2DV2`)

**Processing** (only origin nodes whose type is `Conv2DV2` or `ExtendConv2D`):

1. Use `CopyOpDesc` to obtain the convolution node for the subgraph.
2. Complete the IR attribute `ascendc_op_para_size` (if the name is absent, call `AppendIrAttrName`; if the value is absent, set it to `2MB`).
   Reason: ops-nn adds the private attribute `fixed_shift_value` after `ascendc_op_para_size`. If `ascendc_op_para_size` is absent after copying the subgraph, constructing the context by IR attribute index during secondary tiling goes out of bounds.
3. Traverse and connect edges using the **original input-anchor indices**:
   - `peer == nullptr`: this is an empty optional slot (bias / scale0 is not connected); **continue without compressing indices**.
     If connected inputs were renumbered continuously from 0, scale0 would be misplaced in the bias slot and all tiling arguments would be incorrect.
   - For filter (index=1), **two** descriptions must be changed because they are copies; changing one does not synchronize the other.
     - `conv_input_desc`: the filter of the convolution node in `conv_subgraph`, used by ops-nn secondary tiling, which expects `FRACTAL_Z`.
     - Input 1 of `node` (AscBackend): the fused node's own filter description, read later by Autofuse tiling / graph validation.
     Both use index 1 because cube inputs precede other inputs in AscBackend, so filter remains at 1.
4. Call `node->GetOpDesc()->SetExtAttr("conv_subgraph", sub_graph)`.

**Output**: ExtAttr `conv_subgraph` on the fused node, containing a small ComputeGraph:

```text
Copied Data/Const nodes (x, filter, and any actually connected bias/scale0)
        │  Edge dst indices match the original GE anchors (empty slots remain holes)
        ▼
Copied ExtendConv2D / Conv2DV2
  - Includes ascendc_op_para_size
  - filter format = FRACTAL_Z
```

After Lifting, the graph still contains one `AscBackend`, with this additional subgraph used specifically for runtime tiling. It does **not** execute the convolution computation a second time.

Production—storage—read locations of `conv_subgraph`:

```text
Production:
  liftings.cpp:61 CreateConvSubgraphAttr
  liftings.cpp:69 CopyOpDesc(ExtendConv2D / Conv2DV2)
  liftings.cpp:83-109 Build the graph using original input-anchor indices

Storage:
  liftings.cpp:116
  AscBackend::OpDesc.SetExtAttr("conv_subgraph", sub_graph)

Read:
  op_tiling_rt2.cc:919-925
  OpDesc.TryGetExtAttr("conv_subgraph")
    → Traverse the subgraph
    → Find ExtendConv2D / Conv2DV2

Consumption:
  op_tiling_rt2.cc:867 AutofuseNodeWithConvTiling
  Construct an Operator from the convolution node in the subgraph and invoke ops-nn tiling
```

---

## 6. Runtime Secondary Tiling

**File**: `base/common/op_tiling/op_tiling_rt2.cc`

After `AicoreRtParseAndTiling` finds that the current op is an Autofuse node:

1. First check for `matmul_subgraph` (not expanded here).
2. Then call `TryGetExtAttr("conv_subgraph")`.
3. Find a node of type `Conv2DV2` **or** `ExtendConv2D` in the subgraph and enter `AutofuseNodeWithConvTiling`.

`AutofuseNodeWithConvTiling` has two steps, in a fixed order:

| Step              | Called component                                                    | Result                                               |
| ----------------- | ------------------------------------------------------------------- | ---------------------------------------------------- |
| 1 Cube tiling     | ExtendConv2D/Conv2DV2 in the subgraph, through ops-nn `RtParseAndTiling` | tiling key / tiling data / workspace / `aic_num` |
| 2 Autofuse tiling | The fused node itself                                               | vector-side tiling / `aiv_num`                       |
| 3 Combination     | `block_dim = (aic_num*2 < aiv_num) ? ceil(aiv_num/2) : aic_num`    | Number of cores for the fused kernel                 |

**Input**: AscBackend + ExtAttr `conv_subgraph`.
**Output**: `OpRunInfoV2` (tiling key, tiling data, workspace, `block_dim`) for kernel launch.

If Lifting did not attach `conv_subgraph`, convolution tiling cannot be reached here and cube partitioning for the CV-fused kernel is incorrect.

**More complete data chain**:

```text
op_tiling_rt2.cc:895 AicoreRtParseAndTiling(AscBackend, platform_infos, run_info)
      ↓ IsAutofuseNode
op_tiling_rt2.cc:919 TryGetExtAttr(conv_subgraph)
      ↓
op_tiling_rt2.cc:924 Recognize ExtendConv2D / Conv2DV2
      ↓
op_tiling_rt2.cc:867 AutofuseNodeWithConvTiling
      ├─ :880 RtParseAndTiling(conv_op, ...)
      │       callback → HandleCubeTilingCallback
      │       Produces cube tiling data/workspace/aic_num
      ├─ :885 AutofuseNodeTiling(fused_op, ...)
      │       callback → HandleAutofuseTilingCallback
      │       Adds vector tiling/aiv_num
      └─ :887 Combine new_block_dim and write it back to run_info
```

For ordinary operators without `conv_subgraph` / `matmul_subgraph`, an Autofuse node goes directly through `AutofuseNodeTiling`. Only cube+vector fusion needs cube-specific tiling first and then combines it with the Autofuse-side result.

---

## 7. Data-Chain Summary: Where Data Is Produced, Stored, and Consumed

```text
GE graph: ExtendConv2D (+ Relu/Add ...)
        │ InferSymbolShape / CompleteStaticSymbolicShapeForConv (convolution nodes only)
        │ SymbolicDescAttr
        ▼
Lowering: Conv2DAttr + ASCIR ExtendConv2D{Bias,Scale,...} + KernelBox
        │ FusedLoopToAscBackendOp
        ▼
AscBackend (AutoFuseAttrs: origin + AscGraph)
        │ can_fuse may merge more vector operations (complete desc copy)
        ▼
AscBackend + ExtAttr conv_subgraph     ← Main product of the GE repository
        │ graph-autofusion codegen (another repository)
        ▼
Fused kernel source / binary
        │ AicoreRtParseAndTiling
        ▼
OpRunInfoV2 → launch
```

| Data                          | Production location                                      | Storage location                                      | Consumption location                                          |
| ----------------------------- | -------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------- |
| Convolution output symbolic shape | `infer/conv2d.cc:544 InferShape4Conv2D`               | Output `GeTensorDesc::SymbolicDescAttr`               | `loop_common.cpp:57 GetBufferShape`, and lowering of subsequent operators |
| Static-shape fallback         | `CompleteStaticSymbolicShapeForConv`                     | Convolution-node output desc                          | Convolution lowering that depends on `GetBufferShape`         |
| `Conv2DAttr`                  | `lowering_impl.cpp:1296-1417`                            | Initially `StoreConv2DOp::attrs_`                     | `asc_overrides.h:411 StoreConv2D`                             |
| `has_bias` / `has_scale0`     | `SetConv2DBiasAndOffsetW` / `SetExtendConv2DFixpipeParams` | `Conv2DAttr` → ASCIR ir_attr → Python cube_attributes | `build_conv_args` restores the 10 ExtendConv2D slots (2/4 by flags; 3/5–9 always empty) |
| `KernelBox`                   | `loop_api.cpp:577 StoreConv2D` / other Store APIs        | `LoweringResultAttrs` of the output OpDesc            | Fusion and `BuildOpDescForKernelBox` in `lowerings.cpp`       |
| `AscGraph`                    | `lowerings.cpp:431 kernel_box.Realize<AscOverrides>`     | `AutoFuseAttrs::asc_graph`                            | can_fuse, postprocess, graph-autofusion codegen               |
| origin nodes                  | `lowerings.cpp:454 SetOriginNodes`                       | `AutoFuseAttrs`                                       | Lifting, DFX, `CreateConvSubgraphAttr`                        |
| `AscBackend`                  | `lowerings.cpp:529 FusedLoopToAscBackendOp`              | GE ComputeGraph                                       | can_fuse, Lifting, codegen, runtime tiling                    |
| New fused-node tensor desc    | `backend_utils.cpp:1826/1867`                            | `OpDesc` of the new AscBackend                        | Subsequent fusion, codegen, tiling                            |
| `conv_subgraph`               | `liftings.cpp:61 CreateConvSubgraphAttr`                 | AscBackend OpDesc ExtAttr                             | `op_tiling_rt2.cc:919`                                       |
| Cube tiling result            | `op_tiling_rt2.cc:880 RtParseAndTiling` callback         | `OpRunInfoV2`                                         | Combined with Autofuse tiling                                 |
| Final tiling/launch parameters | `op_tiling_rt2.cc:867 AutofuseNodeWithConvTiling`       | `OpRunInfoV2`                                         | Runtime kernel launch                                         |

### 7.1 Walking Through an Example

Original graph:

```text
Data(x) ─┐
Const(w) ├─ ExtendConv2D ─ Relu ─ Add(residual) ─ NetOutput
Scale ───┘
```

1. `InferShape4Conv2D` computes the convolution output's symbolic shape; `CompleteStaticSymbolicShapeForConv` completes symbolic shapes only for static convolution-node outputs that may have been missed.
2. `LoweringGraph` translates ExtendConv2D, Relu, and Add into LoopOps separately; each output corresponds to a KernelBox. If an operator is unsupported or its shape does not meet the conditions, `FallbackLowering` is used without changing the semantics of the original graph.
3. After KernelBox fusion, an AscGraph is realized, possibly containing `ExtendConv2DScale → Relu → Add`; one AscBackend represents it in the GE graph.
4. can_fuse also evaluates whether more neighbors can be merged; complete tensor descriptions must be copied when a new node is created.
5. Lifting sees that it contains cube computation and that the fusion is worth retaining, so it does not split it back; it copies ExtendConv2D and its inputs from the origin nodes and attaches them as `conv_subgraph`.
6. graph-autofusion generates one CV-fused kernel from the AscGraph.
7. During tiling, `conv_subgraph` is first used to invoke ops-nn for AIC convolution tiling, then the AscBackend is used for AIV fused tiling, and both results are finally written into the same `OpRunInfoV2`.
8. Runtime launches the fused kernel using `OpRunInfoV2`; `conv_subgraph` itself is not run as a second execution graph.

---

## 8. Files Changed in GE

| File                                       | Role                                                                                         |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `infer/conv2d.cc`                          | ExtendConv2D symbolic shape; `pad_mode` index 7                                               |
| `asc_ir_lowerer.cpp`                       | `CompleteStaticSymbolicShapeForConv`: skip non-convolution nodes and complete only static SymbolicDescAttr for convolution outputs |
| `op_helper/cube.h`                         | Extended fields in `Conv2DAttr`                                                              |
| `lowering_impl.cpp`                        | `InnerLowerConv2D` / `LowerExtendConv2D`                                                     |
| `asc_overrides.h`                          | Construct ASCIR variants according to bias/scale0                                            |
| `loop_api.cpp`                             | `IsStaticShape` checks only the shape on the `dst` / connected-input edge                    |
| `lowerings.cpp`                            | KernelBox collection still scans all output anchors                                           |
| `autofuse_utils.h/.cpp`                    | Add the four ExtendConv2D variants to the cube-type set                                       |
| `liftings.cpp`                             | skip lifting recognizes ExtendConv2D; `CreateConvSubgraphAttr` connects by anchor, adds para_size, and sets filter to FRACTAL_Z |
| `backend_utils.cpp`                        | Copy the complete desc when creating a fused node                                             |
| `temporary_dependencies/ascir/ascir_ops.h` | Synchronize the ASCIR header                                                                  |
| `op_tiling_rt2.cc`                         | Recognize ExtendConv2D in conv_subgraph and perform secondary tiling                          |

On the `graph-autofusion` side (not expanded in this document): ASCIR REG; codegen uses fixed `x, filter, bias, offset_w, scale0` slots; `conv2d_v2` adds `scaleGM` and a new tiling-key dimension; secondary tiling uses `has_bias` / `has_scale0` to restore logical ExtendConv2D inputs and no longer passes `nullptr_inputs_index`.
