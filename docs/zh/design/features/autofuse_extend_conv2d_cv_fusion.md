# Autofuse 适配 ExtendConv2D：GE 仓关键流程

本文记录 GE 仓为 Autofuse 自动融合 ExtendConv2D（此前已支持 Conv2DV2）所改动的编译/运行链路。
入口在 `Autofuser::Fuse`（`compiler/graph/optimize/autofuse/autofuse/autofuser.cpp`）。

适用范围：`ge` 仓改动。kernel codegen、`scaleGM`、tiling key 模板维在 `graph-autofusion` 仓，消费的是本文产出的 AscGraph / `Conv2DAttr` / `conv_subgraph`。

LLT 怎么跑、覆盖率怎么看、GE UT 是否依赖 autofuse 仓：见 [autofuse_extend_conv2d_llt.md](./autofuse_extend_conv2d_llt.md)。

---

## 0. 初学者先理解的几个对象

Autofuse 并不是直接把 GE 节点拼成一段 C++。它会经过几种不同层次的表示：


| 对象                             | 可以怎样理解                                                                     | 存放位置                                         |
| ------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------- |
| `ComputeGraph / Node / OpDesc` | 原始 GE 计算图；节点是 Conv、Add、Relu 等算子，`OpDesc` 保存输入输出描述和属性                       | GE 图                                         |
| `SymbolicDescAttr`             | tensor shape 的“可计算表达式版本”；静态维 64 表示为 `Symbol(64)`，动态维也可用表达式描述               | `GeTensorDesc` 的 attrs group                 |
| `LoopVar / LoopOp`             | 单个 GE 算子被翻译后的循环语义，例如 load、broadcast、reduce、store                           | lowering 临时对象                                |
| `KernelBox`                    | 以某个输出为边界的一组 LoopOp，是“可以继续与邻居合并的候选 kernel”                                  | 输出 `OutDataAnchor` 对应的 `LoweringResultAttrs` |
| `AscGraph / ASCIR`             | 更接近 AscendC codegen 的图；节点已变成 `Add`、`Reduce`、`ExtendConv2DScale` 等 ASCIR 类型 | `AutoFuseAttrs::asc_graph`                   |
| `AscBackend`                   | GE 图上代表“这一组原始算子将生成一个融合 kernel”的节点                                          | 替换原始节点后留在 GE 图上                              |
| `AutoFuseAttrs`                | AscBackend 的随身档案：内部 AscGraph、原始 GE 节点、原始输入输出、融合类型等                         | AscBackend 的 `OpDesc`                        |
| `conv_subgraph`                | 原始卷积的最小副本，只给 ops-nn 二次 tiling 使用，不参与再次执行                                   | AscBackend 的 `OpDesc` ExtAttr                |
| `OpRunInfoV2`                  | kernel 启动参数：tiling key/data、workspace、block_dim 等                          | tiling 阶段输出                                  |


可以把主过程理解成：

```text
GE 算子图
  → 每个算子翻译成 LoopOp，并挂到输出上形成 KernelBox
  → 相邻 KernelBox 合并并 realize 成 AscGraph
  → 用一个 AscBackend 节点替换这组 GE 算子
  → can_fuse 尝试把 AscBackend 与更多邻居继续合并
  → Lifting 检查融合是否值得保留；不值得则恢复，值得则保留
  → codegen 根据 AscGraph 生成融合 kernel
  → tiling 根据实际 shape 生成 kernel 启动参数
```

这个框架并不只服务卷积：

- **Pointwise**：Add、Mul、Relu 等通常最容易互相融合。
- **Reduction**：ReduceSum/Max 等需要额外的归约轴和调度信息。
- **View**：Reshape/Transpose 等可能只改变索引，不一定产生真实计算。
- **Cube**：Conv/MatMul 在 AIC 上计算，并常与 AIV vector 尾部做 CV 融合。
- **Fallback/Extern**：没有 lowering 注册、shape 不满足条件或功能开关关闭时，保持为外部算子，不强行融合。

---



## 1. 端到端顺序

```text
[编译期，Autofuse 之前]
  InferSymbolShape          给 tensor 挂 SymbolicDescAttr

[Autofuser::Fuse]
  PatternFusion             图级 pattern（与本次改动弱相关）
  Lowering                  GE 算子 → Loop IR / ASCIR → 合成 AscBackend
  can_fuse                  在已 lowering 的节点上继续并 vector
  Lifting                   多数融合可拆回；卷积/MatMul 这类 cube 不拆，
                            而是把原始卷积拷进融合节点的 ExtAttr（conv_subgraph）
  PostProcess               AscBackend 后处理

[编译后期 / 运行时]
  AicoreRtParseAndTiling    读 conv_subgraph，先跑卷积 tiling，再跑融合节点 tiling
```

对应代码顺序：

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

运行时不在 `Autofuser` 里，由 `AicoreRtParseAndTiling`（`base/common/op_tiling/op_tiling_rt2.cc`）在需要 tiling 时触发。

> 行号用于快速定位当前版本；代码演进后优先按函数名搜索。

---



## 2. InferSymbolShape（Lowering 之前）

**文件**：`compiler/graph/optimize/symbolic/infer_symbolic_shape/infer/conv2d.cc`

**为什么要改**：Autofuse lowering 取 shape 不走 `GeTensorDesc::GetShape()`，只走 `GetBufferShape()` → `SymbolicDescAttr`。ExtendConv2D 若不注册 infer，输出（以及依赖它的后续节点）可能没有符号 shape，lowering 直接失败。

**输入**：

- GE 图上的 `ExtendConv2D` 节点
- 输入 x / filter 的符号 shape
- attr：`strides` / `pads` / `dilations` / `groups` / `pad_mode` 等

**相对 Conv2DV2 的差异**：

- 注册：`IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ExtendConv2D).InferSymbolShape(InferShape4Conv2D)`
- `pad_mode` 的 attr 下标：Conv2D/Conv2DV2 为 6；ExtendConv2D 因前面多了 `round_mode`，为 7
- groups、SAME/VALID pad 走与 Conv2DV2 相同的分支（`is_conv2dv2 == true`）

**输出**：输出 tensor 上的 `SymbolicDescAttr`（`OriginSymbolShape` 为 Expression 列表）。

**代码内的数据流**：

```text
conv2d.cc:552  context->GetInputSymbolShape(0/1)
      ↓
conv2d.cc:563  GetConv2DXShapeDim / GetConv2DWShapeDim
      ↓
conv2d.cc:565  CheckConv2DGroups
      ↓
conv2d.cc:567  GetConv2DStrides / Dilations / Pads
      ↓
conv2d.cc:573  构造 oh / ow 表达式
      ↓
conv2d.cc:594  SetConv2DYShape
      ↓
输出 GeTensorDesc 的 SymbolicDescAttr
```

这里的产物不仅给卷积自身使用。后续 Relu/Add/Reduce 的 lowering 读取其输入生产者 shape 时，也会沿边读到这份符号 shape。

---



## 3. Lowering

入口：`AscIrLowerer::Lowering`。内部三步。

### 3.1 补齐静态符号 shape：`CompleteStaticSymbolicShapeForConv`

**文件**：`lowering/asc_ir_lowerer.cpp`

Infer 只覆盖「有 InferSymbolShape 实现」的算子。ExtendConv2D 的 **optional 输入**（bias / offset_w / scale0）经常是 `Const` / `FileConstant`：`GeShape` 已经是静态的 `[64]`，但没有 `SymbolicDescAttr`。

lowering 里 `GetBufferShape()` 发现 `sym_attr == nullptr` 就返回失败，整段 CV 融合会被丢掉。

这是 **临时规避**：逻辑与原来的全图补齐相同，但只处理 `Conv2D` / `Conv2DV2` / `ExtendConv2D` 节点，其余节点直接跳过。

规则：

- 非卷积节点：跳过
- 已有 `SymbolicDescAttr`：跳过
- origin/当前 shape 为 unknown：跳过（不能瞎填）
- 否则把静态维写成 `Symbol(dim)` 挂上

**例子**：

```text
Data(x) [1,64,56,56]     ← 通常已有 SymbolicDescAttr
Const(filter)            ← 通常已有
Const(scale0) [64]       ← 只有 GeShape，没有 SymbolicDescAttr
        │
        ▼
   ExtendConv2D → Relu
```

`StoreConv2D` 会对每个已连接输入做 `GetBufferShape`。scale0 的 Const 若未补符号 shape，lowering skip。

**输入**：图中卷积节点的输出 desc。
**输出**：卷积节点上静态且缺符号信息的输出，补上 `SymbolicDescAttr = [Symbol(d0), ...]`。

**生产和消费位置**：

```text
生产：
  asc_ir_lowerer.cpp CompleteStaticSymbolicShapeForConv
    仅 Conv2D / Conv2DV2 / ExtendConv2D
      → GetOrCreateAttrsGroup<SymbolicDescAttr>
      → MutableOriginSymbolShape().MutableDims()

消费：
  loop_common.cpp:57-66 GetBufferShape
    从输出 desc 取 SymbolicDescAttr 并返回 Expression 维度

常见调用者：
  lowering_impl.cpp GetConv2DDims
  loop_api.cpp StoreConv2D 读取输出 shape
```

### 3.2 单算子 lowering：`InnerLowerConv2D`

**文件**：

- 注册：`lowering/op_lowering_impl/lowering_impl.cpp`（`REGISTER_LOWERING_WITH_EXISTED(ExtendConv2D, LowerExtendConv2D)`）
- 属性结构：`lowering/op_helper/cube.h` 的 `Conv2DAttr`
- 建 ASCIR：`lowering/asc_lowerer/asc_overrides.h` 的 `StoreConv2D`

`Conv2D` / `Conv2DV2` 走 `LowerConv2D`（`is_extend_conv2d=false`）；`ExtendConv2D` 走 `LowerExtendConv2D`（`true`），共用 `InnerLowerConv2D`。

**输入**：GE `ExtendConv2D` 节点（x、filter，以及可能未连接的 bias / offset_w / scale0）。

**中间产物** `Conv2DAttr`：


| 字段                                                                                    | 含义                                                           |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| strides / pads / dilations / groups / data_format / pad_mode / enable_hf32 / offset_x | 与 Conv2DV2 公共                                                |
| round_mode / enable_relu0 / fixed_shift_value                                         | ExtendConv2D 扩展（`round_mode` 必须从 op_desc 读取，不能只靠默认 `"rint"`） |
| has_bias / has_offset_w / has_scale0                                                  | 看对应输入锚点是否连上生产者，**不是** proto 字段                               |
| is_extend_conv2d                                                                      | 决定后面建哪种 ASCIR                                                |


**输出**：该节点输出锚点上的 `KernelBox`（Loop IR），以及 ASCIR 变体节点。

严格说，这两个产物不是同一时刻产生：

1. `InnerLowerConv2D` 先收集输入和属性，然后调用 `loop::StoreConv2D`。
2. `loop_api.cpp:577-608` 创建 `StoreConv2DOp`，通过 `SetLoopKernel` 把它写进输出锚点对应的 `LoweringResultAttrs`，形成并 realize `KernelBox`。
3. 到 `LoweringManager::BuildOpDescForKernelBox`（`lowerings.cpp:423`）时，才调用 `kernel_box.Realize<AscOverrides>`。
4. Realize 过程中进入 `asc_overrides.h:411 StoreConv2D`，根据 `Conv2DAttr` 创建具体 ASCIR 变体。

数据链：

```text
lowering_impl.cpp:1296 BuildConv2DAttr
lowering_impl.cpp:1326 SetConv2DBiasAndOffsetW
lowering_impl.cpp:1344 SetExtendConv2DFixpipeParams
lowering_impl.cpp:1355 CollectConv2DInputs
      │  产出已连接 inputs + Conv2DAttr（含 has_bias / has_scale0）
      ▼
lowering_impl.cpp:1417 loop::StoreConv2D
      ▼
loop_api.cpp:606 StoreConv2DOp
      ▼
loop_api.cpp:608 SetLoopKernel(...).Realize()
      │  写入输出锚点所在 OpDesc 的 LoweringResultAttrs
      ▼
lowerings.cpp:431 kernel_box.Realize<AscOverrides>
      ▼
asc_overrides.h:411 StoreConv2D
      │  SET_EXTEND_CONV2D_ATTRS 把属性写入 ASCIR ir_attr
      ▼
ExtendConv2D / Bias / Scale / BiasScale ASCIR 节点
```

变体映射（只统计**已连接**输入个数）：


| 已连接输入数 | 条件            | ASCIR 类型                |
| ------ | ------------- | ----------------------- |
| 2      | —             | `ExtendConv2D`          |
| 3      | has_bias      | `ExtendConv2DBias`      |
| 3      | has_scale0    | `ExtendConv2DScale`     |
| 4      | bias + scale0 | `ExtendConv2DBiasScale` |


Conv2DV2 仍按 bias / offset_w 组合映射到 `Conv2D` / `Conv2DBias` / `Conv2DOffset` / `Conv2DOffsetBias`。

后端二次 tiling（`graph-autofusion` `build_conv_args`）**不再**接收 `nullptr_inputs_index`。ExtendConv2D proto 逻辑输入固定 10 槽：

```text
0 x, 1 filter, 2 bias(has_bias), 3 offset_w 恒空,
4 scale0(has_scale0), 5 relu_weight0 / 6 clip_value0 / 7 scale1 / 8 relu_weight1 / 9 clip_value1 恒空
```

`has_bias` / `has_scale0` 已随 ASCIR ir_attr 传到 Python；缺的槽位插 `None`。Conv2DV2 仍按已连接输入紧凑排布。

配套：

- `IsStaticShape`（`loop_api.cpp`）：`dst` / `inputs` 已是本次 lowering 的非空边，只检查这些边自己的 origin shape（`dst` 的 output desc、`input->GetIdx()` 的 input desc）。不再调用 `AreDescriptorsStatic` 扫卷积节点全部锚点，因此 unused `y1` 和未连接 optional 不会把静态图判成动态。
- 未使用的 `y1`：`InnerLowerConv2D` 里对未连接输出调用 `StoreIgnoredOutput`（非 Extern、未 realize 的占位）。`GetNodeKernelBoxes` / `GetRealizedKernelBoxes` 仍扫全部输出锚点，不在公共路径上跳过死输出。占位不会变成第二个 AscBackend（没有 compute、未 realize）。

ASCIR 类型定义在 `temporary_dependencies/ascir/ascir_ops.h`（由 graph-autofusion 的 REG_ASC_IR 生成后同步过来）。

### 3.3 合成融合节点：`FusedLoopToAscBackendOp`

相邻 KernelBox（例如 ExtendConv2D + 后面的 Relu/Add）合并成 **一个 GE** `AscBackend` **节点**。

**输入**：多个 KernelBox。
**输出**：

- 图上的 `AscBackend`
- `AutoFuseAttrs`：origin GE 节点列表、内部 AscGraph、`is_fuse_from_lowering` 等

此时图上已经看不到独立的 ExtendConv2D，只剩融合节点；原始卷积信息还在 attrs 的 origin / AscGraph 里。

关键代码位置：

```text
lowerings.cpp:423 BuildOpDescForKernelBox
  lowerings.cpp:431  KernelBox → AscGraph
  lowerings.cpp:451  fuse_attrs->SetAscGraph(...)
  lowerings.cpp:453  SetOriginOutputBuffers(...)
  lowerings.cpp:454  SetOriginNodes(...)
  lowerings.cpp:458  is_fuse_from_lowering = true

lowerings.cpp:529 FusedLoopToAscBackendOp
  遍历根图和子图，将可融合 KernelBox 替换成 AscBackend
```

`SetOriginNodes` 很重要：Lifting 时需要靠它找到“这个 AscBackend 原来由哪些 GE 节点组成”；数据 dump、问题定位和构建 `conv_subgraph` 也依赖它。

---



## 4. can_fuse（Lowering 之后、Lifting 之前）

**文件**：`can_fuse/backend/backend_utils.cpp` 的 `CreateNewNodeInputDescAttr` / `CreateNewNodeOutputDescAttr`

在已 lowering 的 AscBackend 上，策略求解器可能继续把邻近 vector 融进来，会 **新建** 融合节点并拷贝 I/O desc。

旧逻辑只抄 shape / dtype / format，扩展属性（尤其 `SymbolicDescAttr`、后续 tiling 关心的 format 细节）会丢。新逻辑：

```text
*dst_desc = *src_desc;          // 整份拷贝
*dst_SymbolicDescAttr = *src;  // 再显式拷符号 shape
```

**输入**：已有融合节点及其邻居。
**输出**：更大的融合节点；I/O 上 format 与符号 shape 仍完整。

**代码位置和数据链**：

```text
asc_backend_fusion_decider.cpp:528
  CreateNewNodeInputDescAttr(new_node, node1, node2, input_maps...)
asc_backend_fusion_decider.cpp:530
  CreateNewNodeOutputDescAttr(new_node, node1, node2, output_maps...)
      ↓
backend_utils.cpp:1826 / 1867
  按 input/output map 找到源 tensor desc
      ↓
  *dst_desc = *src_desc
  显式复制 SymbolicDescAttr
      ↓
新 AscBackend 的 I/O desc
```

input/output map 用来回答“新融合节点的第 i 个外部输入，来自 node1 还是 node2 的哪个输入”。这套机制同样用于 Add/Reduce/Transpose 等非卷积融合；本次修改保证它们的完整 tensor 描述不会在融合重建节点时丢失。

---



## 5. Lifting：为什么 cube「不拆回」，而是挂 `conv_subgraph`



### 5.1 Lifting 本来在干什么

Lowering 把很多小算子合成 `AscBackend`。有些合成质量差（算子太少、或后续 can_fuse 没再长大），Lifting 会把融合节点 **拆回** 原来的 GE 算子，避免一个很差的融合 kernel。

判定在 `IsSkipLifting`（`lowering/liftings.cpp`）：

1. 带 `_disable_lifting`：跳过拆回
2. 融合类型是 **cube**（MatMul / Conv）：走 `IsCubeSkipLifting`（见下）
3. 来自 can_fuse 且不是 split 类：跳过拆回
4. 其它：可能真正拆回



### 5.2 卷积为什么不能拆回

CV 融合的目标就是：**卷积（AIC）和后面的 vector（AIV）打在同一个 kernel 里**。

如果 Lifting 把 `AscBackend(ExtendConv2D + Relu)` 拆回 `ExtendConv2D` 和 `Relu`：

- 图上又变成两个单算子
- 运行时各跑各的 kernel
- 编译期白融合

所以对 cube 融合节点，`IsCubeSkipLifting` **返回 true（跳过拆回）**，融合节点留在图上。

但这会带来第二个问题：运行时 tiling 不能只调 Autofuse 自己的 vector tiling。卷积的 tiling key、L1/L0 切分、workspace 必须走 **ops-nn 的 Conv2DV2 / ExtendConv2D tiling**。融合节点本身已经不是 ExtendConv2D 类型，tiling 入口找不到原始卷积。

因此在「不拆回」的同时，把 **一份可供二次 tiling 使用的卷积小图** 挂到融合节点上。这就是「挂 `conv_subgraph`」。

### 5.3 `IsCubeSkipLifting` 在做什么

**输入**：当前 `AscBackend` 节点、其 `AutoFuseAttrs`。

步骤：

1. 从 origin nodes 里抽出 compute 节点（reshape 等不计入）。
2. 若 compute 太少且这次融合只来自 lowering、没被 can_fuse 做大：返回 false，允许拆回（融合收益不够）。
3. 看内部 AscGraph 里的 cube 节点类型：
  - Conv 族：`Conv2D` / `Conv2DBias` / `Conv2DOffset` / `Conv2DOffsetBias` / `ExtendConv2D` / `ExtendConv2DBias` / `ExtendConv2DScale` / `ExtendConv2DBiasScale`
  - 否则视为 MatMul
4. 卷积：`CreateConvSubgraphAttr`；MatMul：`CreateMMSubgraphAttr`
5. **return true**：调用方不再做真正的 lifting 拆回。

`IsCubeNodeType` 已把四个 ExtendConv2D ASCIR 类型加进 cube 集合（`autofuse_utils.h/.cpp`），否则这里会把 ExtendConv2D 误判成 MatMul，去挂 `matmul_subgraph`。

### 5.4 `CreateConvSubgraphAttr`：子图里到底有什么

**输入**：

- 融合节点 `node`（AscBackend）
- origin 里的 compute 节点（含原始 GE `ExtendConv2D` 或 `Conv2DV2`）

**处理**（只处理 type 为 `Conv2DV2` 或 `ExtendConv2D` 的 origin 节点）：

1. `CopyOpDesc` 得到子图卷积节点。
2. 补 IR attr `ascendc_op_para_size`（没有名字就 `AppendIrAttrName`，没有值就设 `2MB`）。
  原因：ops-nn 在 `ascendc_op_para_size` 后面加了 private attr `fixed_shift_value`。子图拷贝后若缺 `ascendc_op_para_size`，二次 tiling 按 IR 属性下标构造上下文会越界。
3. 遍历 **原始输入锚点下标** 连边：
  - `peer == nullptr`：这是 optional 空槽（没接 bias / scale0），**continue，不压缩下标**。
   若改成「有边的输入从 0 连续编号」，scale0 会错位到 bias 槽，tiling 入参全乱。
  - filter（index=1）：要改 **两份** desc，因为它们是拷贝关系，改一份不会同步到另一份。
    - `conv_input_desc`：`conv_subgraph` 里卷积节点的 filter，给 ops-nn 二次 tiling；那边期望 `FRACTAL_Z`。
    - `node`（AscBackend）的输入 1：融合节点自己的 filter 描述，后续 Autofuse tiling / 图上校验会读它。
    两边都用下标 1，是因为 cube 输入排在 AscBackend 前面，filter 仍落在 1。
4. `node->GetOpDesc()->SetExtAttr("conv_subgraph", sub_graph)`。

**输出**：融合节点上的 ExtAttr `conv_subgraph`，其内容是一张小 ComputeGraph：

```text
拷贝后的 Data/Const（x、filter、以及实际接上的 bias/scale0）
        │  边的 dst index 与原始 GE 锚点一致（空槽保持空洞）
        ▼
拷贝后的 ExtendConv2D / Conv2DV2
  - 带齐 ascendc_op_para_size
  - filter format = FRACTAL_Z
```

Lifting 结束后图上仍是一个 `AscBackend`，额外多了这份子图，专门给运行时 tiling 用，**不是**再执行一遍卷积计算。

`conv_subgraph` **的生产—保存—读取位置**：

```text
生产：
  liftings.cpp:61 CreateConvSubgraphAttr
  liftings.cpp:69 CopyOpDesc(ExtendConv2D / Conv2DV2)
  liftings.cpp:83-109 按原输入锚点构图

保存：
  liftings.cpp:116
  AscBackend::OpDesc.SetExtAttr("conv_subgraph", sub_graph)

读取：
  op_tiling_rt2.cc:919-925
  OpDesc.TryGetExtAttr("conv_subgraph")
    → 遍历子图
    → 找 ExtendConv2D / Conv2DV2

消费：
  op_tiling_rt2.cc:867 AutofuseNodeWithConvTiling
  用子图卷积节点构造 Operator，调用 ops-nn tiling
```

---



## 6. 运行时二次 tiling

**文件**：`base/common/op_tiling/op_tiling_rt2.cc`

`AicoreRtParseAndTiling` 发现当前 op 是 Autofuse 节点后：

1. 先看有没有 `matmul_subgraph`（本次不展开）。
2. 再 `TryGetExtAttr("conv_subgraph")`。
3. 子图里找到 type 为 `Conv2DV2` **或** `ExtendConv2D` 的节点，进入 `AutofuseNodeWithConvTiling`。

`AutofuseNodeWithConvTiling` 两步，顺序固定：


| 步                 | 调谁                                                              | 得到什么                                             |
| ----------------- | --------------------------------------------------------------- | ------------------------------------------------ |
| 1 Cube tiling     | 子图里的 ExtendConv2D/Conv2DV2，走 ops-nn `RtParseAndTiling`          | tiling key / tiling data / workspace / `aic_num` |
| 2 Autofuse tiling | 融合节点本身                                                          | vector 侧 tiling / `aiv_num`                      |
| 3 合成              | `block_dim = (aic_num*2 < aiv_num) ? ceil(aiv_num/2) : aic_num` | 融合 kernel 的核数                                    |


**输入**：AscBackend + ExtAttr `conv_subgraph`。
**输出**：`OpRunInfoV2`（tiling key、tiling data、workspace、block_dim），供 kernel launch。

若 lifting 没挂上 `conv_subgraph`，这里走不到卷积 tiling，CV 融合 kernel 的 cube 切分是错的。

**更完整的数据链**：

```text
op_tiling_rt2.cc:895 AicoreRtParseAndTiling(AscBackend, platform_infos, run_info)
      ↓ IsAutofuseNode
op_tiling_rt2.cc:919 TryGetExtAttr(conv_subgraph)
      ↓
op_tiling_rt2.cc:924 识别 ExtendConv2D / Conv2DV2
      ↓
op_tiling_rt2.cc:867 AutofuseNodeWithConvTiling
      ├─ :880 RtParseAndTiling(conv_op, ...)
      │       callback → HandleCubeTilingCallback
      │       产出 cube tiling data/workspace/aic_num
      ├─ :885 AutofuseNodeTiling(fused_op, ...)
      │       callback → HandleAutofuseTilingCallback
      │       补充 vector tiling/aiv_num
      └─ :887 合成 new_block_dim，写回 run_info
```

一般算子没有 `conv_subgraph` / `matmul_subgraph` 时，Autofuse 节点直接走 `AutofuseNodeTiling`。只有 cube+vector 融合需要先取得 cube 专用 tiling，再与 Autofuse 侧结果合成。

---



## 7. 数据链总表：在哪里生产，保存在哪里，在哪里消费

```text
GE 图: ExtendConv2D (+ Relu/Add ...)
        │ InferSymbolShape / CompleteStaticSymbolicShapeForConv（仅卷积节点）
        │ SymbolicDescAttr
        ▼
Lowering: Conv2DAttr + ASCIR ExtendConv2D{Bias,Scale,...} + KernelBox
        │ FusedLoopToAscBackendOp
        ▼
AscBackend（AutoFuseAttrs: origin + AscGraph）
        │ can_fuse 可能再并 vector（整 desc 拷贝）
        ▼
AscBackend + ExtAttr conv_subgraph     ← GE 仓主产物
        │ graph-autofusion codegen（另一仓）
        ▼
融合 kernel 源码 / 二进制
        │ AicoreRtParseAndTiling
        ▼
OpRunInfoV2 → launch
```


| 数据                     | 生产位置                                                 | 保存位置                                          | 消费位置                                                       |
| ---------------------- | ---------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| 卷积输出符号 shape           | `infer/conv2d.cc:544 InferShape4Conv2D`              | 输出 `GeTensorDesc::SymbolicDescAttr`           | `loop_common.cpp:57 GetBufferShape`，并被后继算子 lowering 使用     |
| 静态 shape 兜底            | `CompleteStaticSymbolicShapeForConv`                 | 卷积节点输出 desc                                  | 依赖 `GetBufferShape` 的卷积 lowering                           |
| `Conv2DAttr`           | `lowering_impl.cpp:1296-1417`                        | 先作为 `StoreConv2DOp::attrs_`                   | `asc_overrides.h:411 StoreConv2D`                          |
| `has_bias` / `has_scale0`  | `SetConv2DBiasAndOffsetW` / `SetExtendConv2DFixpipeParams` | `Conv2DAttr` → ASCIR ir_attr → Python cube_attributes | `build_conv_args` 还原 ExtendConv2D 10 槽（2/4 按标志，3/5–9 恒空） |
| `KernelBox`            | `loop_api.cpp:577 StoreConv2D` / 其它 Store API        | 输出 OpDesc 的 `LoweringResultAttrs`             | `lowerings.cpp` 的融合与 `BuildOpDescForKernelBox`             |
| `AscGraph`             | `lowerings.cpp:431 kernel_box.Realize<AscOverrides>` | `AutoFuseAttrs::asc_graph`                    | can_fuse、postprocess、graph-autofusion codegen              |
| origin nodes           | `lowerings.cpp:454 SetOriginNodes`                   | `AutoFuseAttrs`                               | Lifting、DFX、`CreateConvSubgraphAttr`                       |
| `AscBackend`           | `lowerings.cpp:529 FusedLoopToAscBackendOp`          | GE ComputeGraph                               | can_fuse、Lifting、codegen、运行时 tiling                        |
| 新融合节点 tensor desc      | `backend_utils.cpp:1826/1867`                        | 新 AscBackend 的 OpDesc                         | 后续融合、codegen、tiling                                        |
| `conv_subgraph`        | `liftings.cpp:61 CreateConvSubgraphAttr`             | AscBackend OpDesc ExtAttr                     | `op_tiling_rt2.cc:919`                                     |
| cube tiling 结果         | `op_tiling_rt2.cc:880 RtParseAndTiling` callback     | `OpRunInfoV2`                                 | 与 Autofuse tiling 合成                                       |
| 最终 tiling/启动参数         | `op_tiling_rt2.cc:867 AutofuseNodeWithConvTiling`    | `OpRunInfoV2`                                 | runtime kernel launch                                      |




### 7.1 用一个例子串起来

原图：

```text
Data(x) ─┐
Const(w) ├─ ExtendConv2D ─ Relu ─ Add(residual) ─ NetOutput
Scale ───┘
```

1. `InferShape4Conv2D` 算出卷积输出符号 shape；`CompleteStaticSymbolicShapeForConv` 只给卷积节点上可能遗漏的静态输出补符号 shape。
2. `LoweringGraph` 分别把 ExtendConv2D、Relu、Add 翻译成 LoopOp；每个输出对应一个 KernelBox。若某算子不支持或 shape 不满足条件，则 `FallbackLowering`，不会破坏原图语义。
3. KernelBox 融合后 realize 出 AscGraph，里面可能是 `ExtendConv2DScale → Relu → Add`；GE 图上用一个 AscBackend 代表它。
4. can_fuse 还会评估是否能把更多邻居并进来；新节点创建时要完整复制 tensor desc。
5. Lifting 看到它含 cube 且融合值得保留，不拆回；从 origin nodes 复制 ExtendConv2D 及输入，挂成 `conv_subgraph`。
6. graph-autofusion 根据 AscGraph 生成一个 CV 融合 kernel。
7. tiling 时，先用 `conv_subgraph` 调 ops-nn 得到 AIC 卷积 tiling，再对 AscBackend 求 AIV 融合 tiling，最后写入同一个 `OpRunInfoV2`。
8. runtime 用 `OpRunInfoV2` 启动融合 kernel；`conv_subgraph` 本身不会作为第二张执行图运行。

---



## 8. 本次 GE 改动文件对照


| 文件                                         | 角色                                                                                       |
| ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `infer/conv2d.cc`                          | ExtendConv2D 符号 shape；pad_mode 下标 7                                                      |
| `asc_ir_lowerer.cpp`                       | `CompleteStaticSymbolicShapeForConv`：非卷积节点跳过，只补卷积输出的静态 SymbolicDescAttr        |
| `op_helper/cube.h`                         | `Conv2DAttr` 扩展字段                                                                        |
| `lowering_impl.cpp`                        | `InnerLowerConv2D` / `LowerExtendConv2D`                                                 |
| `asc_overrides.h`                          | 按 bias/scale0 建 ASCIR 变体                                                                 |
| `loop_api.cpp`                             | `IsStaticShape` 只检查 dst / 已连接 input 这条边的 shape                                        |
| `lowerings.cpp`                            | KernelBox 收集仍扫全部输出锚点                                                               |
| `autofuse_utils.h/.cpp`                    | cube 类型集合加入 ExtendConv2D 四变体                                                             |
| `liftings.cpp`                             | skip lifting 识别 ExtendConv2D；`CreateConvSubgraphAttr` 按锚点连边、补 para_size、filter FRACTAL_Z |
| `backend_utils.cpp`                        | 融合新节点整份拷贝 desc                                                                           |
| `temporary_dependencies/ascir/ascir_ops.h` | ASCIR 头文件同步                                                                              |
| `op_tiling_rt2.cc`                         | conv_subgraph 中识别 ExtendConv2D，二次 tiling                                                 |


`graph-autofusion` 侧（非本文展开）：ASCIR REG、codegen 固定 `x, filter, bias, offset_w, scale0` 槽位、`conv2d_v2` 增加 `scaleGM` 与新 tiling key 维；二次 tiling 用 `has_bias` / `has_scale0` 还原 ExtendConv2D 逻辑输入，不再传 `nullptr_inputs_index`。
