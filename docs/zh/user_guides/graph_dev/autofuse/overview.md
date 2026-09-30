# 概述

## 什么是AutoFuse

AutoFuse是CANN生态中面向昇腾系列芯片的自动算子融合组件。它接收GE、Inductor等图编译组件经图转换、Lowering和融合范围判定后产出的融合子图及统一IR，在已确定的融合范围内完成调度优化、Tiling求解与代码生成，最终输出高性能的Ascend C融合算子。通过将多个原本独立执行的算子融合为单一Kernel，AutoFuse可减少中间结果的GM读写、Kernel启动次数以及Host-Device调度开销，从而显著提升昇腾NPU上的模型执行性能。

收益原理如下图所示，自动融合通过将多个算子合并为单个算子，理论上在MTE搬运和动态Shape调度开销方面均可获得一定收益；对于小Shape、MTE Bound的推荐网络，一般都能获得正收益。

**图1**  收益原理
![图1示例](../figures/benefit_principle.png "收益原理")

<!-- npu="950,A3,910b" id4 -->
自动融合特性仅支持如下产品型号：
<!-- end id4 -->

<!-- npu="950" id1 -->

- Ascend 950PR/Ascend 950DT
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品
<!-- end id3 -->

## 系统架构

自动融合整体逻辑结构如下图所示：

**图2**  系统架构<a id="fig1"></a>
![图2示例](../figures/autofusion_arch.png "技术路线")

如上图所示，自动融合方案基于昇腾NPU底层统一的AscendLoopIR（面向Ascend C编程语言建模的IR）以及配套的Schedule和代码生成等能力，构建了两条融合实现路径：

- **GE路径**：基于昇腾自研的GE框架，侧重昇腾NPU亲和性，由GE完成符号化、Lowering和融合范围判断，AutoFuse负责后端调度、切分和代码生成。
- **Inductor路径**：对接PyTorch Inductor，侧重生态适配，复用Inductor的融合范围识别能力，后端处理仍由AutoFuse完成。

下面详细介绍各个组件的作用。

### 前端适配

前端适配负责将PyTorch、TensorFlow等主流深度学习框架的模型图转换为AutoFuse可处理的图IR。

- **GE路径**：在线场景通过TorchAir或TensorFlow Adapter将AtenIR、GraphDef等图表示转换为AscendIR，进入GE图编译流程；离线场景通过ATC内置Parser将TensorFlow、ONNX等格式的模型解析为AscendIR。

- **Inductor路径**：将PyTorch的AtenIR转换为InductorIR，采用PyTorch Inductor路径。其中，TorchNPU模块的相关说明请参见[PyTorch项目](https://gitcode.com/Ascend/pytorch)。

### Graph Engine

Graph Engine作为AutoFuse的前端，负责对AscendIR图进行符号化推导、Lowering和CanFuse融合条件判断，确定可融合的算子范围以及融合后的计算表达。

| 步骤                                          | 职责                                                         |
| --------------------------------------------- | ------------------------------------------------------------ |
| **[符号化](symbolization.md)**                | 用符号表达算子Shape，增强动态Shape处理能力，提供化简、推导和Guard功能，为后续循环轴合并和内存优化提供关键信息。 |
| **[Lowering](lowering.md)**                   | 将高层级AscendIR转换为低层级AscendLoopIR，以贴近Ascend C语义表达计算逻辑，确定融合结构和数据依赖。 |
| **[CanFuse（融合策略）](fusion_strategy.md)** | 从语义正确性、是否可表达和资源预算三层确定融合边界。         |

### AutoFuse

AutoFuse作为上层GE或Inductor的后端，是自动融合编译的核心。它接收已确定的融合范围，通过Schedule、Codegen和Auto Tiling三个核心模块完成融合Kernel的调度、代码生成和切分策略求解，并借助Ascend C API提供算子接口支持。

| 模块                              | 职责                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| **[Schedule](schedule.md)**       | 调度策略生成：计算重排、循环合并、并行优化、内存优化和多模板生成。 |
| **[Codegen](codegen.md)**         | 代码生成：解析调度图，生成Host侧和Device侧代码。             |
| **[Auto Tiling](auto_tiling.md)** | Tiling求解：在UB约束下求解Tile大小和分核策略，评估切分方案性能，选择合适的模板和切分策略。 |

### 编译与运行

AutoFuse生成的Host和Device源码由[毕昇编译器](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/compiler/BishengCompiler/atlas_bisheng_10_0001.html)进一步编译为Host侧共享库和Device侧Kernel二进制。运行时，上层框架根据实际输入准备Tiling参数并启动融合Kernel，**Runtime**负责设备资源管理、Kernel下发和执行。对于原本由多个Kernel完成的算子链，融合后通常可减少Kernel启动次数和中间Tensor的全局内存读写，从而降低数据搬运和调度开销，提升昇腾AI处理器的硬件资源利用率和模型执行性能。

## 关键技术方案

AutoFuse按计算特征将网络算子分为两类：一类是Elemwise、Broadcast和View类（Transpose、Slice、Split）等基础计算类型；另一类是Reduce、Concat和MatMul等在基础计算类型上扩展融合能力的计算类型。各类扩展融合能力均需支持与基础计算类型进行融合。

### 算子类型

下表列出了主要支持的算子类型及其对应的计算单元：

| 算子类型 | 计算单元 | 典型算子示例 | 适用场景 | 说明 |
| :------------ | :--------- | :------------------------------ | ------------------------------------------------------------ | ------------- |
| **Elemwise** | Vector | Add、Mul、Abs、Exp、Relu、Cast | GE路径、Inductor路径 | 逐元素计算，每个输出元素与输入元素一一对应。 |
| **Broadcast** | Vector | BroadcastTo、BiasAdd | GE路径、Inductor路径 | 广播计算，将较小Shape的数据沿广播轴扩展，再执行逐元素计算。 |
| **Concat** | MTE/Vector | Concat | GE路径、Inductor路径 | 拼接计算，沿指定轴将多个Tensor拼接为一个Tensor。 |
| **MatMul** | Cube | MatMul | Inductor路径 | 矩阵计算，包括矩阵乘和卷积等。 |
| **Reduce** | Vector | ReduceSum、ReduceMax、ReduceMin | GE路径、Inductor路径 | 规约计算，沿指定轴对多个元素进行聚合。 |
| **泛Norm** | Vector | LayerNorm、RMSNorm | Inductor路径 | 由同轴Reduce、Broadcast和Elemwise等计算组合形成的归一化计算模式，并非单一算子。 |
| **View** | MTE/Vector | Transpose、Slice、Split | Inductor路径 | 视图变换，改变数据的逻辑形状、轴序或切分方式。 |

### 支持的融合能力

AutoFuse当前主要支持VV和CV两类融合：

| 融合类型                       | 可融合算子类型                                               |
| :----------------------------- | :----------------------------------------------------------- |
| **VV融合**（Vector + Vector） | 支持Elemwise、Broadcast、View（包括Transpose、Slice和Split）、Reduce、Concat等Vector类算子的融合。 |
| **CV融合**（Cube + Vector）   | 支持Cube类算子与Vector类算子的融合。                     |

## 融合原理

自动融合的实现包含两部分：自动确定融合范围，以及根据融合范围自动生成融合Kernel源码和Kernel二进制。前者称为自动融合前端，后者称为自动融合后端（对应[图2](#fig1)中AutoFuse及后面部分）：

前端主要根据一定规则或配置判断哪些算子能够融合，并确定融合算子的融合范围。融合范围用FusedGraph表达，如下图[图3](#fig2)所示，FusedGraph内部包含\>=1个AscBackend节点。AscBackend可理解为一个类似于ge::op::partitionedcall的Ascend IR算子，携带一个子图对象；一个AscBackend节点携带一个AscGraph属性，一个AscGraph内包含多个AscIR节点。AscIR与AscGraph的详细介绍请参见[AscIR与AscGraph](../appendix/ascir_and_ascgraph.md)。

**图3**  FusedGraph<a id="fig2"></a>
![图3示例](../figures/fused_graph.png "FusedGraph")

**图4**  AscBackend对应的AscGraph
![图4示例](../figures/ascbackend_ascgraph.png "AscBackend对应的AscGraph")

后端的实现包含Schedule/Codegen/Auto Tiling等。后端接收到FusedGraph后，Schedule首先针对计算/搬运类节点分别生成对应的TilingGroup，并通过TilingGroup合并策略，构建适用于整个融合算子的归一化TilingGroup，最终基于该归一化TilingGroup生成FusedGraph的Tiling策略。经Codegen实现和Auto Tiling，生成Kernel源码、tiling\_func源码、tiling\_data源码，并编译生成Host和Device交付件。自动融合的生效方式请参见[AutoFuse开启方式](autofuse_enable.md)，[Schedule](schedule.md)/[Codegen](codegen.md)/[Auto Tiling](auto_tiling.md)的实现原理请分别参见对应章节。
