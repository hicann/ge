# AutoFuse环境变量参考

本文汇总GE路径和Inductor路径下，AutoFuse自动融合功能运行和调测过程中常用的环境变量及控制项。

## 公共环境变量

以下环境变量为AutoFuse核心功能控制项，同时适用于GE路径和Inductor路径。

| 环境变量             | 适用场景             | 作用、取值与使用约束                                         |
| :------------------- | :------------------- | :----------------------------------------------------------- |
| `AUTOFUSE_FLAGS`     | GE路径、Inductor路径 | AutoFuse功能控制。多个控制项使用英文分号分隔。               |
| `AUTOFUSE_DFX_FLAGS` | GE路径、Inductor路径 | AutoFuse调测控制，用于融合图Dump、代码生成调测和Auto Tiling调测；多个控制项使用英文分号分隔。 |

### AUTOFUSE_FLAGS控制项

`AUTOFUSE_FLAGS`用于控制AutoFuse功能。

**仅开启基础AutoFuse融合功能（最简配置）：**

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

对于G路径，以上配置用于开启基础AutoFuse融合功能。

对于Inductor路径，AutoFuse通过`torch.compile`配置`ascendc`后端来开启，`AUTOFUSE_FLAGS`主要用于配置扩展功能。

下表列出所有可选控制项，可根据需要组合使用：

| 控制项                                   | 适用场景               | 作用、取值与使用约束                                 |
| :--------------------------------------- | :--------------------- | :--------------------------------------------------- |
| `--enable_autofuse`                      | GE路径                 | 控制整体自动融合功能是否开启。                       |
| `--autofuse_enable_pass`                 | GE路径                 | 控制指定的扩展融合能力是否开启。                     |
| `--autofuse_disable_pass`                | GE路径                 | 控制指定的扩展融合能力是否关闭。                     |
| `--autofuse_enable_pgo`                  | GE 路径、Inductor 路径 | 开启PGO调优，通过预先上板采样选择性能更优的 Tiling。 |
| `--autofuse_enhance_precision_blacklist` | GE路径                 | 控制指定AscIR算子类型是否跳过精度提升。              |
| `--experimental_enable_jit_executor_v2`  | GE路径                 | 开启切图编译。                                       |
| `--max_fusion_size`                      | GE路径                 | 设置单个融合算子最多包含的节点数量。                 |
| `--recomputation_threshold`              | GE路径                 | 设置自动融合重计算阈值。                             |

示例：

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"
```

### AUTOFUSE_DFX_FLAGS控制项

`AUTOFUSE_DFX_FLAGS`用于AutoFuse编译、Auto Tiling和融合结果调测。

| 控制项                               | 适用场景              | 作用、取值与使用约束                       |
| :----------------------------------- | :-------------------- | :----------------------------------------- |
| `--autofuse_att_algorithm`           | GE路径、Inductor 路径 | 选择Auto Tiling求解算法。                  |
| `--att_accuracy_level`               | GE路径、Inductor 路径 | 控制Auto Tiling算法的求解精度。            |
| `--att_enable_multicore_ub_tradeoff` | GE路径、Inductor 路径 | 控制多核利用率与UB利用率权衡策略是否开启。 |
| `--att_ub_threshold`                 | GE路径、Inductor 路径 | 设置Auto Tiling的UB利用率阈值。            |
| `--att_corenum_threshold`            | GE路径、Inductor 路径 | 设置Auto Tiling的多核利用率阈值。          |
| `--att_profiling`                    | GE路径、Inductor 路径 | 控制Auto Tiling Profiling是否开启。        |
| `--autofuse_pgo_algo`                | GE路径                | 选择PGO调优算法。                          |
| `--autofuse_pgo_step_max`            | GE路径                | 设置PGO剪枝算法步长。                      |
| `--autofuse_pgo_topn`                | GE路径                | 设置参与PGO静态调优的候选解数量。          |
| `--codegen_compile_debug`            | GE路径、Inductor 路径 | 控制是否保留融合算子生成过程中的中间文件。 |
| `--debug_dir`                        | GE路径、Inductor路径  | 指定融合过程中AscGraph Dump图的保存路径。  |
| `--disable_lifting`                  | GE路径                | 控制是否关闭Lifting。                      |
| `--skip_node_names_cfg`              | GE路径                | 设置需要跳过融合的算子名称或算子类型。     |

示例：

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path/to/dump"
```

## 专属环境变量

以下环境变量为对应路径专属，仅用于特定路径的调测：

### Inductor路径专属

如下环境变量用于Inductor路径的编译或运行调测，不适用于GE路径：

| 环境变量                             | 说明                                                         | 使用方式                                      |
| :----------------------------------- | :----------------------------------------------------------- | :-------------------------------------------- |
| `TORCH_COMPILE_DEBUG`                | 开启PyTorch编译调试信息，并将编译中间产物保存到当前目录的`torch_compile_debug` 目录。以`autofused_`为前缀的目录通常表示 AscendC 后端生成的融合算子产物。 | `export TORCH_COMPILE_DEBUG=1`                |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | 禁用Inductor缓存，强制每次执行都重新编译。该配置会增加编译和图启动耗时，仅用于调试。 | `export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| `TORCHINDUCTOR_ASCENDC_DEBUG`        | 该环境变量用来开启MatMul算子融合。                           | `export TORCHINDUCTOR_ASCENDC_DEBUG="matmu"`  |
| `TORCHINDUCTOR_NPU_BACKEND`          | 该环境变量用于选择Ascend C后端。                             | `export TORCHINDUCTOR_NPU_BACKEND="ascendc"`  |
| `ASCEND_LAUNCH_BLOCKING`             | 使Ascend Kernel同步执行，便于定位首个报错的Kernel。该配置会降低执行性能，仅建议在问题定位时使用。 | `export ASCEND_LAUNCH_BLOCKING=1`             |

### GE路径专属

当前没有GE路径专属环境变量，所有GE路径相关控制项均已包含在AutoFuse共享环境变量中。

## 附录

### AUTOFUSE\_FLAGS环境变量控制点

#### --enable\_autofuse

控制整体自动融合功能是否开启。

**取值：**

- true：表示开启。
- false：（默认值）表示关闭。

不配置表示关闭整体自动融合功能，关闭时以下其他控制点全部失效，无论是否配置。

**配置示例：**

```text
--enable_autofuse=true
```

**使用约束：**

配置开启后，Elemwise算子与Broadcast算子间的自动融合能力便开启。

#### --autofuse\_disable\_pass

控制拓展融合能力是否关闭。

**取值：**

- reduce：控制Reduce类算子融合能力关闭。
- concat：控制Concat类算子融合能力关闭。

配置多个取值时使用英文逗号分割，默认为空，Reduce类、Concat类算子的融合能力目前默认不开启。

**配置示例：**

```text
--autofuse_disable_pass=reduce,concat
```

**使用约束：**

与--autofuse\_enable\_pass不能同时配置相同的取值。

#### --autofuse\_enable\_pass

控制拓展融合能力是否开启。

**取值：**

- reduce：控制Reduce融合能力开启。
- concat：控制Concat融合能力开启。

配置多个取值时使用英文逗号分割，默认为空，Reduce类、Concat类算子的融合能力目前默认不开启。

**配置示例：**

```text
--autofuse_enable_pass=reduce,concat
```

**使用约束：**

与--autofuse\_disable\_pass不能同时配置相同的取值。

#### --autofuse\_enable\_pgo

控制是否开启PGO（Profile-Guided Optimization，基于采样数据的优化）调优。

PGO调优是通过预上板采样，选取表现相对更好的Tiling以提升模型执行性能。

**取值：**

- true：表示开启。
- false：（默认值）表示关闭。

**配置示例：**

```text
--autofuse_enable_pgo=true
```

**使用约束：**

仅支持对静态图调优，首次配置时，不能与其他Profiling功能同时开启，后续配置时无限制。

#### --autofuse\_enhance\_precision\_blacklist

控制自动融合局部融合节点不做提升精度操作。

某个AscGraph中所有的AscIR都配置到黑名单中，该AscGraph才不会升精度。

**取值：**

- AscIR类型的字符串组合，多个类型使用英文逗号分割，默认值为空字符串，表示所有AscGraph都会提升精度。该参数中的AscIR类型是Dump图中对应节点type中的取值，例如Dump图中某节点type的取值为ge:Add，则实际配置为Add。
- all，表示将所有节点列入黑名单，从而禁止自动融合时的精度提升操作，避免不必要的升精度开销。

**配置示例：**

```text
--autofuse_enhance_precision_blacklist=Le,Where,Sub,Add,Sigmoid
```

**使用约束：**

- "Sum"、"Mean"、"Prod"不支持低精度类型，默认必须提升精度，因此不能配置到提升精度黑名单中，即使配置为all，仍旧提升精度。
- 不提升精度可以获得更高性能，但可能会引发精度问题。因此，在配置不提升精度后，用户需确保精度满足业务要求。
- Dump图的方法请参见[DUMP\_GE\_GRAPH](../../../user_guides/env_vars/DUMP_GE_GRAPH.md)环境变量。

#### --experimental\_enable\_jit\_executor\_v2

控制是否打开切图编译。

**取值：**

- true：表示开启。
- false：（默认值）表示关闭。

切图编译是将原始图在无法推导符号的边界进行断图处理，切成N个图，将上游图的输出作为下游图的输入hint继续符号推导并执行。

**配置示例：**

```text
--experimental_enable_jit_executor_v2=true
```

**使用约束：**

如下场景不支持切图：

- 动态分档场景
- 图上包含资源类算子（输入或者输入的类型是DT\_RESOURCE，如TensorArrayWrite）
- 图上包含V1版本控制算子（如："Switch","StreamSwitch","Merge","StreamMerge","Enter","Exit","LoopCond","NextIteration"）
- 开启数据预处理下沉
- 开启AOE调优的场景

#### --max\_fusion\_size

配置最多可融合节点的个数。

**取值：**[0-uint64_t类型最大值]，配置为0表示不融合。

**配置示例：**

```text
--max_fusion_size=64
```

上述示例表示最多支持融合64个Ascir节点。

**使用约束：**
无

#### --recomputation\_threshold

控制自动融合重计算阈值。

该参数用于设置单输出算子的引用阈值，当单输出节点引用个数超过阈值时，首次融合会在当前节点进行融合截断。

**取值：**0-255间的任意整数，默认为1。

**配置示例：**

```text
--recomputation_threshold=5
```

**使用约束：**

无。

### AUTOFUSE\_DFX\_FLAGS环境变量控制点

#### --autofuse\_att\_algorithm

控制Auto Tiling求解算法选择。

**取值：**

- HighPerf：高性能算法。
- AxesReorder：（默认值）轴排序算法。

**配置示例：**

```text
--autofuse_att_algorithm=HighPerf
```

**使用约束：**

- 轴排序算法是默认算法，高性能算法属于试验性算法，不一定能获取到更好的算子执行性能，只是算法理论上限较高。
- 配置非法值时会恢复为默认值。

#### --att\_accuracy\_level

控制Auto Tiling算法求解精度，理论上精度越高kernel性能越好。

**取值：**

- 1：（默认值）高精度求解。
- 0：低精度求解。

默认为高精度求解，保证Tiling耗时不至于过长。

**配置示例：**

```text
--att_accuracy_level=1
```

**使用约束：**

1. 高精度求解可能会得出更优的tiling解，但需要更长的Tiling执行时间，低精度求解则反之。
2. 配置非法值时会恢复为默认值。

#### --att\_enable\_multicore\_ub\_tradeoff

用于控制att\_corenum\_threshold和att\_ub\_threshold功能是否开启。

**取值：**

- true：开启。
- false：（默认值）不开启。

**配置示例：**

```text
--att_enable_multicore_ub_tradeoff=true
```

**使用约束：**配置非法值时会恢复为默认值。

#### --att\_ub\_threshold

控制Auto Tiling的Tiling策略，保证Tiling求解结果与算子实现结合，UB占用率不低于该控制点设置的取值（若UB占用率不满足指定阈值，则按可求解的最大UB占用率设置），该控制点可用于性能问题的定位。

**取值：**0-100间的任意整数，默认值为20。

**配置示例：**

```text
--att_ub_threshold=20
```

**使用约束：**

1. 该参数需要和[--att\_enable\_multicore\_ub\_tradeoff](#--att_enable_multicore_ub_tradeoff)配合使用，只有--att\_enable\_multicore\_ub\_tradeoff开启时，设置了--att\_ub\_threshold控制点才能生效。
2. 配置非法值时会恢复为默认值。

#### --att\_corenum\_threshold

控制Auto Tiling的Tiling策略，保证Tiling求解结果与算子实现结合，多核利用率不低于该控制点设置的取值（若多核利用率不满足指定阈值，则按可求解的最大多核占用率设置），该控制点可用于性能问题的定位。

**取值：**0-100间的任意整数，默认值为40。

当[--att\_enable\_multicore\_ub\_tradeoff](#--att_enable_multicore_ub_tradeoff)开启时，默认值为40，否则不会设置该策略。

**配置示例：**

```text
--att_corenum_threshold=40
```

**使用约束：**

1. 该参数需要和--att\_enable\_multicore\_ub\_tradeoff配合使用，只有--att\_enable\_multicore\_ub\_tradeoff开启时，设置的--att\_corenum\_threshold控制点才能生效。
2. 配置非法值时会恢复为默认值。

#### --att\_profiling

用于控制Auto Tiling的Profiling是否开启。

**取值：**

- true：开启Profiling特性。
- false：（默认值）关闭Profiling特性。

**配置示例：**

```text
--att_profiling=true
```

**使用约束：**

1. 该控制点仅用于定位Auto Tiling模块本身的执行时间问题。
2. 配置非法值时会恢复为默认值。

#### --autofuse\_pgo\_algo

控制PGO调优算法，不同算法求Tiling方式不同。

**取值：**

- core\_select：（默认值）使用控核算法。
- pruning：使用剪枝算法。

默认为控核算法，保证Tiling耗时不至于过长，剪枝算法理论上可能求出更优的Tiling解，但需要远长于控核算法的Tiling时间。

**配置示例：**

```text
--autofuse_enable_pgo=true --autofuse_pgo_algo=core_select
```

**使用约束：**

1. 该参数需要与--autofuse\_enable\_pgo配合使用，只有--autofuse\_enable\_pgo开启时，设置的--autofuse\_pgo\_algo控制点才能生效。
2. 配置非法值时会恢复为默认值。

#### --autofuse\_pgo\_step\_max

控制PGO剪枝算法步长。

**取值：**2-1024间任意2的幂次，默认为16。

**配置示例：**

```text
--autofuse_enable_pgo=true --autofuse_pgo_algo=pruning --autofuse_pgo_step_max=16
```

**使用约束：**

1. 该参数需要和--autofuse\_pgo\_algo配合使用，只有--autofuse\_pgo\_algo=pruning且生效时，设置的--autofuse\_pgo\_step\_max控制点才能生效。

    当step值较小时，可能会得出更优的tiling解，但需要更长的Tiling执行时间，step值较大时则反之。

2. 配置非法值时会恢复为默认值。

#### --autofuse\_pgo\_topn

设置参与PGO静态调优的解的数量。

**取值：**

- 0：选择所有解参与静态调优。
- 任意正整数：表示可选解的数量，默认值为5。

**配置示例：**

```text
--autofuse_enable_pgo=true --autofuse_pgo_topn=5
```

**使用约束：**

1. 该参数需要与--autofuse\_enable\_pgo配合使用，只有--autofuse\_enable\_pgo开启时，设置的--autofuse\_pgo\_topn控制点才能生效。
2. 配置非法值时会恢复为默认值。

#### --codegen\_compile\_debug

控制是否保留融合算子生成时的中间过程文件。

**取值：**

- true：保留文件。
- false：（默认为false）不保留文件。

**配置示例：**

```text
--codegen_compile_debug=true
```

设置为true，保留文件后，在脚本执行目录下将生成kernel\_meta\_\*文件夹，其中包含生成的kernel源码、Tiling源码、cmake工程以及编译结果，同时会在\{debug\_dir\}/autofuse\_compile\_debug/路径下生成融合算子Schedule各个过程中的AscGraph Dump图。

**使用约束：**无

#### --debug\_dir

指定融合中AscGraph dump图保存路径，支持大小写字母（a-z，A-Z）、数字（0-9）、下划线（\_）、中划线（-）、句点（.）、中文字符，执行用户需对该路径具有读、写、执行权限。若未指定该选项，文件将保存至脚本执行的当前目录。

**配置示例：**

```text
--codegen_compile_debug=true;--debug_dir=/path/to/dump
```

**使用约束：**需要先配置--codegen\_compile\_debug=true;打开自动融合debug开关。

#### --disable\_lifting

用于控制lifting功能是否关闭。

自动融合会将算子融合成名为AscBackend的新算子，lifting则将未能满足条件的AscBackend回滚到原算子。

**取值：**

- true：关闭lifting功能。
- false：（默认值）开启lifting功能。

**配置示例：**

```text
--disable_lifting=true
```

**使用约束：**

该控制点仅用于定位Ascbackend回滚结构的分析问题，开启该控制点可能会导致ApplyAdamD算子精度异常。

#### --skip\_node\_names\_cfg

设置跳过融合的算子名字或算子类型。

**取值：** 设置了算子名称或者算子类型的配置文件（.ini格式）路径以及文件名，每个算子单独一行，支持同时设置算子名称或者算子类型，算子类型必须为基于Ascend IR定义的算子的类型。

**配置示例：**

ini配置文件示例如下：

```ini
[ByNodeName]
op_name1
op_name2
[ByNodeType]
op_type1
op_type2
```

参数使用示例：

```text
--skip_node_names_cfg=./skil_node.ini
```

**使用约束：**

配置文件中的内容为非法值时，该配置项不生效。
