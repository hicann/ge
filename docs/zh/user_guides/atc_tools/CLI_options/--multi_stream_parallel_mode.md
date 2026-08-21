# --multi\_stream\_parallel\_mode

## 产品支持情况

<!-- npu="950" id6 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id6 -->
<!-- npu="A3" id5 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id5 -->
<!-- npu="910b" id4 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id4 -->
<!-- npu="310b" id3 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id3 -->
<!-- npu="310p" id2 -->
- Atlas 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910" id1 -->
- Atlas 训练系列产品：支持
<!-- end id1 -->

<!-- @ref: ge/res/docs/zh/user_guides/atc_tools/CLI_options/--multi_stream_parallel_mode_res.md#id1 -->

## 功能说明

**调试功能扩展参数，暂不支持应用于生产环境，后续版本会作为正式功能更新发布。**

该参数适用于静态/动态shape图场景，开发者可通过配置此参数控制多流并行模式的自动分配策略，以提升图执行性能。

## 关联参数

- 该参数与[--enable\_single\_stream](--enable_single_stream.md)互斥，不可同时开启。若同时开启，编译会报错并终止。

- 动态shape场景下，需先通过环境变量[ENABLE\_DYNAMIC\_SHAPE\_MULTI\_STREAM](../../env_vars/ENABLE_DYNAMIC_SHAPE_MULTI_STREAM.md)使能动态shape多流后，此参数才用于选择多流并行算法，不能单独开启动态shape多流。

## 参数取值

**参数值：**

- cv：开启Cube算子与Vector算子的并行执行功能。
- LoadBalance:N：负载均衡算法，将所有算子均匀分布在N条流上执行。
- MainStream:N：主流算法，串行算子分布在主流上执行，其他可并行算子分布在其他流上执行。
- 空字符串（默认值）：不启用任何多流并行优化。

**参数值约束：**

N为最大流数量，正整数，取值范围为\[1,64\]，超出范围会导致编译失败。

## 推荐配置及收益

无。

## 示例

```bash
atc --multi_stream_parallel_mode=LoadBalance:8 ...
```

## 使用约束

- 该参数仅限于推荐类型网络使用。
- 动态shape多流场景下，使用此功能需要先通过[ENABLE\_DYNAMIC\_SHAPE\_MULTI\_STREAM](../../../user_guides/env_vars/ENABLE_DYNAMIC_SHAPE_MULTI_STREAM.md)使能动态shape多流，再配置此参数。
