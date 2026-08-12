# 样例使用指导

## 功能描述

本样例是一个推荐网络的高性能推理示例，演示了如何利用**GE API**和**ACL API**实现推理流程，并通过**多实例**、**批量H2D(Host-to-Device)内存拷贝** 和**AI Core控核**技术来优化推荐网络的推理吞吐性能。

## 目录结构

```
├── src
│   ├──model_inference.cpp               // 模型推理实现文件
│   ├──model_inference.h                 // 模型推理头文件
│   ├──recomand_random_input.cpp         // 推理客户端，构造随机数据，调用模型推理
├── CMakeLists.txt                       // 编译脚本
```

## 环境要求
- 已完成[昇腾AI软件栈在开发环境上的部署](../../docs/zh/quick_install.md)

## 实现步骤
1. 图构建：使用aclgrphParseTensorFlow解析模型文件，构建GE计算图。
2. 图编译与加载：通过GE API(ge::Graph, ge::Session)进行图的编译(Compile)和加载(Load)。
3. 数据准备与执行：根据模型输入结构构造随机数据，使用GE API进行推理。
4. 性能优化：
   - 多实例：通过多线程创建多个推理实例，提升系统并发处理能力。
   - 控核：在创建ge::Session时，通过options参数指定单算子可使用的AI Core数量。
   - 批量H2D：使用aclrtMemcpyBatch接口合并多次内存拷贝操作，减少开销。

## 构建验证

假设toolkit的安装目录为install_path, 例如`/home/HwHiAiUser/Ascend/cann/`

1. 配置环境变量。
   ```bash
   source ${install_path}/set_env.sh
   ```
2. 执行如下命令，创建data目录，并[下载](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ge/DCN_v2.pb)模型pb文件，放入data目录。
   ```shell
   mkdir data
   ```
3. 执行如下命令，编译生成可执行文件
   ```
   mkdir build && cd build
   cmake ..
   make
   ```
   执行后，在**build**目录下产生recomand_exec可执行文件

4. 执行如下命令，测试推荐网络不开优化特性时的推理吞吐性能。
   ```shell
   ./recomand_exec
   ```
5. 测试开启4个多实例、开启批量H2D、控核时的网络推理性能，其中aiCoreNum参考[GE图引擎接口 -> 数据类型 -> options参数说明](https://www.hiascend.com/document/redirect/CannCommunityAscendGraphApi)按照实际硬件信息调整。
   ```shell
   ./recomand_exec --multiInstanceNum=4 --enableBatchH2D=true --aiCoreNum="16|16"
   ```

## 多流参数寻优任务

### 功能说明

在 CANN 社区版 9.2.0 环境中，本样例支持使用 GE 高级调优功能“多流增强”，通过自动分配图内执行流，挖掘推荐网络中算子之间的并行执行空间。

> [!NOTE]
> 多流增强功能需在 CANN 社区版 9.2.0 环境中使用。请前往 [CANN 软件下载页面](https://www.hiascend.com/cann/download?versionId=770&ids=d806%2Ch0501%2Ch0601%2Ch0703)，按以下条件选择安装包：
>
> - 版本类型：`Weekly`
> - 产品系列：A3 系列产品
> - CPU 架构：AArch64
> - 操作系统：openEuler
> - 安装方式：离线安装
>
> 下载完成后，请按照页面中的安装指导部署环境。

本样例属于静态 Shape 在线编译场景，通过 GE Session Option `ge.autoMultistreamParallelMode` 配置多流并行模式。

| 参数值 | 说明 |
| :--- | :--- |
| `cv` | 开启 Cube 算子与 Vector 算子的并行执行 |
| `LoadBalance:N` | 采用负载均衡算法，将算子分配到最多 `N` 条流上执行 |
| `MainStream:N` | 采用主流算法，串行算子在主流执行，其他可并行算子分配到其他流 |
| 不配置 | 不启用自动多流并行优化，作为默认性能基线 |

其中，`N` 为正整数，取值范围为 `[1, 64]`。配置的流数量超过实际可用计算资源时，性能可能下降。详细说明请参见 [`ge.autoMultistreamParallelMode` 参数说明](../../docs/zh/api/graph_engine_api/cpp/ge/options_params/experimental_parameters.md#geautomultistreamparallelmode)。

### 命令行参数

为了便于遍历不同的多流策略，本样例提供 `--multiStreamParallelMode=<模式>` 命令行参数。样例解析该参数后，在创建 `ge::Session` 前将其转换为：

```cpp
session_options["ge.autoMultistreamParallelMode"] = "<模式>";
```

不传入 `--multiStreamParallelMode` 时，不设置上述 Session Option，并将该配置作为默认性能基线。

### TODO：寻找最优多流增强参数

请在 CANN 社区版 9.2.0 环境中测试不同的多流并行模式，找到本样例在指定环境和测试条件下的最优参数，并填写以下结论后提交 PR：

```text
ge.autoMultistreamParallelMode=<待填写>
```

对应的样例命令行参数为：

```text
--multiStreamParallelMode=<待填写>
```

例如，使用 `cv` 模式时执行：

```bash
./recomand_exec \
    --runs=100 \
    --batchSize=128 \
    --multiInstanceNum=1 \
    --multiStreamParallelMode=cv
```

#### 性能评价标准

多流参数寻优以样例输出的平均图执行与流同步时延作为主评价指标，数值越小表示性能越优：

```text
Average ExecuteGraphWithStreamAsync latency: <平均耗时> ms
```

对每个 `--multiStreamParallelMode` 候选参数，固定 `--runs=100` 和 `--multiInstanceNum=1`。前 10 个成功完成的任务作为预热，不统计和输出 benchmark 耗时；后 90 个成功任务用于正式统计。以这 90 个成功任务的算术平均时延作为评价结果，并将平均时延最小者作为当前测试环境下的推荐参数。仅当日志显示 `Success count: 100` 时结果有效；如有任务失败，应排查原因并重新测试。

单次耗时使用 `std::chrono::steady_clock` 统计，起点位于 `ExecuteGraphWithStreamAsync` 调用前，终点位于 `aclrtSynchronizeStream` 成功返回后，统计范围严格为 `ExecuteGraphWithStreamAsync + aclrtSynchronizeStream`。任务排队、Device 内存申请与释放、H2D 输入拷贝和 D2H 输出拷贝均不计入统计。

平均时延计算方式为：

```text
Average ExecuteGraphWithStreamAsync latency = 后 90 个成功任务的 ExecuteGraphWithStreamAsync 与 aclrtSynchronizeStream 耗时总和 / 90
```

样例同时输出后 90 个成功任务的单次耗时，`run` 从 0 开始编号，便于核对平均值和观察时延波动：

```text
BENCHMARK ExecuteGraphWithStreamAsync run=<完成序号> latency_us=<耗时>
```

样例原有的 `Average execution latency` 输出及其计时逻辑保持不变：计时起点位于 `ModelInference::GraphTask::operator()()` 开始处，终点位于 D2H 输出拷贝及 Device 内存释放之后。该指标仍按以下方式计算：

```text
Average execution latency = 成功任务的原始执行耗时总和 / 成功任务数
```

由于 `Average execution latency` 包含 Device 内存申请与释放、H2D 输入拷贝、图执行与流同步以及 D2H 输出拷贝等更宽范围的耗时，因此仅保留用于观察样例原有指标，不作为本次图内多流参数的推荐依据。

> [!NOTE]
> 本样例使用 DCN v2 推荐网络和随机输入测试图执行性能。寻优时仅改变 `--multiStreamParallelMode`，其他参数保持一致，并统一按照“预热 10 个成功任务、正式统计后 90 个成功任务”的方式进行测试。

参数固定原则：

- `--batchSize`：保持输入 Shape 和单次任务计算量一致。
- `--runs=100`：固定提交 100 个推理任务，其中前 10 个成功任务用于预热，后 90 个成功任务用于平均时延统计。
- `--multiInstanceNum=1`：排除多实例并行的影响，仅评价图内多流性能。
- `--enableBatchH2D`、`--aiCoreNum`：保持其他性能优化配置一致。

日志字段含义：

- `BENCHMARK`：便于脚本检索性能日志的固定标识。
- `run`：预热结束后正式统计任务的完成序号，不是多流数量。固定 `--multiInstanceNum=1` 时可视为顺序执行编号；多实例场景下表示任务完成顺序，不保证与提交顺序一致。
- `latency_us`：正式统计任务的单次耗时，单位为微秒。
- `Average ExecuteGraphWithStreamAsync latency`：后 90 个成功任务中 `ExecuteGraphWithStreamAsync + aclrtSynchronizeStream` 的平均时延，单位为毫秒，是多流参数寻优的主评价指标。
- `Average execution latency`：按样例原有计时范围计算的所有成功 recommendation 任务平均时延，单位为毫秒。

## 性能差异对比

在 Ascend 910C 平台测试不同配置下的吞吐量（TPS）与时延（ms）表现：

| 配置方案 | 吞吐 / 时延 |
| :--- | :--- |
| 单实例 | 745,55 TPS / 1.471 ms |
| 单实例 + 批量 H2D | 131,191 TPS / 0.792 ms |
| 多实例（4） | 155,104 TPS / 2.089 ms |
| 多实例（4）+ 控核（16\|16） | 185,415 TPS / 1.797 ms |
| 多实例（4）+ 控核（16\|16）+ 批量 H2D | 251,877 TPS / 1.285 ms |
