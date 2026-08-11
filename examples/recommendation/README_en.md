# Sample Usage Guide

## Function Description

This sample is a high-performance inference example for recommendation networks, demonstrating how to use **GE API** and **ACL API** to implement the inference workflow, and optimize inference throughput for recommendation networks through **multi-instance**, **batch H2D (Host-to-Device) memory copy**, and **AI Core control** technologies.

## Directory Structure

```tree
├── src
│   ├──model_inference.cpp               // Model inference implementation file
│   ├──model_inference.h                 // Model inference header file
│   ├──recomand_random_input.cpp         // Inference client, constructs random data, calls model inference
├── CMakeLists.txt                       // Build script
```

## Environment Requirements

- [Ascend AI Software Stack Deployment in Development Environment](../../docs/zh/quick_install.md) completed

## Implementation Steps

1. Graph Construction: Parse model files using aclgrphParseTensorFlow to build GE computation graph.
2. Graph Compilation and Loading: Compile (Compile) and load (Load) the graph through GE API (ge::Graph, ge::Session).
3. Data Preparation and Execution: Construct random data based on model input structure, use GE API for inference.
4. Performance Optimization:
   - Multi-instance: Create multiple inference instances through multi-threading to improve system concurrent processing capability.
   - Core control: When creating ge::Session, specify the number of AI Cores available for single operators through options parameter.
   - Batch H2D: Use aclrtMemcpyBatch interface to merge multiple memory copy operations to reduce overhead.

## Build and Verification

Assume toolkit installation directory is install_path, for example `/home/HwHiAiUser/Ascend/cann/`

1. Configure environment variables.

   ```bash
   source ${install_path}/set_env.sh
   ```

2. Execute the following command to create data directory, and [download](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ge/DCN_v2.pb) model pb file and place it in data directory.

   ```shell
   mkdir data
   ```

3. Execute the following command to compile and generate executable file

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

   After execution, recomand_exec executable file will be generated in **build** directory

4. Execute the following command to test recommendation network inference throughput without optimization features.

   ```shell
   ./recomand_exec
   ```

5. Test network inference performance with 4 multi-instances, batch H2D, and core control enabled, where aiCoreNum refers to [GE Graph Engine Interface -> Data Types -> options Parameter Description](https://www.hiascend.com/document/redirect/CannCommunityAscendGraphApi) and adjust according to actual hardware information.

   ```shell
   ./recomand_exec --multiInstanceNum=4 --enableBatchH2D=true --aiCoreNum="16|16"
   ```

## Multi-stream Parameter Tuning Task

### Feature Description

In CANN Community Edition 9.2.0, this sample supports the GE advanced tuning feature Multi-Stream Enhancement. The feature automatically allocates execution streams within the graph to expose opportunities for parallel execution among operators in a recommendation network.

> [!NOTE]
> Multi-Stream Enhancement requires CANN Community Edition 9.2.0. Go to the [CANN Software Download page](https://www.hiascend.com/cann/download?versionId=770&ids=d806%2Ch0501%2Ch0601%2Ch0703) and select the installation package using the following criteria:
>
> - Version type: `Weekly`
> - Product series: A3 series
> - CPU architecture: AArch64
> - Operating system: openEuler
> - Installation method: Offline installation
>
> After downloading the package, follow the installation instructions on the page to deploy the environment.

This sample uses Static Shape online compilation and configures the multi-stream parallel mode through the GE Session option `ge.autoMultistreamParallelMode`.

| Value | Description |
| :--- | :--- |
| `cv` | Enables parallel execution of Cube and Vector operators |
| `LoadBalance:N` | Uses the load-balancing algorithm to distribute operators across at most `N` streams |
| `MainStream:N` | Uses the main-stream algorithm, where serial operators run on the main stream and other parallelizable operators are distributed across other streams |
| Not configured | Disables automatic multi-stream parallelism and serves as the default performance baseline |

`N` is a positive integer in the range `[1, 64]`. Performance may degrade if the configured stream count exceeds the available compute resources. For details, see the [`ge.autoMultistreamParallelMode` parameter description](../../docs/zh/api/graph_engine_api/cpp/ge/options_params/experimental_parameters.md#geautomultistreamparallelmode).

### Command-line Argument

To simplify testing different multi-stream strategies, this sample provides the `--multiStreamParallelMode=<mode>` command-line argument. The sample parses this argument and converts it to the following Session option before creating `ge::Session`:

```cpp
session_options["ge.autoMultistreamParallelMode"] = "<mode>";
```

If `--multiStreamParallelMode` is omitted, the Session option is not set and the resulting configuration is used as the default performance baseline.

### TODO: Find the Optimal Multi-Stream Enhancement Parameter

In CANN Community Edition 9.2.0, test different multi-stream parallel modes, determine the optimal value for this sample under the specified environment and test conditions, and submit a PR with the following result:

```text
ge.autoMultistreamParallelMode=<to-be-determined>
```

The corresponding sample command-line argument is:

```text
--multiStreamParallelMode=<to-be-determined>
```

For example, to use `cv` mode, run:

```bash
./recomand_exec \
    --runs=100 \
    --batchSize=128 \
    --multiInstanceNum=1 \
    --multiStreamParallelMode=cv
```

#### Performance Evaluation Criteria

Use the average graph execution and stream synchronization latency reported by the sample as the primary metric for multi-stream parameter tuning. A lower value indicates better performance:

```text
Average ExecuteGraphWithStreamAsync latency: <average latency> ms
```

For each `--multiStreamParallelMode` candidate, set `--runs=100` and `--multiInstanceNum=1`. Treat the first 10 successfully completed tasks as warmup and do not measure or report their benchmark latency. Use the remaining 90 successful tasks for measurement. Evaluate each candidate using the arithmetic mean of these 90 successful tasks, and select the candidate with the lowest average latency as the recommended value for the current test environment. A result is valid only when the log reports `Success count: 100`. If any task fails, identify the cause and rerun the test.

The per-task latency is measured with `std::chrono::steady_clock`. Timing starts immediately before `ExecuteGraphWithStreamAsync` and ends after `aclrtSynchronizeStream` returns successfully, so the measured scope is strictly `ExecuteGraphWithStreamAsync + aclrtSynchronizeStream`. Task queueing, Device memory allocation and release, H2D input copies, and D2H output copies are excluded.

The average latency is calculated as:

```text
Average ExecuteGraphWithStreamAsync latency = total ExecuteGraphWithStreamAsync and aclrtSynchronizeStream latency of the last 90 successful tasks / 90
```

The sample also reports the per-task latency of the last 90 successful tasks. `run` starts at 0 so that the average can be verified and latency variation can be observed:

```text
BENCHMARK ExecuteGraphWithStreamAsync run=<completion index> latency_us=<latency>
```

The sample's existing `Average execution latency` output and timing logic remain unchanged. Timing starts at the beginning of `ModelInference::GraphTask::operator()()` and ends after the D2H output copy and Device memory release. This metric is still calculated as:

```text
Average execution latency = total original execution latency of successful tasks / number of successful tasks
```

Because `Average execution latency` covers the broader interval that includes Device memory allocation and release, H2D input copies, graph execution and stream synchronization, and D2H output copies, retain it only for observing the sample's original metric. Do not use it to select the recommended intra-graph multi-stream parameter.

> [!NOTE]
> This sample uses the DCN v2 recommendation network and random inputs to test graph execution performance. Change only `--multiStreamParallelMode` during tuning, keep all other arguments consistent, and use 10 successfully completed tasks for warmup followed by 90 successful measured tasks for every candidate.

Argument consistency requirements:

- `--batchSize`: Keep the input shape and per-task computational workload consistent.
- `--runs=100`: Submit exactly 100 inference tasks. Use the first 10 successful tasks for warmup and the remaining 90 successful tasks for average-latency measurement.
- `--multiInstanceNum=1`: Exclude the impact of multi-instance parallelism and evaluate only intra-graph multi-stream performance.
- `--enableBatchH2D` and `--aiCoreNum`: Keep other performance optimization settings consistent.

Log field meanings:

- `BENCHMARK`: Fixed marker that allows scripts to locate performance logs.
- `run`: Completion index of a measured task after warmup, not the number of streams. With `--multiInstanceNum=1`, it can be treated as a sequential execution index. In multi-instance scenarios, it indicates task completion order, which is not guaranteed to match submission order.
- `latency_us`: Per-task latency of a measured task in microseconds.
- `Average ExecuteGraphWithStreamAsync latency`: Average `ExecuteGraphWithStreamAsync + aclrtSynchronizeStream` latency of the last 90 successful tasks in milliseconds. This is the primary metric for multi-stream parameter tuning.
- `Average execution latency`: Average latency of all successful recommendation tasks within the sample's original timing scope, in milliseconds.

## Performance Comparison

Throughput (TPS) and latency (ms) performance under different configurations on Ascend 910C platform:

| Configuration | Throughput / Latency |
| :--- | :--- |
| Single instance | 745,55 TPS / 1.471 ms |
| Single instance + Batch H2D | 131,191 TPS / 0.792 ms |
| Multi-instance (4) | 155,104 TPS / 2.089 ms |
| Multi-instance (4) + Core control (16\|16) | 185,415 TPS / 1.797 ms |
| Multi-instance (4) + Core control (16\|16) + Batch H2D | 251,877 TPS / 1.285 ms |
