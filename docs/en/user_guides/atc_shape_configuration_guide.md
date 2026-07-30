# ATC Model Conversion Practice Guide: Static Shape, Dynamic Multi-Gear, and Dynamic Shape

## 1 Introduction

This document is intended for application developers and focuses on two core questions:

* **Will the input size change**
* **Can changes be enumerated in advance**

Based on these two dimensions, practical solutions for using **ATC** to convert models in Ascend inference scenarios are provided. This document does not distinguish between frontend frameworks and applies to all model formats supported by ATC (such as ONNX, TensorFlow PB, Caffe, etc.).

In Ascend inference scenarios, the choice of shape directly affects the compiler optimization level, runtime scheduling method, and final performance stability. Properly choosing between static shape, dynamic multi-gear, or dynamic shape, combined with ATC's capability characteristics, is key to achieving stable throughput and low latency.

This document assumes that readers already understand the complete process of model conversion via ATC and model loading/inference using **aclmdl** interfaces.

---

## 2 Overall Flow of Model Conversion and Execution

Before diving into specific strategies, let's unify the basic concepts of ATC and model execution phases from an overall flow perspective.

Users convert models into `.om` (Offline Model) files via **ATC** command, then load and execute these models via **aclmdl** series interfaces. From GE (Graph Engine) perspective, these two phases are called **compile** and **execute** respectively.

* **Compile Phase**
  GE reads the model file specified in ATC (such as ONNX or PB), analyzes and optimizes the computation graph, and generates a binary model file (`.om`) that can be executed on NPU.

* **Execute Phase**
  GE loads the `.om` file via aclmdl interfaces, deploys it to NPU device, and executes subsequent inference tasks.

It should be clarified that GE adopts a **clear separation of compile-time and runtime responsibilities** model:

* The compile phase takes longer but usually needs to be executed only once to generate `.om`;
* The execute phase no longer performs structural graph optimization, inference overhead is small, and `.om` can be repeatedly executed after loading.

This characteristic determines the **importance of shape information at compile time**.

## 3 Static Shape, Dynamic Shape, and Performance Characteristics

### Static Shape

**Static shape** means that during multiple executions of the model, all tensor (input, output, and intermediate tensors) dimensions are completely fixed, and no dimension is allowed to change.

In this mode, the compile phase can perform the most comprehensive optimizations and enable **sink scheduling** during execution. The specific mechanism of sink scheduling can be found in the official documentation:
[https://www.hiascend.com/developer/techArticles/20240715-1](https://www.hiascend.com/developer/techArticles/20240715-1)

In engineering practice, static shape usually achieves the best inference performance and stability.

---

### Dynamic Shape

**Dynamic shape** means that during multiple executions of the model, the dimensions of input or intermediate tensors may change.

Its advantage is flexibility, but the cost is also obvious:

* Significantly fewer optimizations available at compile time;
* Cannot enable sink scheduling;
* Inference performance and latency stability are usually poor.

Therefore, in performance-sensitive inference scenarios, completely dynamic shape should be avoided.

---

### Dynamic Multi-Gear (Recommended Balanced Solution)

Considering the significant performance advantage brought by static shape, ATC provides **dynamic multi-gear** capability to handle **scenarios where shape changes are limited and enumerable**.

The essence of dynamic multi-gear is:
During model conversion phase, **specify multiple fixed static shape gears at once**. At runtime, select the matching gear to execute based on actual input, but each gear is treated as static shape during compile phase.

For example, if only the batch dimension of the model is variable and may take the following values:

* `[1, 3, 224, 224]`
* `[8, 3, 224, 224]`
* `[16, 3, 224, 224]`

Then these three batch sizes can be passed to ATC simultaneously as three gears.

After enabling dynamic multi-gear:

* The model still appears as "dynamic" at execution level;
* The compiler can perform static shape optimization for each gear;
* Inference performance usually matches that of single static shape.

Note that while dynamic multi-gear brings performance benefits, it also introduces additional costs:

* **Model memory occupation is based on the largest gear**
  Even when executing the smallest gear, the overall model memory occupation is equivalent to the largest gear. For example, if the largest batch gear is 1024, even when executing batch=1, memory occupation is still calculated as 1024.
* **Compile time increases linearly with the number of gears**
  Generally, the compile time for N gears is approximately N times that of single static shape.

## 4 Overview of Shape-Related Parameter Configuration in ATC

This chapter explains from the **ATC parameter configuration perspective** how the three strategies of static shape, completely dynamic shape, and dynamic multi-gear are expressed in ATC.

### Parameter Configuration for Static Shape

Under the static shape strategy, the model needs to **completely determine all input tensor dimensions** during compile phase. When converting with ATC, users need to explicitly specify a fixed shape for each input.

For example:

```shell
--input_shape="input_0_0:16,32,208,208;input_1_0:16,64,208,208"
```

In the above configuration, all input dimensions are completely determined at compile time, and the model is compiled in static shape mode.

Configuration items involved:

* **`--input_shape`**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0016.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0016.html)

---

### Parameter Configuration for Dynamic Shape

Dynamic shape means that at compile time **the dimensions of input or intermediate tensors cannot be completely determined**, and the shape needs to be determined at runtime.

In ATC, dynamic shape is still configured through `--input_shape`, using `-1` to indicate that the corresponding dimension is dynamic.

For example:

```shell
--input_shape="input_0_0:-1,32,208,208;input_1_0:-1,64,208,208"
```

The above configuration indicates that the batch dimension cannot be determined at compile time, and the model will be compiled in dynamic shape mode.

**Dynamic shape and static shape use the same configuration items**, the difference is only reflected in whether there are undetermined dimensions.

---

### Parameter Configuration for Dynamic Multi-Gear

Dynamic multi-gear is used to handle **scenarios where shape changes are limited and can be enumerated in advance**.

In ATC, the core of dynamic multi-gear configuration is not "making the model support runtime dynamic shape", but rather:

> **Enumerate all possible fixed shape gears at once during compile time.**

Each gear is treated as static shape during the compile phase, and at runtime the corresponding gear is selected for execution based on the input shape.

Dynamic multi-gear configuration typically has the following characteristics:

* All shape variations have been enumerated at compile time;
* The compiled product contains execution paths for multiple static shapes;
* Input shapes that do not match any gear will cause execution failure.

Semantically, dynamic multi-gear still belongs to the category of **"compile-time shape determination"**, it is just that there is more than one determined shape.

#### Example: Dynamic Batch Multi-Gear Configuration

If only the batch dimension of the model is variable, it can be configured as follows:

```shell
$ atc \
  --input_shape="input_0_0:-1,32,208,208;input_1_0:-1,64,208,208" \
  --dynamic_batch_size="1,8,16"
```

Where:

* `-1` in `--input_shape` indicates that the batch dimension is dynamic;
* `--dynamic_batch_size` enumerates all possible batch gears at runtime.

This configuration means that during model execution, only the following input shapes are allowed:

* `[1, 3, 224, 224]`
* `[8, 3, 224, 224]`
* `[16, 3, 224, 224]`

#### Dynamic Multi-Gear Related Configuration Items

* **Dynamic batch (`--dynamic_batch_size`)**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0018.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0018.html)

* **Dynamic image size (`--dynamic_image_size`)**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0019.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0019.html)

* **Arbitrary dimension dynamic (`--dynamic_dims`)**
  [https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0020.html](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/atctool/atlasatcparam_16_0020.html)

Among them, the configuration capability of `--dynamic_dims` can cover both dynamic batch and dynamic image size, but the configuration is relatively more complex.
