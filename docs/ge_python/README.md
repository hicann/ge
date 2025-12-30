# GE-PY Python 模块文档

## 概述

GE-PY 是 GraphEngine 的 Python 接口模块，提供了 Pythonic 的图相关接口。为用户提供了便捷的图构建和操作，编译执行等功能。该模块对外头文件位于 `api/python/ge/ge/` 目录下。

GE-PY 模块包含以下核心组件：

- **graph 模块** - 图基础操作模块，提供 Graph、Node、Tensor 等核心类
- **ge_global 模块** - GE 全局初始化和析构接口
- **session 模块** - 图编译执行接口
- **es 模块** - Eager-Style 图构建接口，提供函数式风格的图构建方式

## 文档导航

### 设计文档

- **[GE-PY 模块类关系文档](design/ge_python.md)** - Graph、Node、Tensor、Session 等基础模块的详细说明
  - Graph 类：图操作的主要接口
  - Node 类：图节点操作接口
  - Tensor 类：张量数据类
  - GeApi 类：GE 初始化和析构
  - Session 类：图编译执行接口

- **[ES-PY 模块文档](../es/api/es_python.md)** - Eager-Style 图构建模块的详细说明
  - GraphBuilder 类：Eager-Style 图构建器
  - TensorHolder 类：张量持有者

## 模块关系

- **graph 模块** - 提供图的基础操作能力，是其他模块的基础
- **es 模块** - 提供函数式图构建方式，最终构建出 graph 模块的 Graph 对象
- **session 模块** - 使用 graph 模块构建的图进行编译和执行
- **ge_global 模块** - 提供全局初始化和资源管理

## 使用示例

### 基础图操作示例

参考 [使用es的python api构图sample](../../../examples/es/transformer/python/src/make_transformer_graph.py)

### 更多示例

更多 Python 用例请参考 [examples/es](../../../examples/es) 目录下的各个子目录：



