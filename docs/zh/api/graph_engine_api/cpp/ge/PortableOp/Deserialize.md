# Deserialize

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/custom\_op.h\>
- 库文件：libgraph.so

## 功能说明

GE在模型加载阶段回调本接口，从随离线模型（OM）保存的buffer中读取序列化的二进制数据并反序列化恢复算子数据。与Serialize配合使用实现算子数据的持久化恢复，支持模型加载时重建算子运行环境。

## 函数原型

```c++
virtual graphStatus Deserialize(const std::vector<uint8_t> &buffer) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| buffer | 输入 | 输入的二进制数据。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

Serialize与Deserialize需配合实现。未实现本接口的自定义算子同样可用于生成离线模型：如果算子实现了AnnotatedArgsOp接口，可通过DeclareLaunchArgs将kernel数据提交给GE随模型保存并在模型加载执行阶段用于launch；否则其数据（如kernel bin）不会随OM保存和恢复，且模型加载阶段不会重新回调自定义算子的Compile接口，需自行管理这些数据的保存和恢复。
