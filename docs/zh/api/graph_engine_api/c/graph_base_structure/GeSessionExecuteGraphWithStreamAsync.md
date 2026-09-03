# GeSessionExecuteGraphWithStreamAsync

## 产品支持情况

请参见[Session接口产品支持情况](../../cpp/ge/Session/overview.md)。

## 头文件/库文件

- 头文件：\#include <ge/ge\_api.h\>
- 库文件：libge\_runner.so

## 功能说明

使用指定的Session实例，异步运行指定ID对应的Graph图，输出执行结果。

此接口与[ExecuteGraphWithStreamAsync](../../cpp/ge/Session/ExecuteGraphWithStreamAsync.md)均为运行指定ID对应的图，并输出结果，该接口更多是为了前向兼容上层开源框架的图模式场景；如果用户使用的是8.0.RC3版本及之后的CANN包，则更推荐使用ExecuteGraphWithStreamAsync。

- 该接口执行前需要完成[CompileGraph](../../cpp/ge/Session/CompileGraph.md)及[GeSessionLoadGraph](GeSessionLoadGraph.md)（异步运行Graph场景）流程。
- inputs和outputs数据类型为gert::Tensor。

## 函数原型

```c
ge::Status GeSessionExecuteGraphWithStreamAsync(ge::Session &session, uint32_t graph_id, void *stream, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| session | 输入 | 加载图的Session实例。 |
| graph_id | 输入 | 子图对应的ID。 |
| stream | 输入 | 指定图在哪个Stream上运行。 |
| inputs | 输入 | 当前子图对应的输入数据，为Device上的内存空间。 |
| outputs | 输出 | 当前子图对应的输出数据，为Device上的内存空间。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | Status | SUCCESS：异步运行图成功。<br>FAILED：异步运行图失败。 |

## 约束说明

- 调用该接口前，需要确定好Device上分配的内存大小。
- 调用该接口前，需要完成[CompileGraph](../../cpp/ge/Session/CompileGraph.md)及[GeSessionLoadGraph](GeSessionLoadGraph.md)流程。
- 调用该接口前，需要通过acl提供的**aclrtCreateStream**接口创建Stream。
- 得到输出运行结果前，需要通过**aclrtSynchronizeStream**接口保证Stream上的任务已经执行完。

接口详细说明请参见《[Runtime运行时API](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/README.md)》中的“Stream管理”。
