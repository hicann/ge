# DeclareResourceUsage

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/custom\_op.h\>
- 库文件：libgraph.so

## 功能说明

在图编译阶段申报自定义算子节点的辅流资源使用。GE对图中每个实现该接口的自定义算子节点回调一次，算子上报本节点运行期将请求的资源。

## 函数原型

```c++
virtual graphStatus DeclareResourceUsage(gert::ResourceUsageContext &ctx) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| ctx | 输入 | 资源申报上下文。可获取节点信息并上报资源使用量，仅在回调期间有效。详见[`gert::ResourceUsageContext`](../../gert/ResourceUsageContext/overview.md)。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | graphStatus | `GRAPH_SUCCESS(0)`：申报成功；其他值：申报失败。 |

## 约束说明

无

## 调用示例

```c++
#include <string>

#include "graph/custom_op.h"

class MyEagerOp : public ge::EagerExecuteOp, public ge::ResourceUsageReporter {
 public:
  ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const auto *input = ctx->GetInputTensor(0U);
    if (input == nullptr) {
      return ge::GRAPH_FAILED;
    }
    // 运行期按 key 申请辅流（key 与 DeclareResourceUsage 申报一致）
    gert::rtStream aux_stream =
        ctx->RequestAttachedStream(ge::AscendString(MakeKey(input->GetShape().GetStorageShape().GetDim(0U)).c_str()));
    if (aux_stream == nullptr) {
      return ge::GRAPH_FAILED;
    }
    // ... 在辅流上下发 kernel，并通过 event 与主流同步 ...
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus DeclareResourceUsage(gert::ResourceUsageContext &ctx) override {
    // 编译期按节点静态 shape 分桶申报运行期将使用的全部 key
    const auto *input = ctx.GetInputTensor(0U);
    if (input == nullptr) {
      return ge::GRAPH_FAILED;
    }
    const std::vector<ge::AscendString> keys{
        ge::AscendString(MakeKey(input->GetShape().GetStorageShape().GetDim(0U)).c_str())};
    return ctx.ReportAttachedStream(keys);
  }

 private:
  static std::string MakeKey(int64_t dim0) {
    return "my_op_aux_" + std::to_string(dim0);
  }
};
```
上述示例中，运行期`RequestAttachedStream`与编译期`ReportAttachedStream`共用同一key生成逻辑，保证申报与运行期请求一致；两个输入shape分别为1024和2048的节点，模型统计值为2。
