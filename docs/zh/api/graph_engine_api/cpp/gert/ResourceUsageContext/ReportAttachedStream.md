# ReportAttachedStream

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/resource\_usage\_context.h\>
- 库文件：liblowering.so

## 功能说明

上报本节点运行期将通过[`gert::EagerOpExecutionContext::RequestAttachedStream`](../EagerOpExecutionContext/RequestAttachedStream.md)请求的辅流key列表。框架按key在整个模型内去重统计，统计值在[`CompiledGraphSummary::GetStreamNum`](../../ge/CompiledGraphSummary/GetStreamNum.md)的返回值中合并展示。

## 函数原型

```c++
ge::graphStatus ReportAttachedStream(const std::vector<ge::AscendString> &keys)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| keys | 输入 | 辅流key列表。每个key不能为空字符串。在动态图场景中如果无法确认本节点运行期使用的具体key列表，请上报所有可能情况的key列表。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | graphStatus | `GRAPH_SUCCESS(0)`：上报成功；`GRAPH_FAILED`：存在空key或内部错误。|

## 约束说明

无

## 调用示例

```c++
ge::graphStatus DeclareResourceUsage(gert::ResourceUsageContext &ctx) override {
  const auto *input = ctx.GetInputTensor(0U);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto dim = input->GetShape().GetStorageShape().GetDim(0U);
  if (dim <= 0) { /* -1 未知 / -2 维数未知 / 0 空 tensor */
    // 动态图场景，无法确认运行期信息，这里以input的第0维dim为例，上报所有可能的key
    const std::vector<ge::AscendString> keys{
        ge::AscendString("my_op_aux_fixed"),
        ge::AscendString("my_op_aux_1"),
        ge::AscendString("my_op_aux_2"),
        ge::AscendString("my_op_aux_3")};
  } else {
    // 申报运行期将请求的全部 key：固定辅流 + 按 shape 分桶的辅流
    const std::vector<ge::AscendString> keys{
        ge::AscendString("my_op_aux_fixed"),
        ge::AscendString(("my_op_aux_" + std::to_string(dim)).c_str())};
  }
  return ctx.ReportAttachedStream(keys);
}
```
