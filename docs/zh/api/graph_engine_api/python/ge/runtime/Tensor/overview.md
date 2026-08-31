# 简介

`ge.runtime.Tensor`是Python自定义算子执行回调期间使用的张量运行时视图。它提供张量地址、大小、Shape、Format、DataType和存储位置等元数据，不负责分配、释放或拷贝张量数据。

`Tensor`由GE运行时通过`execute`、`declare_launch_args`或执行context提供，用户不能直接构造。由回调返回的`Tensor`及其`Shape`、`StorageShape`、`StorageFormat`和`ExpandDimsType`视图，仅在当前回调有效。

下面的示例在`execute`回调中读取输入Tensor的元数据，并按照输入Tensor的Shape、Format和DataType申请输出Tensor。

```python
from ge.custom_op import get_execute_ctx, register_op_impl
from ge.runtime import Tensor


@register_op_impl(op_type="AddPythonCustomOp")
class AddPythonCustomOp:
    def execute(self, x: Tensor, y: Tensor) -> None:
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
        print(x.addr, x.size, x.storage_shape.dims, x.placement)
        # output 是当前执行回调中的 Tensor，可继续用于执行参数构造。
        _ = output
```
