# E10406 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为内存大小、tensor数量：

```text
The number of output buffers is %s, which does not match the number of output tensors %s.
```

报错示例如下：

```text
The number of output buffers is 5, which does not match the number of output tensors 4.
```

## 解决方法

检查算子的outputDesc和outputs中的元素个数是否设置正确，可能涉及aclopExecuteV2和aclopCompileAndExecute接口，接口说明请参见官方文档。
