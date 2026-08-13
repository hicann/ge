# E10403 Invalid\_Argument\_Operator\_Output\_Count

## 错误信息

报错格式如下，占位符%s的含义依次为输出数量、最大数量：

```text
The number of operator outputs %s exceeds the allowed maximum %s.
```

报错示例如下：

```text
The number of operator outputs 5 exceeds the allowed maximum 4.
```

## 可能原因

算子执行配置的输出个数与算子规格描述的不一致。

## 解决方法

请检查numoutputs中的元素个数是否设置正确，可能涉及aclopCompile、aclopExecuteV2和aclopCompileAndExecute接口，接口说明请参见官方文档。
