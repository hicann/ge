# set\_multi\_outputs

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

批量设置指定index的output的tensor。

## 函数原型

```python
set_multi_outputs(self, index, outputs: List[Union[FlowMsg, np.ndarray, fw.FlowMsg]], balance_config: BalanceConfig) -> int
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| index | 输入 | 指定输出的index。 |
| output | 输入 | 指定输出的Msg。 |
| balance_config | 输入 | 输出均衡分发相关配置。 |

## 返回值

- 0：SUCCESS
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无

## 约束说明

无
