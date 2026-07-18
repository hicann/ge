# FuseCheckResult

## 产品支持情况

全量芯片支持。

## 功能说明

以冻结数据类形式描述节点集合的融合可行性检查结果。

## 类原型

```python
@dataclass(frozen=True)
class FuseCheckResult:
    ok: bool
    reason: str = ""
```

## 属性说明

| 属性名 | 类型 | 说明 |
| --- | --- | --- |
| ok | bool | 节点集合是否可以融合。 |
| reason | str | 不可融合的原因；`ok=True` 时为空字符串。 |

## 约束说明

该对象不可变，创建后不能修改属性。
