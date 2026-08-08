# Data 默认格式分析记录

## 实验方法

使用 `gen_onnx.py` 生成 ONNX 模型，结构如下：

```
input1(NCHW) → Transpose(perm=[0,2,3,1]) → Add → output(NHWC)
input2(NCHW) → Transpose(perm=[0,2,3,1]) /
```

通过 ATC 编译并开启 `DUMP_GE_GRAPH=2`，分析 `ge_proto_PreRunBegin.txt`。

## 结果

### Data 节点

| 属性 | 值 |
|------|-----|
| type | `Data` |
| input layout | `NCHW` |
| output layout | `NCHW` |
| origin_format | `NCHW` |

### Transpose 节点

| 属性 | 值 |
|------|-----|
| type | `Transpose` |
| 输入列表 | `input1:0`, `Const_2:0` |
| input[0] layout | `ND` (来自 Data) |
| input[1] layout | `ND` (perm 常量) |
| output layout | `ND` |

### Add 节点

| 属性 | 值 |
|------|-----|
| type | `Add` |
| input layout | `ND` |

## 结论

1. Data 节点的默认格式是 **NCHW**（从 ONNX 模型定义继承）。
2. Transpose 的输入/输出 TensorDesc 显示为 `ND`（格式未确定），
   格式信息实际上由 Data 节点的 TensorDesc 携带。
3. Transpose 的 perm 值存储在 Const 节点的 `value` 属性中：
   - dtype: DT_INT64
   - shape: [4]
   - values: [0, 2, 3, 1]
4. 注册在 `kBeforeInferShape` 阶段，删除 Transpose 后，
   InferFormat 能正确处理格式传播。
