# Default Data Format Analysis

## Experiment Method

Generate the ONNX model using `gen_onnx.py` with the following structure:

```
input1(NCHW) → Transpose(perm=[0,2,3,1]) → Add → output(NHWC)
input2(NCHW) → Transpose(perm=[0,2,3,1]) /
```

Compile via ATC with `DUMP_GE_GRAPH=2` enabled, and analyze `ge_proto_PreRunBegin.txt`.

## Results

### Data Node

| Property | Value |
|----------|-------|
| type | `Data` |
| input layout | `NCHW` |
| output layout | `NCHW` |
| origin_format | `NCHW` |

### Transpose Node

| Property | Value |
|----------|-------|
| type | `Transpose` |
| Input list | `input1:0`, `Const_2:0` |
| input[0] layout | `ND` (from Data) |
| input[1] layout | `ND` (perm constant) |
| output layout | `ND` |

### Add Node

| Property | Value |
|----------|-------|
| type | `Add` |
| input layout | `ND` |

## Conclusion

1. The default format of the Data node is **NCHW** (inherited from the ONNX model definition).
2. The input/output TensorDesc of Transpose shows `ND` (format undetermined);
   the format information is actually carried by the Data node's TensorDesc.
3. The perm values of Transpose are stored in the `value` attribute of the Const node:
   - dtype: DT_INT64
   - shape: [4]
   - values: [0, 2, 3, 1]
4. When registered at the `kBeforeInferShape` stage and the Transpose is removed,
   InferFormat correctly handles format propagation.
