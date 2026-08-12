# E16001 Invalid\_Argument\_ONNX\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the node name.

```text
The model has no %s node.
```

Error example:

```text
The model has no input node.
```

## Solution

Check whether the ONNX model contains the input node.
