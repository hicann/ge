# E16005 Invalid\_Argument\_ONNX\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the domain version count.

```text
The model has %s --domain_version fields, but only one is allowed.
```

Error example:

```text
The model has 2 --domain_version fields, but only one is allowed.
```

## Solution

Invalid ONNX model. Modify the ONNX model. If no domain is specified on the operator node, only one domain can be specified on the model.
