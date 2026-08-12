# E10007 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, expected value.

```text
--%s is required. The value must be %s.
```

Error example:

```text
--framework is required. The value must be 0(Caffe) or 1(MindSpore) or 3(TensorFlow) or 5(Onnx).
```

## Solution

Set a valid parameter value.
