# E11009 Not\_Supported\_Operator

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, op type.

```text
No Caffe parser is registered for Op %s with Op type %s.
```

Error example:

```text
No Caffe parser is registered for Op custom_op with Op type CustomOp.
```

## Solution

Check whether the Caffe plugin of the operator has been registered.
