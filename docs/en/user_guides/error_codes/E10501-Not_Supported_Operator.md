# E10501 Not\_Supported\_Operator

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, op type.

```text
IR for Op %s with optype %s is not registered.
```

Error example:

```text
IR for Op custom_op with optype CustomOp is not registered.
```

## Possible Cause

1. The environment variable ASCEND\_OPP\_PATH is not configured.
2. IR is not registered.

## Solution

1. Check whether ASCEND\_OPP\_PATH is correctly set.
2. Check whether the operator prototype has been registered. For details, see the operator developer guide.
