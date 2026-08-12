# E14001 Invalid\_Argument\_Operator\_Compilation\_Parameter

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, op name, op type, error cause.

```text
Argument %s for Op %s with optype %s is invalid. Reason: %s.
```

Error example:

```text
Argument inputs size 2 for Op add with optype Add is invalid. Reason: Input size is not equal to tensor size.
```

## Solution

Check whether the type, input, and output of the operator match the configured parameters.
