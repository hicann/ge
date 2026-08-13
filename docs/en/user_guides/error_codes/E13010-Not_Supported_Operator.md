# E13010 Not\_Supported\_Operator

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, op type.

```text
No operator plugin is registered for Op: %s, optype: %s.
```

Error example:

```text
No operator plugin is registered for Op: acustom_op, optype: CustomOp.
```

## Solution

1. If the operator is a custom operator, register related deliverables.
2. If the operator is a built-in operator, install the package that supports this operator version.
