# E11037 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the op name.

```text
Op %s has zero outputs.
```

Error example:

```text
Op add has zero outputs.
```

## Solution

Nodes in the Caffe model must have at least one output.
