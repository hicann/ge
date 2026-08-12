# E14002 Invalid\_Argument\_Tensor\_Attribute

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: attribute name, error cause.

```text
In the current process, the attribute of %s must be obtained successfully. Reason: %s.
```

Error example:

```text
In the current process, the attribute of storage_format must be obtained successfully. Reason: Failed to get storage shape from node Failed to get storage shape from node add.
```

## Solution

The attribute in the error message must be set for the operator.
