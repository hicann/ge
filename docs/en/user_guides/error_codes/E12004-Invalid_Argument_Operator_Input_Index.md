# E12004 Invalid\_Argument\_Operator\_Input\_Index

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, input index, input count.

```text
Failed to register the prototype of Op %s. If input index is less than 0, then input index -%s (absolute value) must be less than the input count %s.
```

Error example:

```text
Failed to register the prototype of Op add. If input index is less than 0, then input index -2 (absolute value) must be less than the input count 1.
```

## Solution

When the Const input is converted to an attribute, check whether the input index is correctly set.
