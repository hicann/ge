# E13023 Invalid\_Argument\_OM\_Model\_Size

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: model attribute name, memory size, maximum.

```text
Model %s has a size of %s bytes, which exceeds system limit of %s bytes.
```

Error example:

```text
Model tiling data has a size of 4294967298 bytes, which exceeds system limit of 4294967295 bytes.
```

## Possible Cause

The generated OM model is too large and therefore cannot be dumped to the disk.

## Solution

Reduce the model size.
