# E10008 Invalid\_Argument

## Symptom

```text
--weight must not be empty when --framework is set to 0 (Caffe).
```

## Solution

1. If the source model framework is Caffe, try again with a valid --weight argument.
2. If the source model framework is not Caffe, try again with a valid --framework argument.
