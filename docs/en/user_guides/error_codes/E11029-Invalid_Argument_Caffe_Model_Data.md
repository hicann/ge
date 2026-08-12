# E11029 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the op name.

```text
Op %s exists in the model file but is not found in weight file.
```

Error example:

```text
Op add exists in the model file but is not found in weight file.
```

## Solution

Try again with a valid Caffe model or weight file. Ensure that the two files match with each other.
