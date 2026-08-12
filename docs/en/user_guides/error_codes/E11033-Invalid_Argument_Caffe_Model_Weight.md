# E11033 Invalid\_Argument\_Caffe\_Model\_Weight

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: blob name, blob size, error cause.

```text
Failed to convert the weight file. Blob %s of size %s is invalid. Reason: %s.
```

Error example:

```text
Failed to convert the weight file. Blob data of size 100 is invalid. Reason: It does not match shape size 128.
```

## Possible Cause

The blob size of the node in the Caffe weight file does not match the number of elements calculated based on its shape.

## Solution

Try again with a valid Caffe model or weight file. Ensure that the two files match with each other.
