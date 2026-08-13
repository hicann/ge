# E11012 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: bottom blob, layer, index.

```text
Unknown bottom blob %s at layer %s. The bottom blob is indexed %s.
```

Error example:

```text
Unknown bottom blob data at layer conv1. The bottom blob is indexed 1.
```

## Solution

Modify your Caffe model and try again.
