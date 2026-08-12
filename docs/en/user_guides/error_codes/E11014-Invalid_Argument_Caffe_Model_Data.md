# E11014 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the layer.

```text
Failed to find the top blob for layer %s.
```

Error example:

```text
Failed to find the top blob for layer conv1.
```

## Possible Cause

The top blob has no corresponding node in the source Caffe model.

## Solution

Modify your Caffe model and try again.
