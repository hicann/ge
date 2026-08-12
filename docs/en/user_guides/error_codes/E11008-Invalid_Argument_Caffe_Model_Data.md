# E11008 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

```text
Op type DetectionOutput is unsupported.
```

## Solution

Modify your Caffe model and replace DetectionOutput operators with FSRDetectionOutput or SSDDetectionOutput.
