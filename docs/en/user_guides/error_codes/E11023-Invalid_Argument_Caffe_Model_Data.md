# E11023 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

```text
Weight file contains "layers" structures, which have been deprecated in Caffe and unsupported by ATC."
```

## Solution

Replace layers with layer.
