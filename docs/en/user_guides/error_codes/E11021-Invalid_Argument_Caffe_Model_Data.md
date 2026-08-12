# E11021 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the model file name.

```text
Model file %s contains "layers" structures, which have been deprecated in Caffe and unsupported by ATC.
```

Error example:

```text
Model file /home/caffe.prototxt contains "layers" structures, which have been deprecated in Caffe and unsupported by ATC.
```

## Solution

Replace layers with layer.
