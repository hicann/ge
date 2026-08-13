# E11036 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the top blob.

```text
Data nodes have duplicate top blobs %s.
```

Error example:

```text
Data nodes have duplicate top blobs data1.
```

## Solution

Invalid Caffe model. Make sure the data node has a unique output name.
