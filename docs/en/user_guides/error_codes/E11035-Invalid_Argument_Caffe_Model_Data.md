# E11035 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, size.

```text
The top size of data node %s is not 1 but %s.
```

Error example:

```text
The top size of data node data1 is not 1 but 2.
```

## Solution

Invalid Caffe model. Change the number of outputs for the data node to 1.
