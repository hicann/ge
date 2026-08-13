# E11027 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, op type.

```text
Op %s with optype %s in the Caffe model has an input node with shape size 0.
```

Error example:

```text
Op add with optype Add in the Caffe model has an input node with shape size 0.
```

## Solution

Invalid Caffe model. Modify the input shape of the node.
