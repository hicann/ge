# E11003 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: input dimension size, input number.

```text
The number of input_dim fields in the model is %s, which is not 4x the input count %s.
```

Error example:

```text
The number of input_dim fields in the model is 4, which is not 4x the input count 8.
```

## Solution

Modify your Caffe model and try again.
