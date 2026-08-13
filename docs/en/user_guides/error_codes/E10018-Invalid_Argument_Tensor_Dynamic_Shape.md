# E10018 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: shape, index.

```text
Value %s for shape %s is invalid. When --dynamic_batch_size is included, only batch size N can be -1 in --input_shape.
```

Error example:

```text
Value -1 for shape 1 is invalid. When --dynamic_batch_size is included, only batch size N can be -1 in --input_shape.
```

## Solution

Try again with a valid --input\_shape argument. Make sure that non-batch size axes are not -1.
