# E10050 Invalid\_Argument\_Tensor\_Input\_Shape\_Range

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: current dimension size, dimension size minimum, dimension size maximum.

```text
Current dimension size %s is not in the range of %s-%s specified by --input_shape.
```

Error example:

```text
Current dimension size 2 is not in the range of 4-8 specified by --input_shape.
```

## Solution

Set the dimension size according to --input\_shape.
