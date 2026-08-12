# E10049 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

The following is error format. The placeholders %s indicate the dimension count.

```text
Dimension count %s configured in --input_shape does not match dimension count %s of the node.
```

Error example:

```text
Dimension count 3 configured in --input_shape does not match dimension count 4 of the node.
```

## Solution

Set the dimension count in --input\_shape according to the dimension count of the node.
