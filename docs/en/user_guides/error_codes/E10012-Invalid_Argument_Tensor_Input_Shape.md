# E10012 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

```text
--dynamic_batch_size is included, but the dimension count of the dynamic-shape input configured in --input_shape is less than 1.
```

## Solution

1. In static shape scenarios, remove the --dynamic\_batch\_size option from your command line.
2. In dynamic shape scenarios, set the corresponding axis of the dynamic-shape input in --input\_shape to -1.
