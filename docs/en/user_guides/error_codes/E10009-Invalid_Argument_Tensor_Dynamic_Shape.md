# E10009 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

```text
--dynamic_batch_size, --dynamic_image_size, --input_shape_range, and --dynamic_dims are mutually exclusive.
```

## Solution

1. In dynamic shape scenarios, include only one of these options in your command line.
2. In static shape scenarios, remove these options from your command line.
