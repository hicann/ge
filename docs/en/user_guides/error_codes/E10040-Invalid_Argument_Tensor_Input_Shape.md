# E10040 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

```text
As the --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims argument is included, the corresponding nodes specified in --input_shape must have -1 axes and cannot have '~'.
```

## Solution

1. In static shape scenarios, remove the --dynamic\_batch\_size, --dynamic\_image\_size or --dynamic\_dims option from your command line.
2. In dynamic multi-batch scenarios, set the corresponding axis of the dynamic-shape input in --input\_shape to -1.
3. In dynamic shape scenarios, remove the --dynamic\_batch\_size, --dynamic\_image\_size or --dynamic\_dims option from your command line and set --input\_shape to -1 or n1\~n2.
