# E10031 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

```text
--dynamic_batch_size is included, but none of the nodes specified in --input_shape has a batch size equaling -1.
```

## Possible Cause

As --dynamic\_batch\_size is included, ensure that at least one of the nodes specified in --input\_shape has a batch size equaling -1.

## Solution

1. In static shape scenarios, remove the --dynamic\_batch\_size option from your command line.
2. In dynamic shape scenarios, set the corresponding axis of the dynamic-shape input in --input\_shape to -1.
