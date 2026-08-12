# E10011 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

The following is error format. The placeholder %s indicates the parameter value.

```text
Value %s for parameter --input_shape is invalid. Shape values must be positive integers. The error value in the shape is %s.
```

Error example:

```text
Value [-1,2,3,4] for parameter --input_shape is invalid. Shape values must be positive integers. The error value in the shape is -1.
```

## Solution

1. In static shape scenarios, set the shape values in --input\_shape to positive integers.
2. In dynamic shape scenarios, add the related dynamic-input option in your command line, such as --dynamic\_batch\_size, --dynamic\_image\_size, or --dynamic\_dims.
