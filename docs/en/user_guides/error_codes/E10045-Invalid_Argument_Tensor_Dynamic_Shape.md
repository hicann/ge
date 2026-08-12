# E10045 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

```text
The number of -1 axes in the --input_shape argument exceeds the dimension count per profile in --dynamic_dims.
```

## Solution

Ensure that the number of -1 axes in the --input\_shape argument matches the dimension count per profile in --dynamic\_dims.
