# E10046 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

```text
The total number of -1 axes in the --input_shape argument is greater than the dimension count per profile in --dynamic_dims.
```

## Solution

Ensure that the total number of -1 axes in the --input\_shape argument is less than the dimension count per profile in --dynamic\_dims.
