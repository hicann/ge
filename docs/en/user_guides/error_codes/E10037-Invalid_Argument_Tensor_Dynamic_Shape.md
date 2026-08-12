# E10037 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

The following is error format. The placeholder %s indicates the dimension count.

```text
The profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims have inconsistent dimension counts. A profile has %s dimensions while another has %s dimensions.
```

Error example:

```text
The profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims have inconsistent dimension counts. A profile has 4 dimensions while another has 8 dimensions.
```

## Solution

Ensure that the profiles configured in --dynamic\_batch\_size, --dynamic\_image\_size, or --dynamic\_dims have the same dimension count.
