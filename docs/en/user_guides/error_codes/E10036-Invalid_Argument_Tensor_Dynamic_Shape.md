# E10036 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: shape size, shape size maximum.

```text
--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has %s profiles, which is greater than the maximum %s.
```

Error example:

```text
--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has 1024 profiles, which is greater than the maximum 100.
```

## Solution

Ensure that the number of profiles configured in --dynamic\_batch\_size, --dynamic\_image\_size, or --dynamic\_dims is at most the maximum.
