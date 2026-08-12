# E10035 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: shape size, shape size minimum.

```text
--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has %s profiles, which is less than the minimum %s.
```

Error example:

```text
--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has 1 profiles, which is less than the minimum 2.
```

## Solution

Ensure that the number of profiles configured in --dynamic\_batch\_size, --dynamic\_image\_size, or --dynamic\_dims is at least the minimum.
