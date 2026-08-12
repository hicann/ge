# E10019 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

```text
When --dynamic_image_size is included, only the height and width axes can be -1 in --input_shape.
```

## Solution

Try again with a valid --input\_shape argument. Make sure that axes other than height and width are not -1.
