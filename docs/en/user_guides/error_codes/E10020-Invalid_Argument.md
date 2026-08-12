# E10020 Invalid\_Argument

## Symptom

The following is error format. The placeholder %s indicates the parameter value.

```text
Value %s for parameter --dynamic_image_size is invalid.
```

Error example:

```text
Value 1,2,3;4,5,6 for parameter --dynamic_image_size is invalid.
```

## Solution

The value must be formatted as "imagesize1\_height,imagesize1\_width;imagesize2\_height,imagesize2\_width". Make sure that each profile has two dimensions.
