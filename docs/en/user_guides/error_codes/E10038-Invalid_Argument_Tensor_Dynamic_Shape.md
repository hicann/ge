# E10038 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## Symptom

The following is error format. The placeholder %s indicates the dimension size.

```text
Dimension size %s is invalid. The value must be greater than 0.
```

Error example:

```text
Dimension size -1 is invalid. The value must be greater than 0.
```

## Solution

Set the shape values of each profile to positive in --dynamic\_batch\_size, --dynamic\_image\_size, or --dynamic\_dims.
