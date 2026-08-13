# E10402 Invalid\_Argument\_Operator\_Input\_Buffer

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: index, expected buffer size, current buffer size.

```text
Input indexed %s requires a %s buffer, but %s aligned buffer is allocated.
```

Error example:

```text
Input indexed 1 requires a 200 buffer, but 100 aligned buffer is allocated.
```

## Solution

Check whether the data type, dimensions, and shape are correctly set. For details, see the aclGetTensorDescSize API description in API Reference.
