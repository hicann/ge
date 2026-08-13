# E10404 Invalid\_Argument\_Operator\_Output\_Buffer

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: index, expected buffer size, current buffer size.

```text
Output indexed %s requires a %s buffer, but %s aligned buffer is allocated.
```

Error example:

```text
Output indexed 1 requires a 200 buffer, but 100 aligned buffer is allocated.
```

## Solution

Check whether the data type, dimensions, and shape are correctly set. For details, see the aclGetTensorDescSize API description in API Reference.
