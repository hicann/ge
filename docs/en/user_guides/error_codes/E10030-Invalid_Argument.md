# E10030 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: attribute name, op name.

```text
There is an invalid value for attribute name %s of Op %s in the file specified by --singleop.
```

Error example:

```text
There is an invalid value for attribute name datatype of Op Add in the file specified by --singleop.
```

## Solution

Check that the Op attribute value is valid in the file.
