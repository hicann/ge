# E10027 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: attribute name, input or output, tensor index, op name.

```text
Attribute %s of %s tensor %s for Op %s is invalid when --singleop is specified.
```

Error example:

```text
Attribute datatype of input tensor 1 for Op Add is invalid when --singleop is specified.
```

## Solution

Try again with a valid tensor dtype and format.
