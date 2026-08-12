# E10022 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: path, parameter name.

```text
Path %s for parameter --%s does not include the file name.
```

Error example:

```text
Path / for parameter --output does not include the file name.
```

## Solution

Add the file name to the path.
