# E10021 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter name, expected value.

```text
Path for parameter --%s is too long. Keep the length within %s.
```

Error example:

```text
Path for parameter --output is too long. Keep the length within 4096.
```

## Solution

The path name exceeds the maximum length. Specify a valid path name.
