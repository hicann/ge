# E10006 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: error stage \(or API name\), parameter value, parameter name.

```text
Value %s for parameter --%s is invalid. The value must be either 1 or 0.
```

Error example:

```text
Value 2 for parameter --sparsity is invalid. The value must be either 1 or 0.
```

## Solution

Set a valid parameter value. The parameter value can only be 1 or 0.
