# E10005 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: error stage \(or API name\), parameter value, parameter name.

```text
Value %s for parameter --%s is invalid. The value must be either true or false.
```

Error example:

```text
Value enable for parameter --is_input_adjust_hw_layout is invalid. The value must be either true or false.
```

## Solution

Set a valid parameter value. The parameter value can only be true or false.
