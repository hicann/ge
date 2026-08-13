# E10048 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, error cause, configuration example.

```text
Value %s for parameter --input_shape_range or dynamic_inputs_shape_range is invalid. Reason: %s. The value must be formatted as %s.
```

Error example:

```text
Value abc for parameter --input_shape_range or dynamic_inputs_shape_range is invalid. Reason: The current string cannot be converted to a number. The value must be formatted as 16.
```

## Solution

Please retry with valid parameter value.
