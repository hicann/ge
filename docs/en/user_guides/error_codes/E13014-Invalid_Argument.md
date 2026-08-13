# E13014 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, parameter name, op name, error cause.

```text
Value %s of parameter %s for Op %s is invalid. Reason: %s.
```

Error example 1:

```text
Value shape range size for Op add is invalid. Reason: The tensor has dynamic dimensions, but the shape range is not empty.
```

Error example 2:

```text
Value format for Op add is invalid. Reason: The format must be NCHW or NHWC in the dynamic AIPP scenario.
```

## Solution

Please input correct parameter value based on the prompts in the Reason, or refer to the official documentation for parameter usage instructions.
