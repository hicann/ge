# E10001 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, parameter name, error cause.

```text
Value %s for parameter %s is invalid. Reason: %s
```

Error example 1:

```text
Value 2 for parameter ge.exec.enableDump is invalid. Reason: The value must be 1 or 0.
```

Error example 2:

```text
Value -1 for parameter ge.exec.hostSchedulingMaxThreshold is invalid. Reason: The current value is not within the valid range. The valid range is [0, INT64_MAX].
```

Error example 3:

```text
Value FORMAT_ALL for parameter --input_format is invalid. Reason: The current value is not within the valid range.
```

## Solution

Please input correct parameter value based on the prompts in the Reason, or refer to the official documentation for parameter usage instructions.
