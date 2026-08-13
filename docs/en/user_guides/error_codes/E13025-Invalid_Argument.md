# E13025 Invalid\_Argument

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
Input tensor is invalid. Reason: %s.
```

Error example 1:

```text
Input tensor is invalid. Reason: Data indexes must be consecutive and start from 0 when the data shape range is enabled.
```

Error example 2:

```text
Input tensor is invalid. Reason: The number of inputs/outputs provided by the user is inconsistent with that required by the model.
```

## Solution

Please input correct parameter value based on the prompts in the Reason, or refer to the official documentation for parameter usage instructions.
