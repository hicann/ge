# E13031 Invalid\_Argument

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
Output tensor is invalid. Reason: %s.
```

Error example:

```text
Output tensor is invalid. Reason: The output tensor memory size 100 is smaller than the actual size 128 required by tensor.
```

## Solution

Please input correct parameter value based on the prompts in the Reason, or refer to the official documentation for parameter usage instructions.
