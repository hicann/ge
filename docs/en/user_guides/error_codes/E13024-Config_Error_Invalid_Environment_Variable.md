# E13024 Config\_Error\_Invalid\_Environment\_Variable

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: environment value, environment name, error cause.

```text
Value %s for environment variable %s is invalid. Reason: %s.
```

Error example:

```text
Value 1 for environment variable VIRTUAL_TYPE is invalid. Reason: L1_fusion is not supported in the Ascend virtual instance scenario.
```

## Solution

Reset the environment variable by referring to the setup guide.
