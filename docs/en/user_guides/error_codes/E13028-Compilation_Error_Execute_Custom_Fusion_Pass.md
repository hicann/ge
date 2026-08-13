# E13028 Compilation\_Error\_Execute\_Custom\_Fusion\_Pass

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: fusion pass name, return code, error cause.

```text
Failed to run custom fusion pass %s. Return code: %s. Reason: %s.
```

The fusion pattern is user-defined, and the pattern name, return code, and error cause are also user-defined. Therefore, the error example should be based on the user-defined scenario.

## Solution

Check the error log for details and verify whether the fusion logic is correct.
