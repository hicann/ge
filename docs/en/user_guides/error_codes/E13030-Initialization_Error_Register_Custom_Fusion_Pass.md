# E13030 Initialization\_Error\_Register\_Custom\_Fusion\_Pass

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: fusion pass name, error cause.

```text
Failed to get custom fusion pass func %s. Reason: %s.
```

Error example:

```text
Failed to get custom fusion pass func CustomOpPass. Reason: Custom stream allocation pass function is required in stage AfterBuiltinFusionPass, but got nullptr.
```

## Solution

Check that custom pass registration is valid.
