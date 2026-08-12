# E13029 Compilation\_Error\_Load\_Custom\_Fusion\_Pass

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: fusion pass library, error cause.

```text
Failed to load custom fusion pass lib %s. Reason: %s.
```

Error example:

```text
Failed to load custom fusion pass lib /custom_op.so. Reason: undefined symbol: _ZNK7c10_npu9NPUStream6streamEv.
```

## Solution

Analyze the failure reason mentioned above. Below are some typical solutions for common dlopen failures:

1. Verify that the library path is correct and the file exists.
2. Ensure the library and its dependencies have the correct permissions.
3. Check that all dependencies are available using the 'ldd' command.
