# E10063 Environment\_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: API name, error cause.

```text
Failed to call the %s API of the system or third-party software. Reason: %s.
```

Error example:

```text
Failed to call the localtime API of the system or third-party software. Reason: [Errno 75] Value too large for defined data type.
```

## Solution

Please adjust code based on the prompts in the Reason.
