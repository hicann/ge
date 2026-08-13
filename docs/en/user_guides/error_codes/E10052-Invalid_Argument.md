# E10052 Invalid\_Argument

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
AIPP configuration is invalid. Reason: %s.
```

Error example:

```text
AIPP configuration is invalid. Reason: When --dynamic_image_size is set, crop and padding cannot be set to 'true'.
```

## Solution

Please input correct parameter value based on the prompts in the Reason, or refer to the official documentation for parameter usage instructions.
