# E10003 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, parameter name, error cause.

```text
Value %s for parameter --%s is invalid. Reason: %s
```

Error example:

```text
Value 1.1,2,4,8 for parameter --dynamic_batch_size is invalid. Reason: It can only contain digits and ",".
```

## Solution

Please input correct parameter value based on the prompts in the Reason, or refer to the official documentation for parameter usage instructions.
