# E10034 Invalid\_Argument

## Symptom

The following is error format. The placeholder %s indicates the op name.

```text
Nodes (for example, %s) connected to AIPP must not be of type FP16.
```

Error example:

```text
Nodes (for example, Add) connected to AIPP must not be of type FP16.
```

## Solution

1. To enable AIPP, remove the nodes connected to AIPP from the --input\_fp16\_nodes argument.
2. If AIPP is not required, remove the --insert\_op\_conf option from your ATC command line.
