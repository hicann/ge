# E10406 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: memory size, tensor number.

```text
The number of output buffers is %s, which does not match the number of output tensors %s.
```

Error example:

```text
The number of output buffers is 5, which does not match the number of output tensors 4.
```

## Solution

Check whether the number of elements in outputDesc and outputs of the operator is correctly set. The aclopExecuteV2 and aclopCompileAndExecute APIs may be involved. For details, see API Reference.
