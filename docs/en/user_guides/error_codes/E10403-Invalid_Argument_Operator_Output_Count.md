# E10403 Invalid\_Argument\_Operator\_Output\_Count

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: output number, maximum number.

```text
The number of operator outputs %s exceeds the allowed maximum %s.
```

Error example:

```text
The number of operator outputs 5 exceeds the allowed maximum 4.
```

## Possible Cause

The number of outputs configured for operator execution does not match that described in the operator specifications.

## Solution

Check whether the number of elements in numoutputs is correctly set. The aclopCompile, aclopExecuteV2, and aclopCompileAndExecute APIs may be involved. For details, see API Reference.
