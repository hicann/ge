# E10062 Invalid\_Argument\_API\_Call\_Sequence

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: API name, error cause.

```text
Failed to %s. Reason: %s.
```

Error example:

```text
Failed to call RunGraphAsync. Reason: Graph <graph_id> has been compiled by calling CompileGraph. RunGraphAsync and CompileGraph are mutually exclusive and cannot be used together.
```

## Solution

Please adjust code based on the prompts in the Reason.
