# E12009 Invalid\_Argument\_TensorFlow\_Model\_Data

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: input name, op name.

```text
Input %s for Op %s is not found in graph_def.
```

Error example:

```text
Input data for Op input is not found in graph_def.
```

## Possible Cause

The input name of the node is not found in the graph.

## Solution

Try again with a valid TensorFlow model.
