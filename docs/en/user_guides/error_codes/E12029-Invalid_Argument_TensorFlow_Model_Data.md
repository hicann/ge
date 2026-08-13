# E12029 Invalid\_Argument\_TensorFlow\_Model\_Data

## Symptom

```text
Failed to find the subgraph library.
```

## Possible Cause

The model to convert contains function subgraphs, but the graph\_def\_library.pbtxt file is not found.

## Solution

1. To use function subgraphs to convert a TensorFlow model, place the subgraph .proto description file in the same directory as the model file and name it graph\_def\_library.pbtxt.
2. Run the func2graph.py script in the ATC installation directory to save the subgraphs to graph\_def\_library.pbtxt.
