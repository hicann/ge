# E12013 Invalid\_Argument\_TensorFlow\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the graph name.

```text
Failed to find a subgraph by the name %s.
```

Error example:

```text
Failed to find a subgraph by the name tf_subgraph.
```

## Solution

1. To use function subgraphs to convert a TensorFlow model, place the subgraph .proto description file in the same directory as the model file and name it graph\_def\_library.pbtxt.
2. Run the func2graph.py script in the ATC installation directory to save the subgraphs to graph\_def\_library.pbtxt.
