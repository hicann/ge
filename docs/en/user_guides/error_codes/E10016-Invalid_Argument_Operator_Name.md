# E10016 Invalid\_Argument\_Operator\_Name

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, parameter name.

```text
Op name %s specified in --%s is not found in the model. Confirm whether this node name exists, or whether the node is not split with the specified delimiter ';'.
```

Error example:

```text
Op name invalid_op specified in --input_shape is not found in the model. Confirm whether this node name exists, or whether the node is not split with the specified delimiter ';'.
```

## Solution

Specify the name of an existing node in the graph.
