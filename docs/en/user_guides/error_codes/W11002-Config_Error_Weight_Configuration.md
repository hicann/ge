# W11002 Config\_Error\_Weight\_Configuration

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, op name.

```text
In the compression weight configuration file %s, some nodes do not exist in graph: %s.
```

Error example:

```text
In the compression weight configuration file xxx, some nodes do not exist in graph: graph_name.
```

## Solution

Check whether the weight file matches the model file.
