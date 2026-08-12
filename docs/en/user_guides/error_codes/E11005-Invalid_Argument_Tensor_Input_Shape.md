# E11005 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

The following is error format. The placeholder %s indicates the input name.

```text
The shape is not defined by using --input_shape for input %s.
```

Error example:

```text
The shape is not defined by using --input_shape for input Input1.
```

## Solution

Modify your Caffe model, or add the input shape to the --input\_shape argument in your atc command line.
