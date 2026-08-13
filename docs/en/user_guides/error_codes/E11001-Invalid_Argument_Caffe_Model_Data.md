# E11001 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

```text
input_dim and input_shape are mutually exclusive in NetParameter for Caffe model conversion.
```

## Solution

Remove either --input\_dim or --input\_shape from your atc command line.
