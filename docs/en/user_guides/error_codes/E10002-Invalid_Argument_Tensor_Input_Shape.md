# E10002 Invalid\_Argument\_Tensor\_Input\_Shape

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: parameter value, error cause, configuration example.

```text
Value %s for parameter --input_shape is invalid. Reason: %s. The value must be formatted as %s.
```

Error example 1:

```text
Value n1~n2,c1,h1,w1 for parameter --input_shape is invalid. Reason: The shape must contain two parts: name and value. The value must be formatted as "input_name1:n1~n2,c1,h1,w1".
```

Error example 2:

```text
Value input_name1:1.1,3,224,224 for parameter --input_shape is invalid. Reason: The float number is unsupported. The value must be formatted as "input_name1:1,3,224,224".
```

## Solution

The valid format is input\_name1:n1,c1,h1,w1;input\_name2:n2,c2,h2,w2. Replace input\_name with node names. Ensure that the shape values are integers.
