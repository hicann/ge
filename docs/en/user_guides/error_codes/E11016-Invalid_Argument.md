# E11016 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, output index, output index maximum, input index, input index maximum.

```text
Failed to add Op %s to NetOutput. Op output index %s is not less than %s. NetOutput input_index %s is not less than %s.
```

Error example:

```text
Failed to add Op add to NetOutput. Op output index 3 is not less than 2. NetOutput input_index 3 is not less than 2.
```

## Solution

Try again with a valid --out\_nodes argument.
