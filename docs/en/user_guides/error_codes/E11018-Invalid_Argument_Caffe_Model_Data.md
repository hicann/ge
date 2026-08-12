# E11018 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The placeholder %s indicates the op name.

```text
Op name %s contains invalid characters.
```

Error example:

```text
Op name add_&* contains invalid characters.
```

## Solution

Allowed characters include: letters, digits, hyphens \(-\), periods \(.\), underscores \(\_\), and slashes \(/\). Modify the Op name and try again.
