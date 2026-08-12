# E10024 Invalid\_Argument

## Symptom

The following is error format. The placeholder %s indicates the file name.

```text
Failed to open file %s specified by --singleop.
```

Error example:

```text
Failed to open file /home/singleop.json specified by --singleop.
```

## Solution

Check the owner group and permission settings and ensure that the user who runs the ATC command has enough permission to open the file.
