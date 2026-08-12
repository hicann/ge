# E13000 File\_Operation\_Error\_Invalid\_Path

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file path, error cause.

```text
Path %s is empty. Reason: %s.
```

Error example:

```text
Path /home/file is empty. Reason: [Error 2] No such file or directory.
```

## Possible Cause

The file does not exist.

## Solution

Try again with a valid directory.
