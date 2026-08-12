# E10025 Invalid\_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, error cause.

```text
File %s specified by --singleop is not a valid JSON file. Reason: %s.
```

Error example:

```text
File /home/singleop.json specified by --singleop is not a valid JSON file. Reason: ios_base::clear: unspecified iostream_category error.
```

## Solution

Check that the file is in valid JSON format.
