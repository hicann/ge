# E13005 File\_Operation\_Error\_Parse

## Symptom

The following is error format. The placeholder %s indicates the file name.

```text
Failed to parse file %s.
```

Error example:

```text
Failed to parse file /home/caffe.prototxt.
```

## Solution

Check that a matched Protobuf version is installed and try again with a valid file. For details, see section "--framework" in ATC Instructions.
