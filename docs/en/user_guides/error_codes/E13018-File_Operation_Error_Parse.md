# E13018 File\_Operation\_Error\_Parse

## Symptom

The following is error format. The placeholder %s indicates the file name.

```text
Failed to parse file %s through google::protobuf::TextFormat::Parse.
```

Error example:

```text
Failed to parse file /home/file.prototxt through google::protobuf::TextFormat::Parse.
```

## Possible Cause

The file may not be in valid Protobuf format.

## Solution

Check whether the Protobuf file is valid.
