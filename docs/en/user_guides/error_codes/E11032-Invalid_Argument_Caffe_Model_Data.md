# E11032 Invalid\_Argument\_Caffe\_Model\_Data

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: message type, error field, error cause.

```text
Failed to parse message %s. The error field is %s. Reason: %s.
```

Error example:

```text
Failed to parse message model. The error field is LayerParameter. Reason: Cannot find domi.caffe.LayerParameter in google::protobuf::Descriptor.
```

## Solution

Check whether the Caffe model supports the field.
