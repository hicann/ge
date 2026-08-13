# E10032 File\_Operation\_Error\_Parse

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, error cause.

```text
Failed to parse JSON file %s. Reason: %s.
```

Error example:

```text
Failed to parse JSON file /home/singleop.json. Reason: [json.exception.out_of_range.401] array index 5 is out of range.
```

## Solution

Check whether the json file is valid.
