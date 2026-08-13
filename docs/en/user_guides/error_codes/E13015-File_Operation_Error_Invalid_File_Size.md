# E13015 File\_Operation\_Error\_Invalid\_File\_Size

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, file size, maximum.

```text
File %s has a size of %s, which is out of valid range (0, %s].
```

Error example:

```text
File /home/file.txt has a size of 2147483649, which is out of valid range (0, 2147483647].
```

## Solution

Please provide invalid file based on the prompts in the error message.
