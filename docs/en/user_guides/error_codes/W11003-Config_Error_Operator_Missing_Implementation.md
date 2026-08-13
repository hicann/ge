# W11003 Config\_Error\_Operator\_Missing\_Implementation

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: op name, implementation name.

```text
Operator %s lacks required %s implementation.
```

Error example:

```text
Operator CustomAdd lacks required InferShape implementation.
```

## Possible Cause

Incomplete operator implementation.

## Solution

Ensure that all required operator implementations\(e.g., tiling\) are provided. See the operator developer guide for details.
