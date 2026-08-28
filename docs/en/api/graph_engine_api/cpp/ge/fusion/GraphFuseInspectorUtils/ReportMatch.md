# ReportMatch

## Product Support Status

All chips supported.

## Header/Library

- Header: \#include <ge/fusion/graph\_fuse\_inspector\_utils.h>
- Library: libgraph\_base.so

## Functionality Description

Reports a structure match. Called when a target structure is found during graph traversal, counted regardless of whether fusion conditions pass. Internally increments match\_time automatically, does not change effect\_time, corresponding information is persisted to fusion\_result.json.

Used together with [ReportFuse](ReportFuse.md) to calculate structure match hit rate: match\_time is the total number of structure matches (superset), effect\_time is the number of fusions actually applied (subset), match\_time - effect\_time reflects the number of fusions abandoned due to condition filtering.

## Function Prototype

```c++
static Status ReportMatch(const std::vector<GNode> &matched_nodes, CustomPassContext &ctx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| matched_nodes | Input | List of nodes hit by structure matching (all nodes in the list must be connected). |
| ctx | Input | Pass context, uses ctx.GetPassName() to record pass name. |

## Return Value

| Parameter | Type | Description |
| --- | --- | --- |
| - | Status | SUCCESS: report succeeded<br>FAILED: report failed |

## Constraints

This API should be called after discovering the target structure and before [CanFuse](CanFuse.md).
