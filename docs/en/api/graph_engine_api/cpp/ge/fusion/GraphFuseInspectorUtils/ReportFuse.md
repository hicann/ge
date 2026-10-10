# ReportFuse

## Product Support Status

All chips supported.

## Header/Library

- Header: \#include <ge/fusion/graph\_fuse\_inspector\_utils.h>
- Library: libgraph\_base.so

## Functionality Description

Reports the fusion result. After completing modifications to the graph, the fusion result must be reported to update the graph connection matrix and record diagnostic information.

The internal logic of the interface is briefly as follows:

1. The opdesc of the new node records the pass name.
2. Updates the connection matrix used by CanFuse to detect cycles.
3. Records the match count and effect count, and persists the corresponding information to fusion\_result.json.

## Function Prototype

```c++
static Status ReportFuse(const std::vector<GNode> &nodes_before_fuse, const std::vector<GNode> &nodes_after_fuse, CustomPassContext &ctx)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| nodes_before_fuse | Input | Node list before fusion (all nodes in the list must be connected). |
| nodes_after_fuse | Input | New node list after fusion (all nodes in the list must be connected). An empty nodes_after_fuse indicates the scenario of deletion without adding new nodes. |
| ctx | Input | Pass context, uses ctx.GetPassName() to record the pass name. |

## Return Value

| Parameter | Type | Description |
| --- | --- | --- |
| - | Status | SUCCESS: report succeeded<br>FAILED: report failed |

## Constraints

This API must be called after modifying the graph and before releasing the deleted nodes.

Note: The `SubgraphRewriter::Replace` overload with ctx internally calls this API automatically to report the fusion result. Callers using that overload do not need to (and should not) call this API manually, otherwise effect_time will be double-counted. For manual graph modification scenarios (not going through SubgraphRewriter), this API still needs to be called manually as described above.
