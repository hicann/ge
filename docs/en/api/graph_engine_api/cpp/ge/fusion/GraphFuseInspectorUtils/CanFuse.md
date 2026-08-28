# CanFuse

## Product Support Status

All chips supported.

## Header/Library

- Header: \#include <ge/fusion/graph\_fuse\_inspector\_utils.h>
- Library: libgraph\_base.so

## Functionality Description

Determines whether the set of nodes to be fused can be fused. The judgment conditions are:

1. The stream label on a node expresses the shunting strategy on the graph. When the incoming nodes have inconsistent stream labels, the user intent cannot be determined, that is, it cannot be determined whose label the new node should inherit. Therefore, when the stream labels of the incoming node list are inconsistent, it is judged as not fusible.

2. If fusing the incoming node list into a single node creates a cycle, it is judged as not fusible. Note that other scenarios such as replacing incoming nodes with multiple nodes are not covered by the cycle detection here.

## Function Prototype

```c++
static bool CanFuse(const std::vector<GNode> &nodes_before_fuse, AscendString &failed_reason)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| nodes_before_fuse | Input | Node list before fusion (all nodes in the list must be connected). |
| failed_reason | Output | The reason why fusion is not supported. |

## Return Value

| Parameter | Type | Description |
| --- | --- | --- |
| - | bool | - true: can be fused.<br>  - false: cannot be fused (failed_reason is filled with the specific reason). |

The above reason needs to be printed by the user. The following code can be added below the interface:

```c++
// Perform CanFuse check
    AscendString failed_reason;
    bool can_fuse = fusion::GraphFuseInspectorUtils::CanFuse(nodes_before_fuse, failed_reason);
    if (!can_fuse) {
        std::cerr << "[FuseNodes] CanFuse check failed: " << failed_reason.GetString() << std::endl;
        return false;
    }
```

## Constraints

None
