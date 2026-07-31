# Format Modeling and API Semantic Analysis in GE

## 1. Why Format Becomes a Performance Problem

In deep learning models, when users construct computational graphs, they typically focus on **computational semantics itself**: tensor dimensions, mathematical meaning of operators, and dependencies between operators etc.

At this level, how data "looks like" is often considered obvious, and doesn't need extra attention. However, when models enter actual execution phase, this "taken for granted" assumption often no longer holds.

### 1.1 Gap Between User Semantics and Actual Execution

From user perspective, a tensor is just a set of ordered multidimensional data; from execution perspective, this data needs to be stored in some **specific memory layout** to be efficiently accessed by hardware. Format in GE represents exactly this memory layout. In industry graph compilers, this concept is usually also called Layout. For example, NCHW and NHWC in GE belong to two different formats.

Different computational operators often have **different preferences** for formats, for example:

* For Conv2D operator, preferred image input format is NC1HWC0, preferred filter input format is FZ
- For MatMul operator, preferred weight input format is NZ

These differences don't come from the algorithm itself, but originate from **implementation characteristics of underlying hardware architecture**.

### 1.2 Cost of Data Rearrangement Is Not "Free"

When operators have preferred formats, GE usually needs to insert additional **data rearrangement** operations to convert inputs to formats more suitable for that operator's computation. However, this kind of data rearrangement is not cheap in performance:

- Will introduce additional computation overhead and memory bandwidth consumption
- May be triggered multiple times in complex models

More importantly, these data rearrangements often **won't appear in user explicitly constructed computational graph**, yet will directly affect model's actual execution efficiency.

### 1.3 This Is a Systemic Problem

An intuitive idea is:

> Since some operators are more sensitive to data layout, let each operator handle its input/output formats.

But in engineering practice, this is not the optimal solution. Taking a multi-layer convolutional network as example:

```mermaid
graph LR
INPUT -- NCHW --> Conv2D_1
Conv2D_1 -- NCHW --> ReLU_1
ReLU_1 -- NCHW --> Conv2D_2
Conv2D_2 -- NCHW --> ReLU_2
```

If operators "act independently", Conv2D will internally convert input from NCHW to NC1HWC0, after computation convert output back to NCHW. At this point, actual execution process will evolve into:

```mermaid
graph LR
INPUT -- NCHW --> TransData_1((TransData_1))
TransData_1 -- NC1HWC0 --> Conv2D_1
Conv2D_1 -- NC1HWC0 --> TransData_2((TransData_2))
TransData_2 -- NCHW --> ReLU_1
ReLU_1 -- NCHW --> TransData_3((TransData_3))
TransData_3 -- NC1HWC0 --> Conv2D_2
Conv2D_2 -- NC1HWC0 --> TransData_4((TransData_4))
TransData_4 -- NCHW --> ReLU_2
```

For single Conv2D operator, using preferred format through data rearrangement can obtain better computational performance; but from entire network perspective, TransData repeatedly appears at each layer, significantly increasing overall execution overhead.

Therefore, format-related problems are essentially a **network-wide systemic optimization problem**: under premise of ensuring computational semantic correctness, need to select appropriate data layout for different operators, and minimize unnecessary data conversions.

Exactly in such context, GE introduced a **unified format modeling and optimization mechanism**, to systematically handle the gap between user semantics and actual execution.

## 2. Origin and Storage: Two Representation Systems in GE

To solve the data layout and performance problems mentioned in previous chapter, GE internally made explicit distinction for tensor representation, introduced two interconnected but differently responsible representation systems: **Origin** and **Storage**.

These two representations respectively describe user's original semantics and data form during operator actual execution, are the basis for GE to perform format modeling and optimization.

### 2.1 Origin: Expression and Propagation of User Semantics

Origin is used to describe **original semantics expressed by user when constructing computational graph**, including:

- OriginFormat: Tensor's format description at semantic level, e.g. NCHW
- OriginShape: Tensor's dimension information at semantic level, e.g. [8, 3, 224, 224]

Origin's source is usually frontend framework or user explicitly given model definition, its core characteristics are:

- **Directly reflects user intent**
- **Does not contain any assumptions targeting specific hardware or implementation**
- **Does not adjust for performance goals**

When GE receives a computational graph, Origin is usually **explicitly given** by graph inputs and some key operators' attributes (e.g. Conv2D marks its input/output formats via attributes). GE will propagate Origin throughout computational graph as much as possible, its purpose is not for performance optimization, but to **always retain complete understanding of user's original computational semantics** throughout compilation process.

This propagation mechanism provides clear semantic boundary for subsequent optimization, making any form of format adjustment or execution optimization must be built on premise of **not destroying Origin semantics**.

### 2.2 Storage: Representation of Actual Computation and Storage

Unlike Origin, Storage is used to describe representation form adopted by tensor in **actual execution phase**, including:

- StorageFormat: Tensor's specific layout method in memory, e.g. NCHWC0, splits C axis into C0, C1

- StorageShape: Tensor's actual form in memory, e.g. NCHW format with Shape [8, 3, 224, 224], after converting to NC1HWC0, Shape is [8, 1, 224, 224, 16]

Storage is not specified by user, but derived by GE during compilation process based on multiple factors, e.g.:

- Operator capabilities and limitations
- Different operators' format affinity
- Network-wide data flow relationships- Different operators' affinity to formats
- Whole graph scope data flow relationships

Since not all operators support all formats, the Storage derivation process is naturally constrained. For example, some formats may only be valid for specific operators or specific inputs of operators (such as weights).

Therefore, Storage represents an **execution-oriented engineering choice**, its goal is to minimize inserting format conversions (TransData) while satisfying operator capability constraints, to obtain overall better execution efficiency.

### 2.3 Summary: Origin and Storage Division of Labor Cooperation Relationship

In GE, the Origin and Storage division of labor cooperation relationship can be summarized as:

- Origin correctly defines user semantics, doesn't participate in performance trade-offs
- Storage faces execution performance optimization, but must comply with Origin semantics

This division enables GE to flexibly adjust execution-layer Format while guaranteeing user semantics correctness, optimizing performance.

## 3. Format Optimization Basic Principles

After clarifying the Origin and Storage two representation systems, the problems GE needs to solve can be reduced to two points:

1. How to as accurately as possible understand user's original semantics for format in whole computational graph
2. On this basis, how to select suitable execution format for operators, to obtain overall better execution efficiency

Centering on these two problems, GE's Format optimization follows a clear principle path: **first understand semantics, then optimize execution**.

### 3.1 Based on Origin Whole Graph Format Semantic Derivation

The first step of Format optimization does not directly involve performance, but rather aims to **restore and understand format semantics in the whole computational graph** as much as possible.

GE will take computational graph input formats, and format-sensitive operators (e.g. Conv2D, must clearly specify input format during computation) as anchors, propagate forward and backward in computational graph, try to derive each operator's input and output OriginFormat.

The goal of this process is to expand the understanding scope of the user's original format semantics as much as possible, providing a reliable semantic foundation for subsequent optimization.

### 3.2 Format Semantic Derivation Interruption and Uncertainty

In actual computational graphs, not all operators can maintain format semantics continuous propagation.

When encountering operators that change tensor dimension semantics (such as Reshape), the original format semantics often no longer holds. At this point, GE will consider format semantics interrupted at this location, and mark the Reshape peer's format as unknown (usually ND).

This "interruption" isn't failure, but active marking of semantic boundary, avoiding making wrong format deductions without sufficient information.

### 3.3 Based on Operator Capability StorageFormat Selection and Propagation

After completing whole graph scope OriginFormat derivation, GE enters **execution-layer format selection phase**.

At this point, Format optimization focus shifts from "whether semantics correct" to "how to obtain better execution efficiency". StorageFormat selection isn't direct mapping of OriginFormat, but needs comprehensive consideration of following factors:

- Operator's support capability for execution formats
- Different operators' affinity to specific formats
- Whole graph scope overall execution efficiency

In this phase, GE always follows one premise: **do not disrupt already confirmed Origin semantics**.

Since different operators' impact on overall performance is not balanced, GE will prioritize computationally expensive operators (such as convolution and matrix multiplication), and try to select their more affine StorageFormat for these operators. After key operators determine execution format, GE then centers on them, combines upstream/downstream adjacent operators' capabilities and constraints, propagates and coordinates StorageFormat, avoiding introducing unnecessary format conversions on critical paths.

Using computational graph from section 1.3 as example, after completing OriginFormat derivation, Format optimization will anchor on computationally expensive operator Conv2D, select its affine StorageFormat (NC1HWC0). Since subsequent ReLU operator also supports NC1HWC0, format can propagate backward along computation path and maintain consistency, finally obtaining following execution format layout:

```mermaid
graph LR
INPUT -- NCHW --> TransData_1((TransData_1))
TransData_1 -- NC1HWC0 --> Conv2D_1
Conv2D_1 -- NC1HWC0 --> ReLU_1
ReLU_1 -- NC1HWC0 --> Conv2D_2
Conv2D_2 -- NC1HWC0 --> ReLU_2
ReLU_2 -- NC1HWC0 --> TransData_2((TransData_2))
TransData_2 -- NCHW --> OUTPUT
```

### 3.4 Shape and Format Division of Labor in Derivation Process

In GE, Shape and Format derivation undertake different roles.

OriginShape derivation follows the common InferShape process in graph compilers: it takes the computational graph input Shape (that is, the user-understood Shape) as the starting point, and derives layer by layer forward according to operator semantics, until the graph output.

Unlike this, StorageShape is not an independent derivation result. When OriginShape, OriginFormat and StorageFormat are all determined, StorageShape can naturally be calculated based on StorageFormat's corresponding memory layout method.

This division decouples Shape semantic derivation from execution-layer Tensor Format, enabling format optimization to proceed independently without interfering with semantic derivation.

## 4. Understanding Format/Shape Interfaces and Types from GE External API Perspective

This chapter explains how Format/Shape are expressed at interface and type level from GE external API perspective, and explains possible understanding deviations between "concept ↔ class name".

### 4.1 Interface Layer: GetShape / GetOriginShape / GetStorageShape

In external interfaces, Shape/Format usually provide three types of access interfaces:

- `GetOriginShape()` / `GetOriginFormat()`
- `GetStorageShape()` / `GetStorageFormat()`
- `GetShape()` / `GetFormat()`

Where:

1. `GetOrigin*()` explicitly returns **Origin** perspective info, used to express user semantics.
2. `GetStorage*()` explicitly returns **Storage** perspective info, used to describe actual execution-related info.
3. `Get*()` (without Origin/Storage prefix) **doesn't explicitly specify perspective**, so returns "info simultaneously containing Origin and Storage parts".

In other words, `GetShape()` meaning is not "only return one kind of Shape", but returns an object able to simultaneously express Origin and Storage; Format related interfaces follow the same principle.

The value of this interface design lies in:

- When needing semantic info, caller can explicitly use `GetOrigin*()`
- When needing execution info, caller can explicitly use `GetStorage*()`
- If the caller wants to obtain a complete description at once, use `Get*()`

### 4.2 Type (class) Layer: Shape / StorageShape / StorageFormat Responsibility Boundaries

#### 4.2.1 Shape: Pure Data Structure, Not Binding Semantics

`Shape` is a pure data structure class, only responsible for expressing "a shape".
Therefore:

- `Shape` can be used to carry OriginShape
- `Shape` can also be used to carry StorageShape

Whether it belongs to Origin or Storage depends on its **usage context** and which interface returns it, not the `Shape` type itself's attribute.

#### 4.2.2 StorageShape / StorageFormat: Carrying Origin and Storage Description Bodies

From a concept perspective, "StorageShape" and "StorageFormat" are easily understood as only describing execution phase info; but in the class system, these two types of objects actually undertake a stronger responsibility - **they are both composite description bodies simultaneously carrying Origin and Storage two parts of info**.

The reason for binding Origin and Storage in the same type, the fundamental reason lies in Storage's own complexity. Storage may introduce dimension padding, alignment and other rules, thereby making only having "execution phase shape/format" insufficient to accurately describe its correspondence relationship with user semantics.

Using NC1HWC0 format as example, when seeing a Tensor shaped like `[8, 1, 224, 224, 16]`:

- Its StorageFormat is NC1HWC0
- Its OriginFormat could be NCHW, or could be NHWC
- Its OriginShape's C dimension could be any value between 1~16

Only from execution phase StorageShape or StorageFormat, cannot uniquely restore its corresponding semantic meaning. Only binding Origin and Storage two parts info simultaneously, can form an interpretable, stably usable complete description.

Therefore, in external API context:

- `StorageShape` and `StorageFormat` are closer to **Descriptors**
- They provide explicit access to different perspectives through `GetOrigin*()` / `GetStorage*()`
- The type itself undertakes the "binding and encapsulation" responsibility, not direct mapping of a single concept

### 4.3 Explanation and Suggestions Regarding Class Name Ambiguity

Indeed some people will confuse `class StorageShape` (type name) with "StorageShape concept" (execution shape). This confusion comes from naming's natural defect, but from a modeling perspective, this class actually undertakes the responsibility of "simultaneously carrying Origin and Storage complete description".

In practice, it is recommended to **always judge what is obtained through the interface, not through the class**. In the future, without breaking existing interface compatibility, more explicit type naming can also be used to reduce understanding cost, for example:

```C++
class ShapeDescriptor {...};
using StorageShape = ShapeDescriptor; // Deprecated: easily confused, no longer recommended use

class FormatDescriptor {...};
using StorageFormat = FormatDescriptor; // Deprecated: easily confused, no longer recommended use
```

## Appendix A: Same Tensor Contrast Example Under Origin/Storage Perspective

The table below uses a concrete example to illustrate: **for the same Tensor, how expressed info differs under different perspectives, and why a "descriptor" type is needed to simultaneously carry this info.**

| Perspective    | Interface                 | Example Content                                         | Description                         |
| ------- | -------------------- | ------------------------------------------------ | ---------------------------- |
| Origin  | `GetOriginFormat()`  | NCHW                                             | User semantic format             |
| Origin  | `GetOriginShape()`   | [8, 3, 224, 224]                                 | User-understood Shape             |
| Storage | `GetStorageFormat()` | NC1HWC0                                          | Actual execution used format           |
| Storage | `GetStorageShape()`  | [8, 1, 224, 224, 16]                             | Execution phase memory layout (including dimension padding) |
| Composite    | `GetFormat()`        | {Origin=NCHW, Storage=NC1HWC0}                   | Simultaneously carrying semantic and execution info       |
