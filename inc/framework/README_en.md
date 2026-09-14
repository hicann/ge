# OM2 SO-to-Executor C ABI Compatibility Rules

This document defines backward- and forward-compatibility rules for APIs, data structures, and their use between the Executor and SO in OM2. The Executor obtains SO capabilities by API symbol name through `dlopen + dlsym`; the JSON file accompanying the SO separately records the ABI compatibility version and the build/release version.

## 1. APIs Use the C Interface

- APIs invoked in either direction between the Executor and SO use the C ABI, so C++ name mangling, class layouts, STL, templates, exceptions, and RTTI are not part of the interface.
- Exported symbol names, calling conventions, and export visibility are fixed.
- An SO may use C++ internally, but exceptions must not cross the Executor/SO boundary. Failures must be expressed through the existing C-compatible return values or output parameters.

## 2. Published APIs Must Not Be Removed or Changed

- This rule applies to both exported APIs invoked by the Executor and callback APIs through which the SO invokes the Executor.
- A published API's symbol name, function prototype, number and order of parameters, parameter and return types, calling convention, nullability, and export visibility must not be removed or changed.
- The Executor uses `dlopen + dlsym` to determine whether a new API is callable. A new capability may only add an exported symbol; it must not reuse, replace, or remove an existing symbol.
- The behavioral semantics of a published API are also frozen: it must not narrow previously valid input, or change default behavior, return-code meanings, release responsibilities, object lifetimes, or output meanings.
- Once published, the numeric values and semantics of status codes, enum values, and flag bits must remain unchanged. New values may only be appended.
- The thread-safety level, reentrancy, and call-order requirements of a published API must not be narrowed. For example, an API that was safe to invoke concurrently must not be changed to require external serialization.
- A callback API's invocation time, execution thread, context-pointer meaning, object lifetime during the callback, and reentrancy constraints are frozen behavioral semantics.
- String encoding, whether a string is `\0`-terminated, whether a length field includes the terminator, memory ownership, and lifetime are frozen API semantics.

## 3. Parameter Extension for New APIs

- Positional parameters of a new API must remain minimal and stable.
- A group of parameters that needs extension uses separate Input and Output structures with `struct_size`; it must not be extended by continually adding positional parameters to the function.
- Input and Output use different structures and parameters: the Executor fills Input and the SO reads it; the Executor provides storage for Output and the SO writes it.
- Stable, simple scalar parameters do not require an additional structure.

## 4. Numeric Types Across the ABI

- New public API parameters and public structure fields use numeric types with explicit widths.
- The type of a published field or parameter must not be changed.
- The definitions and semantics of cross-boundary numeric values, such as status codes, enums, and flags, are frozen together with the API; the interpretation of existing values must not change.

## 5. Memory, Objects, and Handles Across the Boundary

- Every cross-boundary memory allocation, object, and handle must define its creator, releaser, and lifetime; these conventions must not change once published.
- A resource created by one side must not be released directly by the other side using its own allocator. The creating side must provide the corresponding release or destroy API.
- An evolvable object is exposed only through an opaque handle; its internal object layout is not exposed.
- A handle's validity and invalidation time, behavior after destruction, and required resource cleanup before SO unload are API behavioral semantics.

## 6. Extensible Public Structures

- Every cross-ABI public structure that needs future extension uses `struct_size`.
- `struct_size` represents the byte range of the structure supplied for this invocation. The caller initializes it correctly before the call, and neither the Executor nor the SO may modify it during the call.
- When reading Input or writing Output, the SO may access only fields covered by `struct_size`.
- `struct_size` does not represent the actual result size, element count, or required buffer size. Those values use separate fields.
- A structure and its published fields must not be removed, renamed, have their types or order changed, have their meanings changed, or have their alignment requirements changed.
- New fields may only be appended to the end of a structure. The new structure's `sizeof` must be strictly greater than its previous value, and the first new field's `offsetof` must be no less than the previous `sizeof`.
- A new field must have a default value that preserves old behavior. An old structure that lacks the field must not be rejected for that reason.
- If a new capability has no default value that preserves old behavior, do not append a required field to an old structure. Add a new API or parameter structure instead.
- Initialization syntax sugar is only recommended. The rule only requires `struct_size` to match the supplied memory range before the call; it does not require initialization at definition, subsequent initialization, or heap allocation.

### Field Semantic Validity

`struct_size` can determine only whether a field is within the memory range supplied by the caller. It cannot determine whether the caller intentionally set the field, nor whether a feature is enabled.

- Every field controlling an optional feature or optional configuration must distinguish among “field not supplied/unspecified”, “field supplied but feature disabled or using default behavior”, and “field supplied with feature enabled”.
- Do not infer that a field has business meaning solely because it is non-zero or covered by `struct_size`.
- According to the field semantics, use an explicit enable field, an independent validity bit, or a mode field containing states such as `UNSPECIFIED`, `DISABLED`, and `ENABLED` to express the distinction.

## 7. Structure Nesting

- An extensible structure should normally be nested through a pointer, allowing the inner structure to evolve independently according to its `struct_size`; its extension must not cause published outer-member offsets to drift.
- If an aggregate type, such as a union, is already pointer-indirected before crossing the ABI boundary and the aggregate itself is not passed across that boundary by value, its by-value members may be extended. The published member prefix layout must remain unchanged, and no published outer member may move.
- An inner structure may be nested by value without additional constraints only when its layout is permanently frozen, it will never gain fields, and it contains no extensible structure.

## 8. Explicit ABI Padding and Layout Stability

- A public ABI structure must not contain unnamed, compiler-inserted alignment padding.
- Alignment padding is uniformly declared as `uint8_t abi_pad_<index>[<byte_count>]`; use the array form even when there is only one byte.
- `abi_pad_N` is frozen physical layout padding and carries no business data. It may be zero-initialized with the structure, but the SO must not read, write, or make logical decisions based on its content.
- A published `abi_pad_N` must not be enabled, reused, removed, shortened, have its type changed, be moved, or be renamed. New fields must be appended after it.
- If new padding is required after appending a field, add a new `abi_pad_N`; do not use padding from an older version.
- The packing/alignment strategy of a public ABI structure must not change or drift because of `#pragma pack`, compiler options, or local attributes.
