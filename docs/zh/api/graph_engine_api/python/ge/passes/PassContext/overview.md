# 简介

编译期Pass上下文，为C++侧CustomPassContext的Python视图。由引擎注入到FusionBasePass.run\(graph, context\)中；当PatternFusionPass或DecomposePass的`meet_requirements`、`replacement`声明该参数时，也会注入到对应回调中，用于查询或设置Pass名称、错误信息以及编译选项。该对象仅在当前回调栈内有效。
