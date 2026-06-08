# EsCreateConstV2<a name="ZH-CN_TOPIC_ES_CREATE_CONST_V2"></a>

## 功能说明<a name="section44282627"></a>

使用ABI安全接口创建指定类型的Const算子（原始指针版本）。

## 函数原型<a name="section1831611148519"></a>

```
template <typename T>
EsCTensorHolder *EsCreateConstV2(EsCGraphBuilder *graph, const T *value, const int64_t *dims, int64_t dim_num, ge::DataType dt, ge::Format format = FORMAT_ND)
```

## 参数说明<a name="section62999330"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="11.790000000000001%">参数名</th>
<th class="cellrowborder" valign="top" width="12.6%">输入/输出</th>
<th class="cellrowborder" valign="top" width="75.61%">说明</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="11.790000000000001%">T</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">张量数据类型。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="11.790000000000001%">graph</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">图构建器指针。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="11.790000000000001%">value</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">张量数据指针。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="11.790000000000001%">dims</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">张量维度数组指针。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="11.790000000000001%">dim_num</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">维度数量。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="11.790000000000001%">dt</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">张量的数据类型。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="11.790000000000001%">format</td>
<td class="cellrowborder" valign="top" width="12.6%">输入</td>
<td class="cellrowborder" valign="top" width="75.61%">张量格式，默认为FORMAT_ND。</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="11.06%">参数名</th>
<th class="cellrowborder" valign="top" width="13.59%">类型</th>
<th class="cellrowborder" valign="top" width="75.35%">说明</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="11.06%">-</td>
<td class="cellrowborder" valign="top" width="13.59%">EsCTensorHolder</td>
<td class="cellrowborder" valign="top" width="75.35%">返回创建的Const的张量持有者算子，失败时返回nullptr。</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

该接口使用`CompliantNodeBuilder` V2接口构造Const节点，避免旧IR定义结构体中`std::string`跨C++ ABI传递的风险。

使用该接口要求运行时GE包支持`CompliantNodeBuilder` V2符号。如果需要兼容不包含V2符号的老GE包，请使用`EsCreateConst`。

## 调用示例<a name="section16305113853313"></a>

```
EsCGraphBuilder *graph = EsCreateGraphBuilder("graph_name");
std::vector<int64_t> data = {1, 2, 3};
std::vector<int64_t> dims = {3};
auto const_tensor = ge::es::EsCreateConstV2<int64_t>(graph, data.data(), dims.data(), dims.size(), ge::DT_INT64);
```
