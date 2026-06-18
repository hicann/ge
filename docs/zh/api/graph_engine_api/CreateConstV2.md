# CreateConstV2<a name="ZH-CN_TOPIC_CREATE_CONST_V2"></a>

## 产品支持情况<a name="section789110355111"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="57.99999999999999%">产品</th>
<th class="cellrowborder" align="center" valign="top" width="42%">是否支持</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="57.99999999999999%">Atlas A3 训练系列产品/Atlas A3 推理系列产品</td>
<td class="cellrowborder" align="center" valign="top" width="42%">√</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="57.99999999999999%">Atlas A2 训练系列产品/Atlas A2 推理系列产品</td>
<td class="cellrowborder" align="center" valign="top" width="42%">√</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section44282627"></a>

使用ABI安全接口创建Const算子。

## 函数原型<a name="section1831611148519"></a>

```
template <typename T>
EsTensorHolder CreateConstV2(const std::vector<T> &value, const std::vector<int64_t> &dims, ge::DataType dt, ge::Format format = FORMAT_ND)
```

## 参数说明<a name="section62999330"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="10.73%">参数名</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%">输入/输出</th>
<th class="cellrowborder" valign="top" width="77.47%">说明</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="10.73%">value</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%">输入</td>
<td class="cellrowborder" valign="top" width="77.47%">张量数据向量。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="10.73%">dims</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%">输入</td>
<td class="cellrowborder" valign="top" width="77.47%">张量维度向量。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="10.73%">T</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%">输入</td>
<td class="cellrowborder" valign="top" width="77.47%">张量数据类型。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="10.73%">dt</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%">输入</td>
<td class="cellrowborder" valign="top" width="77.47%">张量的数据类型。</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="10.73%">format</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%">输入</td>
<td class="cellrowborder" valign="top" width="77.47%">张量格式，默认为FORMAT_ND。</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section30123063"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="11.06%">参数名</th>
<th class="cellrowborder" valign="top" width="13.74%">类型</th>
<th class="cellrowborder" valign="top" width="75.2%">说明</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="11.06%">-</td>
<td class="cellrowborder" valign="top" width="13.74%">EsTensorHolder</td>
<td class="cellrowborder" valign="top" width="75.2%">返回创建的Const的张量持有者算子，失败时返回无效的EsTensorHolder。</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section24049039"></a>

该接口使用`CompliantNodeBuilder` V2接口构造Const节点，避免旧IR定义结构体中`std::string`跨C++ ABI传递的风险。

使用该接口要求运行时GE包支持`CompliantNodeBuilder` V2符号。如果需要兼容不包含V2符号的老GE包，请使用`CreateConst`。

## 调用示例<a name="section16305113853313"></a>

```
EsGraphBuilder builder("test_graph");
std::vector<float> vecf = {1.1, 2.0, 3.2, 4.4};
std::vector<int64_t> dims = {4};
auto c1 = builder.CreateConstV2<float>(vecf, dims, ge::DT_FLOAT);
```
