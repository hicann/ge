# CopyKvBlocks<a name="ZH-CN_TOPIC_0000002407742981"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="zh-cn_topic_0000002407583129_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002407583129_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002407583129_p1883113061818"><a name="zh-cn_topic_0000002407583129_p1883113061818"></a><a name="zh-cn_topic_0000002407583129_p1883113061818"></a><span id="zh-cn_topic_0000002407583129_ph20833205312295"><a name="zh-cn_topic_0000002407583129_ph20833205312295"></a><a name="zh-cn_topic_0000002407583129_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002407583129_p783113012187"><a name="zh-cn_topic_0000002407583129_p783113012187"></a><a name="zh-cn_topic_0000002407583129_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002407583129_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002407583129_p48327011813"><a name="zh-cn_topic_0000002407583129_p48327011813"></a><a name="zh-cn_topic_0000002407583129_p48327011813"></a><span id="zh-cn_topic_0000002407583129_ph583230201815"><a name="zh-cn_topic_0000002407583129_ph583230201815"></a><a name="zh-cn_topic_0000002407583129_ph583230201815"></a><term id="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002407583129_p7948163910184"><a name="zh-cn_topic_0000002407583129_p7948163910184"></a><a name="zh-cn_topic_0000002407583129_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002407583129_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002407583129_p14832120181815"><a name="zh-cn_topic_0000002407583129_p14832120181815"></a><a name="zh-cn_topic_0000002407583129_p14832120181815"></a><span id="zh-cn_topic_0000002407583129_ph980713477118"><a name="zh-cn_topic_0000002407583129_ph980713477118"></a><a name="zh-cn_topic_0000002407583129_ph980713477118"></a><term id="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term454024162214"><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term454024162214"></a><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term454024162214"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002407583129_p19948143911820"><a name="zh-cn_topic_0000002407583129_p19948143911820"></a><a name="zh-cn_topic_0000002407583129_p19948143911820"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002407583129_row15882185517522"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002407583129_p1682479135314"><a name="zh-cn_topic_0000002407583129_p1682479135314"></a><a name="zh-cn_topic_0000002407583129_p1682479135314"></a><span id="zh-cn_topic_0000002407583129_ph14880920154918"><a name="zh-cn_topic_0000002407583129_ph14880920154918"></a><a name="zh-cn_topic_0000002407583129_ph14880920154918"></a><term id="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000002407583129_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002407583129_p578615025316"><a name="zh-cn_topic_0000002407583129_p578615025316"></a><a name="zh-cn_topic_0000002407583129_p578615025316"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 函数功能<a name="section435319584185"></a>

PA场景下，通过block列表的方式拷贝KV Cache。支持D2D，D2H，H2D的拷贝。

-   D2D场景主要是针对当多个回答需要共用相同block，block没填满时，新增的token需要拷贝到新的block上继续迭代。
-   H2D和D2H的拷贝主要用于对应block\_index上Cache内存的换入换出。

## 函数原型<a name="section1335335821812"></a>

```
Status CopyKvBlocks(const Cache &src_cache,
                    const Cache &dst_cache,
                    const std::vector<uint64_t> &src_blocks,
                    const std::vector<std::vector<uint64_t>> &dst_blocks_list)
```

## 参数说明<a name="section535355891816"></a>

<a name="table1635315851812"></a>
<table><thead align="left"><tr id="row2353205871819"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="p1135313587185"><a name="p1135313587185"></a><a name="p1135313587185"></a><strong id="b735355821814"><a name="b735355821814"></a><a name="b735355821814"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="35.89%" id="mcps1.1.4.1.2"><p id="p435345871812"><a name="p435345871812"></a><a name="p435345871812"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="41.89%" id="mcps1.1.4.1.3"><p id="p435314585183"><a name="p435314585183"></a><a name="p435314585183"></a><strong id="b835345891816"><a name="b835345891816"></a><a name="b835345891816"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row12353258151811"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p535335816188"><a name="p535335816188"></a><a name="p535335816188"></a>src_cache</p>
</td>
<td class="cellrowborder" valign="top" width="35.89%" headers="mcps1.1.4.1.2 "><p id="p1835313587185"><a name="p1835313587185"></a><a name="p1835313587185"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="41.89%" headers="mcps1.1.4.1.3 "><p id="p1235345851814"><a name="p1235345851814"></a><a name="p1235345851814"></a>源Cache。</p>
</td>
</tr>
<tr id="row83531858191812"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p5353195813181"><a name="p5353195813181"></a><a name="p5353195813181"></a>dst_cache</p>
</td>
<td class="cellrowborder" valign="top" width="35.89%" headers="mcps1.1.4.1.2 "><p id="p13353258141811"><a name="p13353258141811"></a><a name="p13353258141811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="41.89%" headers="mcps1.1.4.1.3 "><p id="p193532058191817"><a name="p193532058191817"></a><a name="p193532058191817"></a>目的Cache。</p>
</td>
</tr>
<tr id="row535325811810"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p15353558191810"><a name="p15353558191810"></a><a name="p15353558191810"></a>src_blocks</p>
</td>
<td class="cellrowborder" valign="top" width="35.89%" headers="mcps1.1.4.1.2 "><p id="p435335811186"><a name="p435335811186"></a><a name="p435335811186"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="41.89%" headers="mcps1.1.4.1.3 "><p id="p2353195851819"><a name="p2353195851819"></a><a name="p2353195851819"></a>源Cache的block index列表。</p>
</td>
</tr>
<tr id="row735365817188"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p19353125812180"><a name="p19353125812180"></a><a name="p19353125812180"></a>dst_blocks_list</p>
</td>
<td class="cellrowborder" valign="top" width="35.89%" headers="mcps1.1.4.1.2 "><p id="p8353185891810"><a name="p8353185891810"></a><a name="p8353185891810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="41.89%" headers="mcps1.1.4.1.3 "><p id="p1035335817180"><a name="p1035335817180"></a><a name="p1035335817180"></a>目标Cache的block index列表的列表，一组src_blocks可以拷贝到多组dst_blocks。</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="section10353358171810"></a>

```
Status ret = llm_datadist.CopyKvCache(src_cache, dst_cache, {1,2}, {{1,2},{3,4}})
```

## 返回值<a name="section73536584182"></a>

-   LLM\_SUCCESS：成功
-   LLM\_PARAM\_INVALID：参数错误
-   其他：失败

## 约束说明<a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_section28090371"></a>

该接口调用之前，需要先调用[Initialize](Initialize.md)接口完成初始化。不支持Host-\>Host的拷贝。

