# deallocate\_cache<a name="ZH-CN_TOPIC_0000002407890437"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="zh-cn_topic_0000002407890393_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002407890393_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002407890393_p1883113061818"><a name="zh-cn_topic_0000002407890393_p1883113061818"></a><a name="zh-cn_topic_0000002407890393_p1883113061818"></a><span id="zh-cn_topic_0000002407890393_ph20833205312295"><a name="zh-cn_topic_0000002407890393_ph20833205312295"></a><a name="zh-cn_topic_0000002407890393_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002407890393_p783113012187"><a name="zh-cn_topic_0000002407890393_p783113012187"></a><a name="zh-cn_topic_0000002407890393_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002407890393_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002407890393_p48327011813"><a name="zh-cn_topic_0000002407890393_p48327011813"></a><a name="zh-cn_topic_0000002407890393_p48327011813"></a><span id="zh-cn_topic_0000002407890393_ph583230201815"><a name="zh-cn_topic_0000002407890393_ph583230201815"></a><a name="zh-cn_topic_0000002407890393_ph583230201815"></a><term id="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002407890393_p7948163910184"><a name="zh-cn_topic_0000002407890393_p7948163910184"></a><a name="zh-cn_topic_0000002407890393_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002407890393_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002407890393_p14832120181815"><a name="zh-cn_topic_0000002407890393_p14832120181815"></a><a name="zh-cn_topic_0000002407890393_p14832120181815"></a><span id="zh-cn_topic_0000002407890393_ph980713477118"><a name="zh-cn_topic_0000002407890393_ph980713477118"></a><a name="zh-cn_topic_0000002407890393_ph980713477118"></a><term id="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term454024162214"><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term454024162214"></a><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term454024162214"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002407890393_p19948143911820"><a name="zh-cn_topic_0000002407890393_p19948143911820"></a><a name="zh-cn_topic_0000002407890393_p19948143911820"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002407890393_row15882185517522"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002407890393_p1682479135314"><a name="zh-cn_topic_0000002407890393_p1682479135314"></a><a name="zh-cn_topic_0000002407890393_p1682479135314"></a><span id="zh-cn_topic_0000002407890393_ph14880920154918"><a name="zh-cn_topic_0000002407890393_ph14880920154918"></a><a name="zh-cn_topic_0000002407890393_ph14880920154918"></a><term id="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000002407890393_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002407890393_p578615025316"><a name="zh-cn_topic_0000002407890393_p578615025316"></a><a name="zh-cn_topic_0000002407890393_p578615025316"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 函数功能<a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_section3870635"></a>

释放Cache。

如果该Cache在Allocate时关联了CacheKey，则实际的释放会延后到所有的CacheKey被拉取或执行了[remove\_cache\_key](remove_cache_key.md)。

## 函数原型<a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_section24431028171314"></a>

```
deallocate_cache(cache: KvCache)
```

## 参数说明<a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_section34835721"></a>

<a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_table2051894852017"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row4558174815206"><th class="cellrowborder" valign="top" width="22.220000000000002%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p255884814201"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b145581148152018"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="35.89%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p537710614477"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="41.89%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_p14558184812200"></a><strong id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a><a name="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_b19165651193118"></a>取值说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001417673572_zh-cn_topic_0000001359609816_row35581048202018"><td class="cellrowborder" valign="top" width="22.220000000000002%" headers="mcps1.1.4.1.1 "><p id="p6621349454"><a name="p6621349454"></a><a name="p6621349454"></a>cache</p>
</td>
<td class="cellrowborder" valign="top" width="35.89%" headers="mcps1.1.4.1.2 "><p id="p9541205974512"><a name="p9541205974512"></a><a name="p9541205974512"></a><a href="KvCache构造函数.md">KvCache</a></p>
</td>
<td class="cellrowborder" valign="top" width="41.89%" headers="mcps1.1.4.1.3 "><p id="p7172700591"><a name="p7172700591"></a><a name="p7172700591"></a>要释放的KV Cache。</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="section17821439839"></a>

```
kv_cache_manager.deallocate_cache(kv_cache)
```

## 返回值<a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_section45086037"></a>

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

执行时间超过[sync\_kv\_timeout](sync_kv_timeout.md)配置会抛出[LLMException](LLMException.md)异常。

## 约束说明<a name="zh-cn_topic_0000001481404214_zh-cn_topic_0000001488949573_zh-cn_topic_0000001357384997_zh-cn_topic_0000001312399929_section28090371"></a>

-   如果KvCache不存在或已释放，该操作为空操作。
-   本接口不支持并发调用。

