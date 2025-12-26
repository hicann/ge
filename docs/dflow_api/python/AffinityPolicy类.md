# AffinityPolicy类<a name="ZH-CN_TOPIC_0000002013400977"></a>

## 产品支持情况<a name="section8178181118225"></a>

<a name="zh-cn_topic_0000002013832557_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002013832557_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002013832557_p1883113061818"><a name="zh-cn_topic_0000002013832557_p1883113061818"></a><a name="zh-cn_topic_0000002013832557_p1883113061818"></a><span id="zh-cn_topic_0000002013832557_ph20833205312295"><a name="zh-cn_topic_0000002013832557_ph20833205312295"></a><a name="zh-cn_topic_0000002013832557_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002013832557_p783113012187"><a name="zh-cn_topic_0000002013832557_p783113012187"></a><a name="zh-cn_topic_0000002013832557_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002013832557_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002013832557_p48327011813"><a name="zh-cn_topic_0000002013832557_p48327011813"></a><a name="zh-cn_topic_0000002013832557_p48327011813"></a><span id="zh-cn_topic_0000002013832557_ph583230201815"><a name="zh-cn_topic_0000002013832557_ph583230201815"></a><a name="zh-cn_topic_0000002013832557_ph583230201815"></a><term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term12835255145414"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term12835255145414"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term12835255145414"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002013832557_p7948163910184"><a name="zh-cn_topic_0000002013832557_p7948163910184"></a><a name="zh-cn_topic_0000002013832557_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002013832557_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002013832557_p14832120181815"><a name="zh-cn_topic_0000002013832557_p14832120181815"></a><a name="zh-cn_topic_0000002013832557_p14832120181815"></a><span id="zh-cn_topic_0000002013832557_ph1483216010188"><a name="zh-cn_topic_0000002013832557_ph1483216010188"></a><a name="zh-cn_topic_0000002013832557_ph1483216010188"></a><term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1551319498507"><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1551319498507"></a><a name="zh-cn_topic_0000002013832557_zh-cn_topic_0000001312391781_term1551319498507"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002013832557_p19948143911820"><a name="zh-cn_topic_0000002013832557_p19948143911820"></a><a name="zh-cn_topic_0000002013832557_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 函数功能<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

亲和策略枚举定义。

## 函数原型<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
NO_AFFINITY = (fw.AffinityPolicy.NO_AFFINITY)         # 不需要亲和
ROW_AFFINITY = (fw.AffinityPolicy.ROW_AFFINITY)   # 按行亲和，即将行相同的数据路由到相同节点。
COL_AFFINITY = (fw.AffinityPolicy.COL_AFFINITY)      # 按列亲和，即将列相同的数据路由到相同节点。
def __init__(self, inner_type):
    self.inner_type = inner_type
```

>![](public_sys-resources/icon-note.gif) **说明：** 
>按行和按列亲和的策略只用于设置[BalanceConfig](BalanceConfig构造函数.md)均衡分发。

## 参数说明<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section63604780"></a>

无

## 返回值<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572112"></a>

无

## 异常处理<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section51713556"></a>

无

## 约束说明<a name="zh-cn_topic_0000001481724054_zh-cn_topic_0000001357504889_zh-cn_topic_0000001265399770_zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768825"></a>

无

