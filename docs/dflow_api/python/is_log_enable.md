# is\_log\_enable<a name="ZH-CN_TOPIC_0000002013520489"></a>

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

## 函数功能<a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_section41982722"></a>

查询对应级别和类型的日志是否开启。

## 函数原型<a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_section3642756131310"></a>

```
is_log_enable(self, log_type: fw.FlowFuncLogType, log_level: fw.FlowFuncLogLevel) -> bool
```

## 参数说明<a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_section42300179"></a>

<a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_table66993202"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_row41236172"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p51795644"><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p51795644"></a><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p51795644"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="25.629999999999995%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p34697616"><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p34697616"></a><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p34697616"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="46.739999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p17794566"><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p17794566"></a><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p17794566"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_row32073719"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p47834478"><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p47834478"></a><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p47834478"></a><strong id="b8408125113163"><a name="b8408125113163"></a><a name="b8408125113163"></a>log_type</strong></p>
</td>
<td class="cellrowborder" valign="top" width="25.629999999999995%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"></a><a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_p49387472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p5819320165314"><a name="p5819320165314"></a><a name="p5819320165314"></a>日志类型，取值如下：</p>
<a name="ul111081431195316"></a><a name="ul111081431195316"></a><ul id="ul111081431195316"><li>DEBUG_LOG</li><li>RUN_LOG</li></ul>
</td>
</tr>
<tr id="row1959155615166"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p1960456161614"><a name="p1960456161614"></a><a name="p1960456161614"></a><strong id="b1945125819168"><a name="b1945125819168"></a><a name="b1945125819168"></a>log_level</strong></p>
</td>
<td class="cellrowborder" valign="top" width="25.629999999999995%" headers="mcps1.1.4.1.2 "><p id="p1309755173"><a name="p1309755173"></a><a name="p1309755173"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="46.739999999999995%" headers="mcps1.1.4.1.3 "><p id="p9940135812520"><a name="p9940135812520"></a><a name="p9940135812520"></a>日志级别，取值如下：</p>
<a name="ul6113111415310"></a><a name="ul6113111415310"></a><ul id="ul6113111415310"><li>DEBUG</li><li>INFO</li><li>WARN</li><li>ERROR</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_section45157297"></a>

-   True：开启
-   False：未开启

## 异常处理<a name="zh-cn_topic_0000001516398764_zh-cn_topic_0000001304225452_zh-cn_topic_0000001312720989_section3762492"></a>

无

## 约束说明<a name="zh-cn_topic_0000001516398764_section177661547145017"></a>

无

