# 多func处理函数<a name="ZH-CN_TOPIC_0000002013837205"></a>

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

## 函数功能<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001312640893_section3729174918713"></a>

用户自定义的多flow func处理函数。

## 函数原型<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001312640893_section84161445741"></a>

-   普通场景下，函数输入由框架准备完毕后直接给用户使用，用户直接使用入参中的flowMsg即可。

    ```
    std::function<int32_t> (const std::shared_ptr<MetaRunContext> &runContext, const std::vector<std::shared_ptr<FlowMsg>> &flowMsg)
    ```

-   流式输入（即函数入参为队列）场景下，由用户自行从输入队列中获取数据使用。

    ```
    std::function<int32_t> (const std::shared_ptr<MetaRunContext> &runContext, const std::vector<std::shared_ptr<FlowMsgQueue>> &flowMsgQueue)
    ```

## 参数说明<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001312640893_zh-cn_topic_0182636394_section63604780"></a>

<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_table47561922"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row29169897"><th class="cellrowborder" valign="top" width="27.63%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p13951479"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="27.900000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p56327989"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="44.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p66531170"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_row20315681"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p34957489"><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p34957489"></a><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p34957489"></a>runContext</p>
</td>
<td class="cellrowborder" valign="top" width="27.900000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p12984378"><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p12984378"></a><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p12984378"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p15921183410"><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p15921183410"></a><a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001265240866_zh-cn_topic_0182636394_p15921183410"></a>处理函数的上下文信息。</p>
</td>
</tr>
<tr id="row49834401673"><td class="cellrowborder" valign="top" width="27.63%" headers="mcps1.1.4.1.1 "><p id="p16984040478"><a name="p16984040478"></a><a name="p16984040478"></a>FlowMsg/FlowMsgQueue</p>
</td>
<td class="cellrowborder" valign="top" width="27.900000000000002%" headers="mcps1.1.4.1.2 "><p id="p1798413401977"><a name="p1798413401977"></a><a name="p1798413401977"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="44.47%" headers="mcps1.1.4.1.3 "><p id="p1998413405712"><a name="p1998413405712"></a><a name="p1998413405712"></a>函数的入参/输入队列。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001312640893_section413535858"></a>

-   0：SUCCESS。
-   other：FAILED，具体请参考[UDF错误码](UDF错误码.md)。

## 异常处理<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001312640893_section1548781517515"></a>

-   如果有不可恢复的异常信息发生，返回ERROR。
-   其他情况则调用SetRetcode设置输出tensor的错误码。
-   如果返回success，调度会终止。

## 约束说明<a name="zh-cn_topic_0000001357479629_zh-cn_topic_0000001312640893_section2021419196520"></a>

使用流式输入flow func时，需要在ProcessPoint编译配置文件中，将对应func的stream\_input字段设置为true。例如：

```
{
    "func_list": [
        {
            "func_name": "Func",
            "stream_input": true
        }
    ],
    "input_num": 1,
    "output_num": 1,
    "target_bin": "libfunc.so",
    "workspace": "./",
    "cmakelist_path": "CMakeLists.txt",
}
```

