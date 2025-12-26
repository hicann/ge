# UDF错误码<a name="ZH-CN_TOPIC_0000002013837281"></a>

## flowfunc<a name="section1390959132616"></a>

flow\_func\_defines.h提供了flowfunc的错误码供用户使用，主要用于对异常逻辑的判断处理。每个错误码含义如下。

<a name="table124618224416"></a>
<table><thead align="left"><tr id="row7246102215412"><th class="cellrowborder" valign="top" width="35.61643835616437%" id="mcps1.1.4.1.1"><p id="p324682215414"><a name="p324682215414"></a><a name="p324682215414"></a>返回码</p>
</th>
<th class="cellrowborder" valign="top" width="22.977702229777016%" id="mcps1.1.4.1.2"><p id="p132471122448"><a name="p132471122448"></a><a name="p132471122448"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="41.40585941405859%" id="mcps1.1.4.1.3"><p id="p6247122216410"><a name="p6247122216410"></a><a name="p6247122216410"></a>解决方法</p>
</th>
</tr>
</thead>
<tbody><tr id="row1627501811410"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p437517374149"><a name="p437517374149"></a><a name="p437517374149"></a>FLOW_FUNC_SUCCESS = 0</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p527581812141"><a name="p527581812141"></a><a name="p527581812141"></a>执行成功</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p17275101871412"><a name="p17275101871412"></a><a name="p17275101871412"></a>不涉及</p>
</td>
</tr>
<tr id="row024710221414"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p824716221941"><a name="p824716221941"></a><a name="p824716221941"></a>FLOW_FUNC_ERR_PARAM_INVALID = 164000</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p102471228410"><a name="p102471228410"></a><a name="p102471228410"></a>参数校验无效</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p335018561446"><a name="p335018561446"></a><a name="p335018561446"></a>参数校验失败返回该错误码，包括但不限于输入参数超出系统支持范围，过程中某些参数不匹配。返回该错误码时日志会打印异常的参数及异常原因，请结合具体日志定位原因。</p>
</td>
</tr>
<tr id="row1524792217415"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p9663521848"><a name="p9663521848"></a><a name="p9663521848"></a>FLOW_FUNC_ERR_ATTR_NOT_EXITS = 164001</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p1666175215415"><a name="p1666175215415"></a><a name="p1666175215415"></a>获取属性时属性不存在</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p136520522415"><a name="p136520522415"></a><a name="p136520522415"></a>请检查获取属性的名称，确认是否在获取前对该属性进行了设置。</p>
</td>
</tr>
<tr id="row1024772214414"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p106517528412"><a name="p106517528412"></a><a name="p106517528412"></a>FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH = 164002</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p146405219417"><a name="p146405219417"></a><a name="p146405219417"></a>获取属性时属性类型不匹配</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p964552940"><a name="p964552940"></a><a name="p964552940"></a>请检查调用GetAttr接口时入参属性名称所对应的属性值类型与出参变量的数据类型是否一致。该错误码对应错误日志打印属性名对应的实际属性的数据类型。</p>
</td>
</tr>
<tr id="row724718221441"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p17627521416"><a name="p17627521416"></a><a name="p17627521416"></a>FLOW_FUNC_FAILED = 564000</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p193121929141519"><a name="p193121929141519"></a><a name="p193121929141519"></a>UDF内部错误码</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p92031112195415"><a name="p92031112195415"></a><a name="p92031112195415"></a>请根据日志排查问题，或联系工程师处理（<span id="ph13211738103110"><a name="ph13211738103110"></a><a name="ph13211738103110"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p13203131245418"><a name="p13203131245418"></a><a name="p13203131245418"></a>日志的详细介绍，请参见<span id="ph11735155905"><a name="ph11735155905"></a><a name="ph11735155905"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row229031313319"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p125761334530"><a name="p125761334530"></a><a name="p125761334530"></a>FLOW_FUNC_ERR_DATA_ALIGN_FAILED = 364000</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p10846155822412"><a name="p10846155822412"></a><a name="p10846155822412"></a>数据对齐失败</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p7352185173416"><a name="p7352185173416"></a><a name="p7352185173416"></a>可能的原因如下：</p>
<a name="ul240244910405"></a><a name="ul240244910405"></a><ul id="ul240244910405"><li>Flow func实现不正确，比如给定的输入不匹配。</li><li>某个节点执行时间超时，导致数据对齐等待超时。</li></ul>
</td>
</tr>
<tr id="row624719228413"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p36010520414"><a name="p36010520414"></a><a name="p36010520414"></a>FLOW_FUNC_ERR_TIME_OUT_ERROR = 564001</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p186113521441"><a name="p186113521441"></a><a name="p186113521441"></a>执行NN超时</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p12596521343"><a name="p12596521343"></a><a name="p12596521343"></a>请检查日志中是否存在其他报错导致模型执行失败，若存在其他报错针对实际报错定位。若无报错日志，显示模型正常执行，请调整fetch data接口传递的timeout入参，可增加其值或直接设置为-1。</p>
</td>
</tr>
<tr id="row1835315273405"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p12386114354015"><a name="p12386114354015"></a><a name="p12386114354015"></a>FLOW_FUNC_ERR_NOT_SUPPORT = 564002</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p15353102716406"><a name="p15353102716406"></a><a name="p15353102716406"></a>功能不支持</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p12970154312118"><a name="p12970154312118"></a><a name="p12970154312118"></a>可能的原因如下：</p>
<a name="ul110519016259"></a><a name="ul110519016259"></a><ul id="ul110519016259"><li>单func接口未开放该能力，替换成多func接口可以规避该报错。</li><li>用户未实现对应的接口，如故障恢复场景ResetFlowFuncState未实现默认会返回不支持。</li></ul>
</td>
</tr>
<tr id="row13482832144019"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p1496028194118"><a name="p1496028194118"></a><a name="p1496028194118"></a>FLOW_FUNC_STATUS_REDEPLOYING = 564003</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p1648213327401"><a name="p1648213327401"></a><a name="p1648213327401"></a>降级部署中</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p11482183294014"><a name="p11482183294014"></a><a name="p11482183294014"></a>可恢复错误触发降级部署导致当前获取不到数据，等待降级部署结束后会返回其他返回码。若降级部署成功，正常返回数据；若降级部署失败，返回其他不可恢复错误码。</p>
</td>
</tr>
<tr id="row2952037112018"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p0952137102013"><a name="p0952137102013"></a><a name="p0952137102013"></a>FLOW_FUNC_STATUS_EXIT = 564004</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p99521437202019"><a name="p99521437202019"></a><a name="p99521437202019"></a>UDF进程退出中</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p595233714202"><a name="p595233714202"></a><a name="p595233714202"></a>Flow func在等待输入数据的过程中，如果进程收到退出信号，会返回该错误码，表示进程准备退出，停止输入数据准备。需要根据日志排查UDF进程收到退出信号的原因。</p>
</td>
</tr>
<tr id="row324818228411"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p1258752045"><a name="p1258752045"></a><a name="p1258752045"></a>FLOW_FUNC_ERR_DRV_ERROR = 564100</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p1759752944"><a name="p1759752944"></a><a name="p1759752944"></a>driver通用错误</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p195710521949"><a name="p195710521949"></a><a name="p195710521949"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph184672031172210"><a name="ph184672031172210"></a><a name="ph184672031172210"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p973344519523"><a name="p973344519523"></a><a name="p973344519523"></a>日志的详细介绍，请参见<span id="ph2843153695913"><a name="ph2843153695913"></a><a name="ph2843153695913"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row1024812228411"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p1857352347"><a name="p1857352347"></a><a name="p1857352347"></a>FLOW_FUNC_ERR_MEM_BUF_ERROR = 564101</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p19582521149"><a name="p19582521149"></a><a name="p19582521149"></a>驱动内存buffer接口错误</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p12864155235320"><a name="p12864155235320"></a><a name="p12864155235320"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志报错排查问题，或联系工程师（<span id="ph155061036142217"><a name="ph155061036142217"></a><a name="ph155061036142217"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p16864175215312"><a name="p16864175215312"></a><a name="p16864175215312"></a>日志的详细介绍，请参见<span id="ph9925144375914"><a name="ph9925144375914"></a><a name="ph9925144375914"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row13248192214413"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p13555521747"><a name="p13555521747"></a><a name="p13555521747"></a>FLOW_FUNC_ERR_QUEUE_ERROR = 564102</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p145519521147"><a name="p145519521147"></a><a name="p145519521147"></a>驱动队列接口错误</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p1221854175316"><a name="p1221854175316"></a><a name="p1221854175316"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志报错排查问题，或联系工程师（<span id="ph175251342192217"><a name="ph175251342192217"></a><a name="ph175251342192217"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p18285455317"><a name="p18285455317"></a><a name="p18285455317"></a>日志的详细介绍，请参见<span id="ph1616315018597"><a name="ph1616315018597"></a><a name="ph1616315018597"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row16248422241"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p135455211410"><a name="p135455211410"></a><a name="p135455211410"></a>FLOW_FUNC_ERR_EVENT_ERROR = 564103</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p15667103813153"><a name="p15667103813153"></a><a name="p15667103813153"></a>驱动事件接口错误</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p735005655312"><a name="p735005655312"></a><a name="p735005655312"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志报错排查问题，或联系工程师（<span id="ph748334572214"><a name="ph748334572214"></a><a name="ph748334572214"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p13350956155314"><a name="p13350956155314"></a><a name="p13350956155314"></a>日志的详细介绍，请参见<span id="ph813412577593"><a name="ph813412577593"></a><a name="ph813412577593"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row140184251818"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p12402124221817"><a name="p12402124221817"></a><a name="p12402124221817"></a>FLOW_FUNC_ERR_USER_DEFINE_START  = 9900000</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p4402194291815"><a name="p4402194291815"></a><a name="p4402194291815"></a>用户自定义错误码，从当前错误码开始定义</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p6402204261819"><a name="p6402204261819"></a><a name="p6402204261819"></a>-</p>
</td>
</tr>
<tr id="row10511943191816"><td class="cellrowborder" valign="top" width="35.61643835616437%" headers="mcps1.1.4.1.1 "><p id="p55121043181813"><a name="p55121043181813"></a><a name="p55121043181813"></a>FLOW_FUNC_ERR_USER_DEFINE_END = 9999999</p>
</td>
<td class="cellrowborder" valign="top" width="22.977702229777016%" headers="mcps1.1.4.1.2 "><p id="p4512164341819"><a name="p4512164341819"></a><a name="p4512164341819"></a>用户自定义错误码，以当前错误码结束</p>
</td>
<td class="cellrowborder" valign="top" width="41.40585941405859%" headers="mcps1.1.4.1.3 "><p id="p17512643201818"><a name="p17512643201818"></a><a name="p17512643201818"></a>-</p>
</td>
</tr>
</tbody>
</table>

## AICPU<a name="section119131377263"></a>

AICPU在执行模型的过程中，有可能向用户上报以下错误码，每个错误码含义如下。

<a name="table1049031341218"></a>
<table><thead align="left"><tr id="row5518171315125"><th class="cellrowborder" valign="top" width="53.94460553944606%" id="mcps1.1.4.1.1"><p id="p551821351219"><a name="p551821351219"></a><a name="p551821351219"></a>返回码</p>
</th>
<th class="cellrowborder" valign="top" width="13.938606139386062%" id="mcps1.1.4.1.2"><p id="p85189131122"><a name="p85189131122"></a><a name="p85189131122"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="32.11678832116788%" id="mcps1.1.4.1.3"><p id="p175181413161216"><a name="p175181413161216"></a><a name="p175181413161216"></a>解决方法</p>
</th>
</tr>
</thead>
<tbody><tr id="row175185138123"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p1851821312120"><a name="p1851821312120"></a><a name="p1851821312120"></a>int32_t AICPU_SCHEDULE_ERROR_PARAMETER_NOT_VALID = 521001</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p1451810134125"><a name="p1451810134125"></a><a name="p1451810134125"></a>参数校验无效</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p7518161319124"><a name="p7518161319124"></a><a name="p7518161319124"></a>参数校验失败返回该错误码，包括但不限于输入参数超出系统支持范围，过程中某些参数不匹配。返回该错误码时日志会打印异常的参数及异常原因，请结合具体日志定位原因。</p>
</td>
</tr>
<tr id="row5518713201219"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p6518181315120"><a name="p6518181315120"></a><a name="p6518181315120"></a>int32_t AICPU_SCHEDULE_ERROR_FROM_DRV = 521003</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p55181135129"><a name="p55181135129"></a><a name="p55181135129"></a>Driver接口返回错误</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p1035817581"><a name="p1035817581"></a><a name="p1035817581"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph1852218515228"><a name="ph1852218515228"></a><a name="ph1852218515228"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p10358971884"><a name="p10358971884"></a><a name="p10358971884"></a>日志的详细介绍，请参见<span id="ph183590715820"><a name="ph183590715820"></a><a name="ph183590715820"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row951821311211"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p14518151311120"><a name="p14518151311120"></a><a name="p14518151311120"></a>int32_t AICPU_SCHEDULE_ERROR_NOT_FOUND_LOGICAL_TASK = 521005</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p17518101351216"><a name="p17518101351216"></a><a name="p17518101351216"></a>未找到需要执行的AICPU任务</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p123427347257"><a name="p123427347257"></a><a name="p123427347257"></a>请检查环境驱动包与CANN包版本是否兼容。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph18379105812219"><a name="ph18379105812219"></a><a name="ph18379105812219"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p12342143411256"><a name="p12342143411256"></a><a name="p12342143411256"></a>日志的详细介绍，请参见<span id="ph834223416255"><a name="ph834223416255"></a><a name="ph834223416255"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row25189133125"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p135181613201216"><a name="p135181613201216"></a><a name="p135181613201216"></a>int32_t AICPU_SCHEDULE_ERROR_INNER_ERROR = 521008</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p9518111316121"><a name="p9518111316121"></a><a name="p9518111316121"></a>AICPU内部错误</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p155187131124"><a name="p155187131124"></a><a name="p155187131124"></a>请检查环境驱动包与CANN包版本是否兼容。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph6184143112311"><a name="ph6184143112311"></a><a name="ph6184143112311"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p9723708347"><a name="p9723708347"></a><a name="p9723708347"></a>日志的详细介绍，请参见<span id="ph3723408345"><a name="ph3723408345"></a><a name="ph3723408345"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row15518111317125"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p1651816138126"><a name="p1651816138126"></a><a name="p1651816138126"></a>int32_t AICPU_SCHEDULE_ERROR_OVERFLOW = 521011</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p651810136128"><a name="p651810136128"></a><a name="p651810136128"></a>发生溢出</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p152769235162"><a name="p152769235162"></a><a name="p152769235162"></a>乘法或加法运算发生溢出，请结合具体日志定位原因。</p>
</td>
</tr>
<tr id="row1651851371217"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p19518121316128"><a name="p19518121316128"></a><a name="p19518121316128"></a>int32_t AICPU_SCHEDULE_ERROR_MODEL_EXIT_ERR = 521104</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p16518111318125"><a name="p16518111318125"></a><a name="p16518111318125"></a>模型触发执行失败</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p179081152172815"><a name="p179081152172815"></a><a name="p179081152172815"></a>模型执行过程中，返回值被置为异常标记位，因此模型无法继续执行。请查看日志中是否有其他报错，结合具体日志定位原因。</p>
</td>
</tr>
<tr id="row2518191318124"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p16518313191210"><a name="p16518313191210"></a><a name="p16518313191210"></a>int32_t AICPU_SCHEDULE_ERROR_MODEL_EXECUTE_FAILED = 521106</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p20518121311217"><a name="p20518121311217"></a><a name="p20518121311217"></a>模型执行过程中TSCH上报的模型执行失败</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p3670171863015"><a name="p3670171863015"></a><a name="p3670171863015"></a>模型执行过程中，收到异常终止消息，需要终止模型（终止原因为模型流执行失败）。请查看日志中是否有其他报错，结合具体日志定位原因。</p>
</td>
</tr>
<tr id="row17518513181216"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p751811381219"><a name="p751811381219"></a><a name="p751811381219"></a>int32_t AICPU_SCHEDULE_ERROR_TSCH_OTHER_ERROR = 521107</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p1151820131124"><a name="p1151820131124"></a><a name="p1151820131124"></a>模型执行过程中TSCH上报的其他错误</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p351931313124"><a name="p351931313124"></a><a name="p351931313124"></a>模型执行过程中，收到异常终止消息，需要终止模型。请查看日志中是否有其他报错，结合具体日志定位原因。</p>
</td>
</tr>
<tr id="row216641681116"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p713646162015"><a name="p713646162015"></a><a name="p713646162015"></a>int32_t A<span>ICPU_SCHEDULE_ERROR_DISCARD_DATA</span> = 521108</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p112512552016"><a name="p112512552016"></a><a name="p112512552016"></a>模型执行过程中丢弃Mbuf数据</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p6118844112014"><a name="p6118844112014"></a><a name="p6118844112014"></a>模型执行过程中，缓存的Mbuf数据超过阈值，需要丢弃Mbuf数据。解决方法为调整缓存Mbuf的数量或者时间阈值。</p>
</td>
</tr>
<tr id="row951917136120"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p1751911138126"><a name="p1751911138126"></a><a name="p1751911138126"></a>int32_t AICPU_SCHEDULE_ERROR_DRV_ERR = 521206</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p4519121351211"><a name="p4519121351211"></a><a name="p4519121351211"></a>driver接口返回错误</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p1227311261314"><a name="p1227311261314"></a><a name="p1227311261314"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph641412121233"><a name="ph641412121233"></a><a name="ph641412121233"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p4273192131316"><a name="p4273192131316"></a><a name="p4273192131316"></a>日志的详细介绍，请参见<span id="ph527314251315"><a name="ph527314251315"></a><a name="ph527314251315"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row11519201361210"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p95195137125"><a name="p95195137125"></a><a name="p95195137125"></a>int32_t AICPU_SCHEDULE_ERROR_MALLOC_MEM_FAIL_THROUGH_DRV = 521207</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p1451951317126"><a name="p1451951317126"></a><a name="p1451951317126"></a>通过driver接口申请内存失败</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p3148114910915"><a name="p3148114910915"></a><a name="p3148114910915"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请检查设备内存使用情况，是否达到设备内存上限。请根据日志报错排查问题，或联系工程师（<span id="ph562361812314"><a name="ph562361812314"></a><a name="ph562361812314"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p2148549194"><a name="p2148549194"></a><a name="p2148549194"></a>日志的详细介绍，请参见<span id="ph1014814496918"><a name="ph1014814496918"></a><a name="ph1014814496918"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row9519171311210"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p8519213131211"><a name="p8519213131211"></a><a name="p8519213131211"></a>int32_t AICPU_SCHEDULE_ERROR_SAFE_FUNCTION_ERR = 521208</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p651919130128"><a name="p651919130128"></a><a name="p651919130128"></a>memcpy_s等安全函数执行失败</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p19556193910359"><a name="p19556193910359"></a><a name="p19556193910359"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph7539422152320"><a name="ph7539422152320"></a><a name="ph7539422152320"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p115561939143520"><a name="p115561939143520"></a><a name="p115561939143520"></a>日志的详细介绍，请参见<span id="ph1855643914356"><a name="ph1855643914356"></a><a name="ph1855643914356"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row175196134122"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p9519201321219"><a name="p9519201321219"></a><a name="p9519201321219"></a>int32_t AICPU_SCHEDULE_ERROR_INVAILD_EVENT_SUBMIT = 521209</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p65191613161213"><a name="p65191613161213"></a><a name="p65191613161213"></a>AICPU提交事件失败</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p9794317150"><a name="p9794317150"></a><a name="p9794317150"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph16330269233"><a name="ph16330269233"></a><a name="ph16330269233"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p1381543131516"><a name="p1381543131516"></a><a name="p1381543131516"></a>日志的详细介绍，请参见<span id="ph688430152"><a name="ph688430152"></a><a name="ph688430152"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row16519613101214"><td class="cellrowborder" valign="top" width="53.94460553944606%" headers="mcps1.1.4.1.1 "><p id="p7519111301216"><a name="p7519111301216"></a><a name="p7519111301216"></a>int32_t AICPU_SCHEDULE_ERROR_CALL_HCCL = 521500</p>
</td>
<td class="cellrowborder" valign="top" width="13.938606139386062%" headers="mcps1.1.4.1.2 "><p id="p18519121314126"><a name="p18519121314126"></a><a name="p18519121314126"></a>AICPU调用HCCL接口失败</p>
</td>
<td class="cellrowborder" valign="top" width="32.11678832116788%" headers="mcps1.1.4.1.3 "><p id="p18552234103512"><a name="p18552234103512"></a><a name="p18552234103512"></a>请检查环境驱动包安装是否正常，检查设备状态是否正常。若当前环境安装包符合预期且环境状态正常，请根据日志排查问题，或联系工程师处理（<span id="ph8507183052312"><a name="ph8507183052312"></a><a name="ph8507183052312"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p1552123416350"><a name="p1552123416350"></a><a name="p1552123416350"></a>日志的详细介绍，请参见<span id="ph255233483517"><a name="ph255233483517"></a><a name="ph255233483517"></a>《日志参考》</span>。</p>
</td>
</tr>
</tbody>
</table>

