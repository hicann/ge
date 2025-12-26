# DataFlow错误码<a name="ZH-CN_TOPIC_0000001976993810"></a>

DataFlow错误码含义如下。

<a name="table124618224416"></a>
<table><thead align="left"><tr id="row7246102215412"><th class="cellrowborder" valign="top" width="35.65643435656434%" id="mcps1.1.4.1.1"><p id="p324682215414"><a name="p324682215414"></a><a name="p324682215414"></a>返回码</p>
</th>
<th class="cellrowborder" valign="top" width="22.93770622937706%" id="mcps1.1.4.1.2"><p id="p132471122448"><a name="p132471122448"></a><a name="p132471122448"></a>含义</p>
</th>
<th class="cellrowborder" valign="top" width="41.4058594140586%" id="mcps1.1.4.1.3"><p id="p6247122216410"><a name="p6247122216410"></a><a name="p6247122216410"></a>解决方法</p>
</th>
</tr>
</thead>
<tbody><tr id="row024710221414"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p19683111558"><a name="p19683111558"></a><a name="p19683111558"></a>PARAM_INVALID=145000U</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p102471228410"><a name="p102471228410"></a><a name="p102471228410"></a>参数校验无效</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p62141512185513"><a name="p62141512185513"></a><a name="p62141512185513"></a>结合具体接口和日志找到不符合校验规则的参数，请按照资料或结合日志提示进行修改。</p>
</td>
</tr>
<tr id="row1612782310553"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p434764425510"><a name="p434764425510"></a><a name="p434764425510"></a>SHAPE_INVALID=145021U</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p10127142312558"><a name="p10127142312558"></a><a name="p10127142312558"></a>输入tensor的shape异常</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p41271123105512"><a name="p41271123105512"></a><a name="p41271123105512"></a>请结合日志修改输入的shape。</p>
</td>
</tr>
<tr id="row1057072775519"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p161211847115516"><a name="p161211847115516"></a><a name="p161211847115516"></a>DATATYPE_INVALID=145022U</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p35701727135512"><a name="p35701727135512"></a><a name="p35701727135512"></a>输入tensor的datatype异常</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p1257010274557"><a name="p1257010274557"></a><a name="p1257010274557"></a>请结合日志修改输入的datatype。</p>
</td>
</tr>
<tr id="row145793116555"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p6134450175511"><a name="p6134450175511"></a><a name="p6134450175511"></a>NOT_INIT=145001U</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p24578313550"><a name="p24578313550"></a><a name="p24578313550"></a>dataflow未初始化</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p1445783155517"><a name="p1445783155517"></a><a name="p1445783155517"></a>请参照资料和样例代码在使用dataflow接口前先调用<a href="dataflow-init.md">init</a>接口。</p>
</td>
</tr>
<tr id="row85851234105510"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p684815220553"><a name="p684815220553"></a><a name="p684815220553"></a>INNER_ERROR=545000U</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p20585103419552"><a name="p20585103419552"></a><a name="p20585103419552"></a>Python层内部错误</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p92031112195415"><a name="p92031112195415"></a><a name="p92031112195415"></a>请根据日志排查问题，或联系工程师处理（<span id="ph13211738103110"><a name="ph13211738103110"></a><a name="ph13211738103110"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p13203131245418"><a name="p13203131245418"></a><a name="p13203131245418"></a>日志的详细介绍，请参见<span id="ph11735155905"><a name="ph11735155905"></a><a name="ph11735155905"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row10792183765517"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p147931737105515"><a name="p147931737105515"></a><a name="p147931737105515"></a>FAILED=0xFFFFFFFF</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p2079353711557"><a name="p2079353711557"></a><a name="p2079353711557"></a>C++层内部错误</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p875311523412"><a name="p875311523412"></a><a name="p875311523412"></a>请根据日志排查问题，或联系工程师处理（<span id="ph19552107241"><a name="ph19552107241"></a><a name="ph19552107241"></a>您可以获取日志后单击<a href="https://www.hiascend.com/support" target="_blank" rel="noopener noreferrer">Link</a>联系技术支持。</span>）。</p>
<p id="p87531522415"><a name="p87531522415"></a><a name="p87531522415"></a>日志的详细介绍，请参见<span id="ph47531452114110"><a name="ph47531452114110"></a><a name="ph47531452114110"></a>《日志参考》</span>。</p>
</td>
</tr>
<tr id="row11759815151016"><td class="cellrowborder" valign="top" width="35.65643435656434%" headers="mcps1.1.4.1.1 "><p id="p157592158108"><a name="p157592158108"></a><a name="p157592158108"></a>SUCCESS=0</p>
</td>
<td class="cellrowborder" valign="top" width="22.93770622937706%" headers="mcps1.1.4.1.2 "><p id="p11759615181017"><a name="p11759615181017"></a><a name="p11759615181017"></a>执行成功</p>
</td>
<td class="cellrowborder" valign="top" width="41.4058594140586%" headers="mcps1.1.4.1.3 "><p id="p15759141541012"><a name="p15759141541012"></a><a name="p15759141541012"></a>-</p>
</td>
</tr>
</tbody>
</table>

