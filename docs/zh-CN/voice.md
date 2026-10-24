# 声音克隆

换脸仍是主路径。声音转换是后面一步，只处理**本来就有人声的视频**。

## 源

页面默认是 **视频（脸+声音，推荐）**。上传一段这个人的视频。运行时会在整段里抽帧，按脸的大小、清晰度和正脸程度打分，再对最好的几张做向量。

平均不是默认就用。分数较低的约三分之一张脸会留出来做核对。如果其余脸的平均向量比「最好的一张」更接近这些留出的脸，身份就用平均；否则用最好的一张。只有一张可用的脸时就用那一张。BlendSwap 要的是源脸裁剪，所以始终用最好的一帧。

克隆用的人声也来自这段源视频，会去掉静音和像噪声的片段。右侧声音面板默认用这段人声。在那里上传文件可以覆盖。**照片**只换脸。**照片 + 音频**用照片，再加上旁边那一段单独的音频。目标没有音轨时绝不做声音转换，即使源视频里有人声。

**资料库**是第四种。一份命名资料一直记着同一个人：继续加入的视频、照片或音频会融成一张脸和一段人声。图像·视频页负责新建、加入样本、移除样本和删除。摄像头页只选择一份资料。可以用这份资料来换，也可以继续一次性上传。资料存在 `library/`，不会提交进仓库。

## 摄像头

摄像头页用同样的选项，顺序也一样，末尾多一个已保存的资料。打开页面时选中 **视频（脸+声音，推荐）**。只显示当前选项需要的上传框。资料在图像·视频页管理。多张脸的对应在图像·视频页。

这一页的声音转换发生在摄像头开着、麦克风正在送音频、并且有参考音色的时候。有上传的音频就用上传的。否则从源视频里裁出人声。照片模式没有参考音色。照片 + 音频用照片旁边的那一段。上传的音频优先于从源视频抽出的人声。

没有麦克风，或没有参考音色，就只换脸。这时不会把麦克风播回去。干跑不做转换。如果没有安装 OpenVoice 转换器，预览继续换脸，声音那一行会写没有安装模型。

`barrett-faceswap live` 只采集摄像头，不读麦克风。一步换脸加换声用摄像头页。

处理文件时，只有下面几条同时成立才会跑：

- 声音开关是开的（或命令行传了 `--voice-reference`）
- 提供了参考音色（wav、mp3 或 m4a）
- 目标是视频，**并且**有音轨

目标没有音轨时，只换脸，不会加载声音模型。静态图、只预览第一帧、以及 `--dry-run` 也只换脸。干跑时如果原来有音轨，就保留它。

Gradio 开关**一开始是关的**。选中一个带音轨的目标视频后，开关会**打开**。静音视频或静态图会把它**关掉**。`configs/default.yaml` 里的 `voice.enabled` 控制的是这个自动打开。设成 `false` 时，开关会一直关着，直到你自己点开。声音面板在大预览的右侧，源脸、目标和预设仍留在左边。

## 引擎

引擎是 **OpenVoice V2 的音色转换器**（MyShell）。代码和 V2 权重是 MIT。它拿目标视频里原来的说话内容，加上一段短参考音频，只换说话人音色。不需要按说话人训练，也不需要付费 API。

在这台机器上没有选另外两个本地方案：

- **Seed-VC** 的零样本转换效果好，但协议是 GPL-3.0，仓库已归档，依赖锁在 PyTorch 2.4。那份构建没有 RTX 5090（`sm_120`）的核。
- **CosyVoice** 的 `inference_vc` 是真正的声音转换，代码是 Apache-2.0，但安装锁在较旧的 CUDA 12.1 torch，还要 ModelScope、wetext 和 Matcha。这和本仓库的 onnxruntime-gpu CUDA 13 栈冲突。
- OpenVoice 的转换器是普通 PyTorch。RTX 5090 上请安装 **CUDA 12.8 或更新** 的 PyTorch 轮子，让轮子带上 `sm_120`。`voice.device: auto` 在这份构建能看到 GPU 时用 CUDA，否则用 CPU。

本仓库的换脸权重仍然是非商业用途。OpenVoice 的检查点本身是 MIT。本仓库不会下载它。

`voice.watermark` 保持 **false**。OpenVoice 可以写入自己的水印，那需要另外的 `wavmark` 包。关掉它，这样合规水印仍是唯一的标识。合规本身默认打开。

## 安装

不要安装 OpenVoice 自带的 `requirements.txt`。它把 `numpy` 锁在 1.22，把 `gradio` 锁在 3.48，会弄坏这个项目。

在仓库根目录，如果 PyTorch 已经和 GPU 匹配：

```bash
pip install -e ".[voice]"
pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
```

RTX 5090（Blackwell，`sm_120`），驱动要支持 CUDA 12.8：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[voice]"
pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
```

如果第二条命令把 CUDA 版 torch 换掉了，再装一次 `cu128` 轮子。可以用下面这行确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

5090 上算力应当是 `(12, 0)`。更旧的轮子会警告不支持 `sm_120`，真正计算时失败。

### 权重

转换器是两个文件。Hugging Face 上的 `myshell-ai/OpenVoiceV2` 是 MIT。这条路径不需要基础说话人嵌入。

```bash
pip install -U huggingface_hub
huggingface-cli download myshell-ai/OpenVoiceV2 \
  converter/config.json converter/checkpoint.pth \
  --local-dir models/openvoice_v2
```

目录应当是：

```text
models/openvoice_v2/converter/config.json
models/openvoice_v2/converter/checkpoint.pth
```

`voice.checkpoint_dir` 指向 `models/openvoice_v2`。文件缺失时，会在换脸开始之前给出安装提示，而且只在这次确实要做声音转换时才检查。

## 控件

| 设置 | 含义 |
| --- | --- |
| 克隆强度 | `1` 只用转换后的声音。`0` 保留原音轨，并且不加载模型。中间值会在两段声音都对齐到画面之后再混合。 |
| 随机度（tau） | OpenVoice 的后验噪声。`0.3` 是上游默认。更高更随机，不是“克隆更强”。音色替换靠的是参考说话人嵌入。 |

高级默认收起。进度先走换脸帧（会做声音转换时，大约到 85%），然后是“声音转换”，再是“混流”。

时长：差距大约在 15% 以内时做时间拉伸，让词还靠近嘴型。差距更大就补静音或裁掉多余部分，避免声音严重变调。画面不会被音轨裁短。重新混流时复制视频流，音频用 AAC，并带上 `+faststart`，和帧编码器一致。

## 命令行

```bash
barrett-faceswap swap \
  --source path/to/authorized-source.mp4 \
  --target path/to/authorized-clip.mp4 \
  --output outputs/clip.mp4
```

源是视频时会打开声音转换，参考音色是从这段视频里截出的人声。`--voice-reference` 会换掉这段人声。`--no-voice` 强制只换脸。源是静态图时，除非传了 `--voice-reference`，否则不做声音转换。`--library 名字` 用已保存的资料代替 `--source`。融合后的脸就是身份，保存的人声就是参考音色，除非另外传了 `--voice-reference` 或 `--no-voice`。`--voice-strength` 和 `--voice-tau` 覆盖这一次的配置。目标没有音轨时，两种情况都只换脸。
