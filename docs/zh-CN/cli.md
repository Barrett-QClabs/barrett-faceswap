# 命令行参考

由 `scripts/build_cli_reference.py` 生成。不要手改。

```text
usage: barrett-faceswap [-h] [--version] [--lang LANG]
                        {swap,jobs,live,inspect-models,list-processors,ui,benchmark,doctor,download-models}
                        ...

对静态图像或视频换脸。开发默认关闭合规强制，避免挡住本地试跑。

positional arguments:
  {swap,jobs,live,inspect-models,list-processors,ui,benchmark,doctor,download-models}
    swap                做人脸检测、对齐、替换并贴回。目标若是视频，则逐帧处理。
    jobs                把换脸放进队列，中断后可以续跑。
    live                运行实时摄像头循环。源脸向量只计算一次。
    inspect-models      查看模型注册表、许可标记以及本地文件的哈希。
    list-processors     列出已注册的帧处理器。
    ui                  启动 Gradio 界面。
    benchmark           为处理器链计时，并打印每秒帧数。
    doctor              报告执行提供者、CUDA 与 cuDNN 库，以及模型文件。
    download-models     下载 buffalo_l 和 inswapper_128，校验 sha256，并在镜像之间重试。

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --lang LANG           界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap swap [-h] --source SOURCE --target TARGET --output OUTPUT
                             [--config CONFIG] [--dry-run] [--enable-compliance] [--lang LANG]
                             [--consent-file CONSENT_FILE] [--acknowledge-authorized-use]
                             [--consent-subject CONSENT_SUBJECT]
                             [--consent-statement CONSENT_STATEMENT] [--no-watermark]
                             [--disable-content-filter]
                             [--content-filter-provider {none,stub,nudenet,custom}]
                             [--content-filter-threshold CONTENT_FILTER_THRESHOLD]
                             [--no-block-on-minor] [--processors PROCESSORS]
                             [--execution-provider EXECUTION_PROVIDER]
                             [--target-face-index TARGET_FACE_INDEX] [--trim-start TRIM_START]
                             [--trim-end TRIM_END] [--video-encoder {libx264,h264_nvenc}]
                             [--video-threads VIDEO_THREADS] [--no-audio]
                             [--filter-every FILTER_EVERY]
                             [--enhancer-model {gfpgan_1.4,codeformer}]
                             [--enhancer-blend ENHANCER_BLEND] [--enhancer-weight ENHANCER_WEIGHT]
                             [--swapper-model {inswapper_128,hyperswap_1a_256,hyperswap_1b_256,hyperswap_1c_256,simswap_256,simswap_unofficial_512,ghost_1_256,ghost_2_256,ghost_3_256}]
                             [--pixel-boost PIXEL_BOOST] [--many-faces] [--reference REFERENCE]
                             [--face-map FACE_MAP] [--blend-mode {alpha,poisson}]
                             [--color-transfer {none,mean_std,lab,histogram}]
                             [--video-mode {disk,pipe}] [--eye-mode {swap,blend,target}]
                             [--with-enhancer] [--job-id JOB_ID] [--resume] [--commercial]

options:
  -h, --help            show this help message and exit
  --source SOURCE       源图像。人脸身份来自这个文件。
  --target TARGET       目标图像或视频。这个文件里的人脸会被替换。
  --output OUTPUT       图像或视频的写出路径。
  --config CONFIG       YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --dry-run             跳过 ONNX 推理。权重还没下载时可以用它走通命令。
  --enable-compliance   为本轮打开同意校验、水印、内容过滤和审计。
  --lang LANG           界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。
  --consent-file CONSENT_FILE
                        源人脸的 JSON 同意清单。
  --acknowledge-authorized-use
                        声明你已获授权使用源人脸。
  --consent-subject CONSENT_SUBJECT
                        与 --acknowledge-authorized-use 一起记录的主体编号。
  --consent-statement CONSENT_STATEMENT
                        可选的确认语句。省略时会记入一句默认声明。
  --no-watermark        跳过可见标识和内嵌元数据。这次跳过会写入审计日志。
  --disable-content-filter
                        关闭内容过滤。未打开权限开关时会被拒绝。
  --content-filter-provider {none,stub,nudenet,custom}
                        覆盖本轮的 content_filter.provider。更改提供者需要权限开关。
  --content-filter-threshold CONTENT_FILTER_THRESHOLD
                        覆盖过滤分数阈值。
  --no-block-on-minor   不要因为未成年人信号而拦截。需要权限开关。
  --processors PROCESSORS
                        逗号分隔的处理器名称。默认使用配置里的列表。
  --execution-provider EXECUTION_PROVIDER
                        优先使用的执行提供者。可重复该参数以指定顺序。列表中包含 CUDA 时会优先 CUDA。
  --target-face-index TARGET_FACE_INDEX
                        要替换的目标人脸序号。0 表示最大的一张脸。
  --trim-start TRIM_START
                        视频裁剪起点，单位秒。目标必须是视频。
  --trim-end TRIM_END   视频裁剪终点，单位秒。目标必须是视频。
  --video-encoder {libx264,h264_nvenc}
                        视频编码器：libx264 或 h264_nvenc。
  --video-threads VIDEO_THREADS
                        同时处理的帧数。
  --no-audio            丢弃源音轨，不再混回输出。
  --filter-every FILTER_EVERY
                        合规打开时，每隔 N 帧过滤一次。最后一帧总会被检查。
  --enhancer-model {gfpgan_1.4,codeformer}
                        增强模型：gfpgan_1.4 或 codeformer。
  --enhancer-blend ENHANCER_BLEND
                        增强混合系数，0 到 1。1 表示用修复结果替换裁剪。
  --enhancer-weight ENHANCER_WEIGHT
                        CodeFormer 权重，0 到 1。图里没有该输入时会被忽略。
  --swapper-model {inswapper_128,hyperswap_1a_256,hyperswap_1b_256,hyperswap_1c_256,simswap_256,simswap_unofficial_512,ghost_1_256,ghost_2_256,ghost_3_256}
                        换脸图。默认是 inswapper_128。其他名称需要对应的 ONNX 文件。
  --pixel-boost PIXEL_BOOST
                        对齐裁剪的边长。0 表示模型原生尺寸（inswapper 则用 swap.resolution）。
  --many-faces          用源身份替换每一张检测到的脸。
  --reference REFERENCE
                        参考静态图。替换与这张身份最接近的目标脸。
  --face-map FACE_MAP   JSON 文件，把源图像映射到目标序号或参考图。
  --blend-mode {alpha,poisson}
                        贴回方式：alpha 或 poisson。
  --color-transfer {none,mean_std,lab,histogram}
                        色彩匹配：none、mean_std、lab 或 histogram。
  --video-mode {disk,pipe}
                        视频路径：disk（默认）或 pipe。
  --eye-mode {swap,blend,target}
                        眼部处理：swap、blend（默认，尽量保留目标眼睛）或 target。
  --with-enhancer       同时运行 face_enhancer。眼部使用 enhancer.eye_blend。
  --job-id JOB_ID       任务编号。字母、数字、下划线和连字符。
  --resume              从上次完成的帧继续任务。
  --commercial          拒绝注册表里 commercial_use 为 false 的模型。

usage: barrett-faceswap jobs [-h] [--config CONFIG] [--lang LANG] {submit,resume,list,run} ...

positional arguments:
  {submit,resume,list,run}
    submit              只保存任务，先不运行。
    resume              从下一帧运行已保存的任务。
    list                列出任务编号和状态。
    run                 运行最早的待处理或中断任务。

options:
  -h, --help            show this help message and exit
  --config CONFIG       YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG           界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap jobs submit [-h] --source SOURCE --target TARGET --output OUTPUT
                                    [--config CONFIG] [--lang LANG] [--dry-run]
                                    [--processors PROCESSORS]
                                    [--execution-provider EXECUTION_PROVIDER]
                                    [--swapper-model {inswapper_128,hyperswap_1a_256,hyperswap_1b_256,hyperswap_1c_256,simswap_256,simswap_unofficial_512,ghost_1_256,ghost_2_256,ghost_3_256}]
                                    [--pixel-boost PIXEL_BOOST] [--many-faces]
                                    [--reference REFERENCE] [--face-map FACE_MAP]
                                    [--blend-mode {alpha,poisson}]
                                    [--color-transfer {none,mean_std,lab,histogram}]
                                    [--video-mode {disk,pipe}] [--eye-mode {swap,blend,target}]
                                    [--with-enhancer] [--job-id JOB_ID]

options:
  -h, --help            show this help message and exit
  --source SOURCE       源图像。人脸身份来自这个文件。
  --target TARGET       目标图像或视频。这个文件里的人脸会被替换。
  --output OUTPUT       图像或视频的写出路径。
  --config CONFIG       YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG           界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。
  --dry-run             跳过 ONNX 推理。权重还没下载时可以用它走通命令。
  --processors PROCESSORS
                        逗号分隔的处理器名称。默认使用配置里的列表。
  --execution-provider EXECUTION_PROVIDER
                        优先使用的执行提供者。可重复该参数以指定顺序。列表中包含 CUDA 时会优先 CUDA。
  --swapper-model {inswapper_128,hyperswap_1a_256,hyperswap_1b_256,hyperswap_1c_256,simswap_256,simswap_unofficial_512,ghost_1_256,ghost_2_256,ghost_3_256}
                        换脸图。默认是 inswapper_128。其他名称需要对应的 ONNX 文件。
  --pixel-boost PIXEL_BOOST
                        对齐裁剪的边长。0 表示模型原生尺寸（inswapper 则用 swap.resolution）。
  --many-faces          用源身份替换每一张检测到的脸。
  --reference REFERENCE
                        参考静态图。替换与这张身份最接近的目标脸。
  --face-map FACE_MAP   JSON 文件，把源图像映射到目标序号或参考图。
  --blend-mode {alpha,poisson}
                        贴回方式：alpha 或 poisson。
  --color-transfer {none,mean_std,lab,histogram}
                        色彩匹配：none、mean_std、lab 或 histogram。
  --video-mode {disk,pipe}
                        视频路径：disk（默认）或 pipe。
  --eye-mode {swap,blend,target}
                        眼部处理：swap、blend（默认，尽量保留目标眼睛）或 target。
  --with-enhancer       同时运行 face_enhancer。眼部使用 enhancer.eye_blend。
  --job-id JOB_ID       任务编号。字母、数字、下划线和连字符。

usage: barrett-faceswap jobs resume [-h] [--config CONFIG] [--lang LANG] job_id

positional arguments:
  job_id           任务编号。字母、数字、下划线和连字符。

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG      界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap jobs list [-h] [--config CONFIG] [--lang LANG]

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG      界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap jobs run [-h] [--config CONFIG] [--lang LANG]

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG      界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap live [-h] --source SOURCE [--config CONFIG] [--lang LANG]
                             [--camera CAMERA] [--frames FRAMES] [--no-preview]
                             [--virtual-camera VIRTUAL_CAMERA] [--dry-run] [--enable-compliance]
                             [--consent-file CONSENT_FILE] [--acknowledge-authorized-use]
                             [--consent-subject CONSENT_SUBJECT]
                             [--consent-statement CONSENT_STATEMENT] [--no-watermark]
                             [--commercial] [--execution-provider EXECUTION_PROVIDER]
                             [--swapper-model {inswapper_128,hyperswap_1a_256,hyperswap_1b_256,hyperswap_1c_256,simswap_256,simswap_unofficial_512,ghost_1_256,ghost_2_256,ghost_3_256}]
                             [--pixel-boost PIXEL_BOOST] [--many-faces] [--reference REFERENCE]
                             [--face-map FACE_MAP] [--blend-mode {alpha,poisson}]
                             [--color-transfer {none,mean_std,lab,histogram}]
                             [--video-mode {disk,pipe}] [--eye-mode {swap,blend,target}]
                             [--with-enhancer]

options:
  -h, --help            show this help message and exit
  --source SOURCE       源图像。人脸身份来自这个文件。
  --config CONFIG       YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG           界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。
  --camera CAMERA       OpenCV 摄像头序号。默认是 live.camera。
  --frames FRAMES       处理这么多帧后停止。0 表示直到摄像头关闭。
  --no-preview          不打开预览窗口。
  --virtual-camera VIRTUAL_CAMERA
                        可选的 v4l2loopback 设备，例如 /dev/video2。
  --dry-run             跳过 ONNX 推理。权重还没下载时可以用它走通命令。
  --enable-compliance   为本轮打开同意校验、水印、内容过滤和审计。
  --consent-file CONSENT_FILE
                        源人脸的 JSON 同意清单。
  --acknowledge-authorized-use
                        声明你已获授权使用源人脸。
  --consent-subject CONSENT_SUBJECT
                        与 --acknowledge-authorized-use 一起记录的主体编号。
  --consent-statement CONSENT_STATEMENT
                        可选的确认语句。省略时会记入一句默认声明。
  --no-watermark        跳过可见标识和内嵌元数据。这次跳过会写入审计日志。
  --commercial          拒绝注册表里 commercial_use 为 false 的模型。
  --execution-provider EXECUTION_PROVIDER
                        优先使用的执行提供者。可重复该参数以指定顺序。列表中包含 CUDA 时会优先 CUDA。
  --swapper-model {inswapper_128,hyperswap_1a_256,hyperswap_1b_256,hyperswap_1c_256,simswap_256,simswap_unofficial_512,ghost_1_256,ghost_2_256,ghost_3_256}
                        换脸图。默认是 inswapper_128。其他名称需要对应的 ONNX 文件。
  --pixel-boost PIXEL_BOOST
                        对齐裁剪的边长。0 表示模型原生尺寸（inswapper 则用 swap.resolution）。
  --many-faces          用源身份替换每一张检测到的脸。
  --reference REFERENCE
                        参考静态图。替换与这张身份最接近的目标脸。
  --face-map FACE_MAP   JSON 文件，把源图像映射到目标序号或参考图。
  --blend-mode {alpha,poisson}
                        贴回方式：alpha 或 poisson。
  --color-transfer {none,mean_std,lab,histogram}
                        色彩匹配：none、mean_std、lab 或 histogram。
  --video-mode {disk,pipe}
                        视频路径：disk（默认）或 pipe。
  --eye-mode {swap,blend,target}
                        眼部处理：swap、blend（默认，尽量保留目标眼睛）或 target。
  --with-enhancer       同时运行 face_enhancer。眼部使用 enhancer.eye_blend。

usage: barrett-faceswap inspect-models [-h] [--config CONFIG] [--lang LANG]

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG      界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap list-processors [-h] [--lang LANG]

options:
  -h, --help   show this help message and exit
  --lang LANG  界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap ui [-h] [--config CONFIG] [--lang LANG] [--host HOST] [--port PORT]
                           [--share]

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG      界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。
  --host HOST      Gradio 服务监听的地址。
  --port PORT      Gradio 服务端口。
  --share          向 Gradio 申请一个临时的公开链接。

usage: barrett-faceswap benchmark [-h] [--config CONFIG] [--lang LANG] [--frames FRAMES]
                                  [--width WIDTH] [--height HEIGHT] [--source SOURCE]
                                  [--target TARGET] [--dry-run]
                                  [--execution-provider EXECUTION_PROVIDER] [--live]

options:
  -h, --help            show this help message and exit
  --config CONFIG       YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG           界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。
  --frames FRAMES       用于计时的帧数。
  --width WIDTH         帧宽度。与 --source 和 --target 一起传入的静态图会缩放到这个宽度。
  --height HEIGHT       帧高度。与 --source 和 --target 一起传入的静态图会缩放到这个高度。
  --source SOURCE       作为源身份的静态图像。与 --target 一起使用时，基准测试会检测这张脸，而不是使用绘制的后备人脸。
  --target TARGET       按 --frames 重复的静态图像。需要和 --source 一起传入，检测才会跑在真实人脸上。
  --dry-run             跳过 ONNX 推理。权重还没下载时可以用它走通命令。
  --execution-provider EXECUTION_PROVIDER
                        优先使用的执行提供者。可重复该参数以指定顺序。列表中包含 CUDA 时会优先 CUDA。
  --live                为一次实时会话计时，并打印各阶段延迟。不会打开摄像头。

usage: barrett-faceswap doctor [-h] [--config CONFIG] [--lang LANG]

options:
  -h, --help       show this help message and exit
  --config CONFIG  YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG      界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。

usage: barrett-faceswap download-models [-h] [--config CONFIG] [--lang LANG] [--with-gfpgan]
                                        [--with-masks] [--force] [--from-dir FROM_DIR]

options:
  -h, --help           show this help message and exit
  --config CONFIG      YAML 配置路径。配置里的相对路径相对当前工作目录解析。
  --lang LANG          界面语言：en 或 zh-CN。优先于环境变量 BARRETT_FACESWAP_LANG 和 ui.language。
  --with-gfpgan        同时下载可选的 GFPGAN 1.4 ONNX。
  --with-masks         同时把 xseg_1.onnx 和 bisenet_resnet_34.onnx 下载到蒙版路径。这两份文件的许可未经核实，很可能不能用于商业用途。
  --force              即使本地文件已经匹配，也重新下载。
  --from-dir FROM_DIR  下载前先从这个目录复制匹配的 ONNX 文件。仍会校验 SHA-256。
```
