[English](README.md) | [中文](README.zh-CN.md)

# Contributing / 参与开发

The published default keeps responsible-use checks on. Local work can turn that master switch off without editing a tracked file.

随仓库发布的默认值会打开负责任使用检查。本地开发可以关掉这个总开关，而不必修改已被跟踪的文件。

## Turn compliance off locally / 在本地关闭合规

Do one of the following. Do not commit the result.

任选一种。不要把结果提交进仓库。

1. Environment, for one shell / 只对当前 shell 生效的环境变量:

```bash
export BARRETT_DEV_MODE=1
```

`BARRETT_DEV_MODE=1`, `true`, `yes`, or `on` turns `compliance.enabled` off.

`BARRETT_DEV_MODE=1`、`true`、`yes` 或 `on` 会把 `compliance.enabled` 关掉。

2. Environment, explicit / 显式环境变量:

```bash
export BARRETT_COMPLIANCE=off
```

`off`, `0`, `false`, or `no` forces off. `on`, `1`, `true`, or `yes` forces on.

`off`、`0`、`false` 或 `no` 强制关闭。`on`、`1`、`true` 或 `yes` 强制打开。

3. Gitignored overlay / 被 git 忽略的覆盖文件:

```bash
cp configs/local.example.yaml configs/local.yaml
```

`configs/local.yaml` is merged on top of `configs/default.yaml`. The example sets `compliance.enabled: false`.

`configs/local.yaml` 会盖在 `configs/default.yaml` 上面。示例把 `compliance.enabled` 设为 `false`。

## Precedence / 优先级

`BARRETT_COMPLIANCE` wins over `BARRETT_DEV_MODE`. Dev mode wins over `configs/local.yaml`. The overlay wins over the shipped file. When none of those are set, enforcement is on.

`BARRETT_COMPLIANCE` 优先于 `BARRETT_DEV_MODE`。开发模式优先于 `configs/local.yaml`。覆盖文件优先于随仓库提供的文件。三者都没设置时，强制是打开的。

There is no public CLI flag and no Gradio checkbox that turns a true config off. The Advanced compliance checkbox can only force a run on. `--no-watermark` is the logged way to skip the visible label while enforcement stays on.

没有公开的命令行参数，也没有 Gradio 复选框，可以把已经为 true 的配置关掉。高级设置里的合规复选框只能强制打开一轮。`--no-watermark` 是在强制仍然打开时跳过可见标识的方式，并且会被记入日志。

## What stays on / 打开时会做什么

Consent or an acknowledgement with a subject name, a visible `AI-generated` / `人工智能生成` label, embedded metadata, a content-filter pass on the source and the target, and one audit line. The shipped filter provider is `stub`. It allows the frame and logs that no detector is loaded. It is not an NSFW detector and it does not classify age. `content_filter.provider: nudenet` is the optional detector. NudeNet is not a dependency.

同意记录，或带授权人姓名的确认；可见的 `AI-generated` / `人工智能生成` 标识；内嵌元数据；对源和目标做一次内容过滤；以及一行审计。随仓库提供的过滤提供者是 `stub`。它放行该帧，并写明没有加载检测器。它不是 NSFW 检测器，也不判断年龄。`content_filter.provider: nudenet` 是可选的检测器。NudeNet 不是依赖。

## Tests / 测试

The test suite clears `BARRETT_DEV_MODE` and `BARRETT_COMPLIANCE`. A developer shell cannot turn the shipped-default assertions off.

测试会清掉 `BARRETT_DEV_MODE` 和 `BARRETT_COMPLIANCE`。开发者的 shell 不能把“随仓库默认打开”的断言关掉。

```bash
python3 -m pytest
python3 scripts/build_cli_reference.py --check
mkdocs build --strict
```
