# 商业路线

`--commercial` 把 `execution.commercial` 设为真。门禁会拒绝这次运行将打开、且注册表行的 `commercial_use` 为 false 或缺失的每一个角色。内置模型全部标成 false，因此商业构建不能加载 inswapper、可选换脸器、GFPGAN、CodeFormer、遮挡模型或解析模型。没有配置键能改写这些行。

可商业授权的换脸器需要你自己训练并登记。脚手架沿着 Apache-2.0 的 GHOST / AEI-Net 这一类身份条件生成器来说明步骤。它不复制它们的训练源码，也不下载它们的权重。

```bash
python scripts/train_swapper.py --dataset configs/dataset.example.yaml --check
python scripts/export_onnx.py --check
python scripts/eval_swapper.py --self-test
```

`configs/dataset.example.yaml` 是一份清单。集合和每一条样本的 `license` 都必须是字符串 `commercial-ok`。分辨率是 256 或 512。检查器不会打开图像路径，也不会去获取它们。

没有 `--check` 时，`train_swapper.py` 打印计划并以状态 2 退出。`export_onnx.py` 打印约定（NCHW RGB 归一化到 [-1, 1]，512 维隐向量，ArcFace 模板按 size/112 缩放），没有检查点时以状态 2 退出。评估是全局 SSIM，以及调用者已经算好的两个向量的余弦相似度。

法律复核之后，新增一行注册表并把 `commercial_use` 设为 true。门禁只读这个标记。
