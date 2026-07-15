# 第二阶段实验隔离与稳定混合精度设计

## 目标

在不复制 `building_simplify` 源码、不改变现有 Transformer 默认结构的前提下，为 1:5000 第二阶段建立可复现的实验目录，并修复 FP16 AMP 训练出现 `NaN` 后仍继续运行的问题。

1:10000 暂不训练，但使用相同目录和配置格式预留位置。

## 阶段状态

- 第一阶段数据准备、矩形筛选、BPE 和洛杉矶 80/20 划分已经完成。
- 1:5000 当前 epoch 5 checkpoint 作为临时基线候选，验证集总体 `greedy_exact=0.26282932283537935`。
- epoch 8 及之后出现 `NaN`，不得作为有效 checkpoint。
- 1:10000 基线和两个比例尺的最终分组验收仍未完成，因此不得将整个第一阶段标记为全部完成。

## 目录边界

`building_simplify/` 是唯一可导入源码。`experiments/` 只包含配置、服务器脚本、运行产物和来源清单，不创建 `building_simplify_phase2` 或其他源码副本。

```text
experiments/
  phase1/
    scale5000/
      README.md
      baseline.json
      source_manifest.json
      checkpoint_best_epoch5.pt       # 服务器 checkpoint，稍后人工放入
  phase2/
    README.md
    configs/
      scale5000_diagnostic.json
      scale5000_dropout0.json
      scale5000_preln.json
      scale5000_larger.json
    scripts/
      run_diagnostic_5000.sh
      run_dropout0_5000.sh
      run_preln_5000.sh
      run_larger_5000.sh
    runs/                              # 训练时生成，不存放源码
```

每个脚本显式指定配置和输出目录。实验之间不得复用输出目录，也不得隐式从其他实验 checkpoint 初始化。

## 混合精度策略

训练公开参数由布尔 `use_amp` 改为明确的精度模式：

- `auto`：CUDA 且 `torch.cuda.is_bf16_supported()` 时使用 BF16；否则使用 FP32。
- `bf16`：要求 CUDA 和 BF16 支持，不满足时立即报错。
- `fp32`：关闭 autocast。
- 不再提供 FP16 训练模式，避免重复当前数值崩溃。

BF16 autocast 不使用 GradScaler。模型参数和优化器状态继续保持 FP32。

CLI 使用 `--precision {auto,bf16,fp32}`。为兼容现有服务器命令，`--no-amp` 暂时保留并映射到 `fp32`，但不得与非默认 `--precision` 同时使用。

## 非有限值保护

每个 optimizer step 检查：

1. loss 是否为有限值；
2. 梯度裁剪返回的总范数是否为有限值；
3. optimizer 更新后参数是否仍为有限值。

首次出现 `NaN` 或 `Inf` 时：

- 不执行后续训练；
- 写入 `failure.json`，包括 epoch、global step、batch 内最大源/目标长度、精度模式、学习率和失败阶段；
- 保存 `checkpoint_last_finite.pt`；
- 以非零退出码结束，禁止生成看似正常的后续 epoch 日志。

为了避免每步扫描全部参数造成明显开销，更新后参数检查只在 checkpoint 保存点和 epoch 结束时执行；loss 与梯度范数每步检查。

## Checkpoint 完整性

checkpoint 除现有模型和优化器状态外，还必须保存：

- warmup/cosine scheduler 状态；
- 当前精度模式；
- Python 和 PyTorch CPU/CUDA 随机状态；
- 完整训练配置，包括 batch size、学习率、weight decay、scheduled sampling 和模型参数；
- 当前 epoch、global step 和最佳指标元数据。

恢复训练时必须恢复上述状态。旧 checkpoint 缺少这些字段时允许只用于推理；若用于续训则明确警告其调度器和随机状态不完整。

## 第二阶段实验顺序

所有实验只使用 1:5000 洛杉矶数据选择模型，纽约数据不参与。

1. `diagnostic`：固定 512 条复杂建筑，dropout=0、weight decay=0、scheduled sampling=0，最多 3000 steps。通过标准为 teacher-forced token accuracy >=99.5% 且训练集 greedy exact >=98%。
2. `dropout0`：只有 diagnostic 通过后才执行；基于第一阶段结构，只把 dropout 改为 0。
3. `preln`：只有 dropout0 未达到预期时执行；相对同一对照只启用 Pre-LN。
4. `larger`：只有 Pre-LN 仍不足时执行；使用 d_model=384、layers=6、heads=8、FFN=1536。

脚本不会自动串行启动下一实验，避免未满足条件时误跑后续实验。

## 配置与记录

JSON 配置包含数据路径、输出目录、模型参数、训练参数、精度模式、种子和预期前置条件。启动时把最终解析后的配置写入运行目录，checkpoint 和 metrics 都引用其 SHA-256。

阶段一目录只登记 checkpoint 的预期文件名、原服务器路径、日志指标和 SHA-256。由于本地当前只有日志，没有实际 checkpoint，代码不得伪造 checkpoint 文件或哈希。

## 验证标准

- `--precision auto` 在 RTX 5090 上解析为 BF16，GradScaler 关闭。
- CPU 或不支持 BF16 的 CUDA 环境下，`auto` 解析为 FP32。
- 非有限 loss 或梯度会在当步终止并写 `failure.json`。
- 新 checkpoint 可以恢复模型、优化器、scheduler、精度和随机状态。
- 四个第二阶段配置一次只改变计划允许的变量。
- 全部现有测试通过，并增加精度解析、非有限值终止和 checkpoint 字段测试。

## 非目标

- 本次不执行正式训练。
- 本次不运行 1:10000。
- 本次不修改 Pre-LN 默认值、模型容量或 token 分辨率。
- 本次不复制或删除原始数据、prepared datasets 或服务器 checkpoint。
