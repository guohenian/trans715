# 第二阶段实验隔离实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为唯一训练源码增加稳定 BF16/FP32 精度策略、非有限值立即终止、可完整恢复的 checkpoint，并建立隔离的第二阶段 1:5000 实验目录。

**Architecture:** `building_simplify/train.py` 负责精度解析、训练安全和状态持久化；`pipeline.py` 仅暴露统一 CLI。`experiments/phase2` 不包含 Python 包，只通过配置和 shell 脚本调用统一入口。

**Tech Stack:** Python 3.12、PyTorch、pytest、JSON、Bash。

## Global Constraints

- `building_simplify/` 是唯一可导入源码。
- `auto` 在支持 BF16 的 CUDA 上使用 BF16，否则使用 FP32；不再使用 FP16。
- 第一次发现非有限 loss 或梯度时必须立即停止并写 `failure.json`。
- 第二阶段顺序固定为 diagnostic、dropout0、Pre-LN、larger。
- 本次不训练模型、不改变 Transformer 默认结构、不修改 prepared datasets。

---

### Task 1: 明确精度模式

**Files:**
- Modify: `building_simplify/train.py`
- Modify: `building_simplify/pipeline.py`
- Test: `tests/test_training_precision.py`

**Interfaces:**
- Produces: `resolve_precision_mode(requested: str, device: str) -> PrecisionPolicy`。

- [ ] 写失败测试：CPU `auto -> fp32`、支持 BF16 的 CUDA `auto -> bf16`、不支持时 `bf16` 报错。
- [ ] 运行 `python -m pytest tests/test_training_precision.py -q`，确认因接口缺失而失败。
- [ ] 添加不可变 `PrecisionPolicy`，由其提供 `autocast_enabled` 和 `autocast_dtype`；把 CLI 改为 `--precision {auto,bf16,fp32}` 并保留 `--no-amp` 兼容映射。
- [ ] 再次运行测试并确认通过。

### Task 2: 非有限值立即终止

**Files:**
- Modify: `building_simplify/train.py`
- Test: `tests/test_training_precision.py`

**Interfaces:**
- Produces: `NonFiniteTrainingError`、`ensure_finite_tensor(...)` 和 `failure.json` 格式。

- [ ] 写失败测试，向有限值检查传入 NaN，要求写出包含 `failure_stage`、epoch、global_step 和长度信息的报告并抛错。
- [ ] 运行目标测试，确认失败。
- [ ] 在 backward 前检查 loss，在梯度裁剪后检查梯度范数；捕获失败后保存最后正常 checkpoint、写原子 `failure.json` 并终止。
- [ ] 运行目标测试并确认通过。

### Task 3: 完整 checkpoint 和恢复

**Files:**
- Modify: `building_simplify/train.py`
- Modify: `tests/test_training_smoke.py`

**Interfaces:**
- Checkpoint produces: `scheduler`, `precision`, `training_config`, `random_state` 和现有字段。

- [ ] 扩展 smoke test，断言 checkpoint 包含精度、两个 scheduler、随机状态和完整训练配置。
- [ ] 运行 smoke test，确认旧 checkpoint 格式不满足断言。
- [ ] 保存与恢复 scheduler、Python/PyTorch/CUDA RNG、precision 和训练配置；旧 checkpoint 缺字段时输出明确警告。
- [ ] 运行 smoke test 并确认通过。

### Task 4: 第二阶段隔离目录

**Files:**
- Create: `experiments/phase1/scale5000/README.md`
- Create: `experiments/phase1/scale5000/baseline.json`
- Create: `experiments/phase1/scale5000/source_manifest.json`
- Create: `experiments/phase2/README.md`
- Create: `experiments/phase2/configs/*.json`
- Create: `experiments/phase2/scripts/*.sh`
- Modify: `building_simplify/pipeline.py`
- Test: `tests/test_phase2_experiments.py`

**Interfaces:**
- Pipeline consumes: `--config PATH`，CLI 显式参数覆盖配置文件值。

- [ ] 写失败测试，验证四份配置存在、只改变允许变量、脚本顺序与输出目录互不重叠。
- [ ] 运行测试确认失败。
- [ ] 添加 JSON 配置加载，生成 diagnostic/dropout0/preln/larger 配置和服务器脚本；脚本只运行一个实验且检查前置结果文件。
- [ ] 运行测试并确认通过。

### Task 5: 全量验证和文档

**Files:**
- Modify: `SERVER_WORKFLOW.md`
- Modify: `experiments/phase2/README.md`

- [ ] 运行 `python -m pytest -q`，期望全部通过。
- [ ] 运行 `python -m compileall -q building_simplify tests`，期望退出码 0。
- [ ] 运行四个命令的 `--help`/配置解析检查，不启动训练。
- [ ] 检查 `git diff --check`、忽略 checkpoint/runs、确认没有数据文件进入 Git。
- [ ] 提交实现，提交信息为 `feat: add isolated phase two training workflow`。
