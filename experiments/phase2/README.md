# 第二阶段 1:5000 实验

该目录不包含 Python 源码。所有脚本调用根目录唯一的 `building_simplify` 包，运行结果写入被 Git 忽略的 `experiments/phase2/runs/`。

严格执行顺序：

1. `scripts/run_diagnostic_5000.sh`
2. 诊断达到 token accuracy >=99.5% 且 greedy exact >=98% 后，创建 `experiments/phase2/runs/scale5000/diagnostic/diagnostic-passed`。
3. `scripts/run_dropout0_5000.sh`
4. 人工比较第一阶段基线后创建相应 `*-reviewed` 标记，再决定是否运行 Pre-LN 和 larger。

RTX 5090 使用 `--precision bf16`。脚本发现缺少前置审核标记时会退出，不会自动串联后续实验。
