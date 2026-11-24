# 评测基线快照

`eval/reports/` 为运行产物（gitignored）；本目录保存**纳入版本库的基线结果**，用于跨提交对比。

| 文件 | 内容 |
|---|---|
| `*_retrieval.json` | KB 多跳检索评测（mock）：Recall@k / MRR / 按 hop 分组 / 延迟 |
| `*_cohort.json` | 科研入组筛查评测：三值混淆矩阵 / accuracy / macro-F1 / UNKNOWN 漏判率 |

重跑：`python -m eval.run_eval retrieval --corpus main --mode mock` 等（见 README 第 7 步）。
