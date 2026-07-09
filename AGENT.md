# AGENT.md

## 目标

在本仓库中复现论文 `No Time Like the Present: Agentic Test-Time Training for LLM Agents` 的最小可运行版本，优先保证：

1. 工程结构清晰。
2. 服务器端代码全部放在 `/root/autodl-tmp` 下。
3. 本地工作区保留一份同步代码镜像。
4. 先完成可验证的 smoke test，再逐步接入真实 benchmark。

## 约束

- 真实模型下载、缓存、数据、日志都必须限制在 `/root/autodl-tmp` 内。
- 不要把大文件、模型缓存或数据写到系统默认缓存目录以外的位置。
- 代码变更优先使用 `apply_patch`。
- 如果需要联网下载，先在远端 shell 里执行 `source /etc/network_turbo`。

## 当前复现范围

先做最小集：

- baseline: ReAct / no-TTT
- method: aTTT
- 默认模型配置: 论文中的 `Qwen3.5-9B`
- smoke test: 小样本或玩具 episode，验证 repetition weighting 逻辑和运行链路

后续再扩展到：

- ALFWorld
- SWE-bench Lite
- 论文其余 baselines 和 ablations

