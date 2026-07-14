# CLAUDE.md — evo-math: 用进化算法提升 Llama 3.2 1B 数学能力

本仓库包含四个共享同一评估基础设施的实验track。**任何时候只实现当前被指派的 milestone，不要提前实现后续 track 的代码。**

## 项目结构（目标状态）

```
evo-math/
├── CLAUDE.md
├── DESIGN.md                  # 研究设计文档，动手前必读
├── pyproject.toml
├── configs/                   # 每个实验一个 yaml
├── data/
│   ├── eval_indices.json      # 固定的 GSM8K dev 子集索引（500 题，一经生成永不改动）
│   └── heldout_indices.json   # 最终报告用 held-out 集索引（与 dev 不相交）
├── src/evomath/
│   ├── harness/               # M0: 评估基础设施（所有 track 共享）
│   │   ├── model.py           # 模型加载、LoRA/norm-patch 注入
│   │   ├── generate.py        # batched greedy decoding
│   │   ├── parse.py           # GSM8K 答案解析
│   │   ├── fitness.py         # accuracy fitness + loglik proxy fitness
│   │   └── cache.py           # fitness 缓存（个体参数 hash → 分数）
│   ├── track1_pbt_lora/       # PBT-LoRA
│   ├── track2_curriculum/     # 数据生成器进化
│   ├── track3_norm_es/        # RMSNorm-ES
│   ├── track4_map_elites/     # MAP-Elites
│   └── common/                # seed 管理、W&B 封装、checkpoint IO
├── scripts/                   # 每个实验一个入口脚本
└── tests/
```

## 环境

- GPU: NVIDIA RTX 5080 16GB（**Blackwell 架构，必须 PyTorch 2.7+ / CUDA 12.8 wheel**；
  装错版本的症状是 `no kernel image is available`）
- 主机: iailab71（远程，通过 SSH 使用；长任务一律 tmux，不要前台阻塞）
- Python 3.11+，包管理用 `uv`
- 模型: `meta-llama/Llama-3.2-1B-Instruct`（2026-06-10 用户拍板从 base 改为
  Instruct：M0 实测 base 在 GSM8K 仅 ~5%，25–42% 验收区间对应 Instruct ~34–36%；
  本地缓存路径见 `configs/base.yaml` 的 `model_path`，不要硬编码）
- 推理统一 bf16。fitness 评估阶段模型权重冻结、`torch.no_grad()`
- 关键依赖: `transformers`, `peft`, `cma`, `datasets`, `wandb`, `pytest`
- 16GB 显存约束: 1B bf16 权重约 2.5GB，留足 KV cache 余量；
  generation batch size 从 32 起步，OOM 就减半，**不要**为省显存改用 fp16（数值口径要统一）

## 全局设计约束（违反任何一条 = 实现错误）

1. **fitness 评估的确定性**: 同一个体 + 同一 seed，两次评估结果必须 bit-level 一致。
   greedy decoding（`do_sample=False`），eval 题目固定为 `data/eval_indices.json`，
   题目顺序固定，prompt 模板固定（见 `harness/generate.py` 顶部常量，所有 track 共用）。
2. **dev / held-out 隔离**: 进化过程中任何代码不得读取 `heldout_indices.json`。
   held-out 评估只存在于 `scripts/final_eval.py`，且该脚本拒绝在实验目录内被 import。
3. **个体 = 参数 delta，不是模型副本**: LoRA 个体存 `(A, B)` 张量字典，
   norm 个体存 gain 向量的 delta。任何时刻 GPU 上只有一份 base model，
   评估个体时 patch in / patch out。磁盘上也只存 delta。
4. **两级 fitness**:
   - `fitness_loglik(individual)`: 对标准 CoT 解答的平均 token log-likelihood，
     单次 forward pass，便宜、连续、低噪 → 进化内循环用这个
   - `fitness_acc(individual)`: greedy 生成 + 答案解析的准确率，贵 →
     每 K 代对当前精英评估一次，以及 PBT 的 exploit 决策用
   两者的相关性验证是 M0 的验收项之一。
5. **可复现性**: 每个实验入口接受 `--seed`，seed 控制 EA 的全部随机性
   （初始化、变异、采样、数据 shard 划分）。W&B run name 含 seed 和 git short hash。
6. **所有入口脚本必须支持 `--smoke-test`**: population≤2、eval 题目=10、
   generation/迭代=1，全程 < 3 分钟，跑完打印 "SMOKE OK"。
   你（Claude Code）自验证只用 smoke-test，**永远不要发起真实长跑**——
   长跑由用户自己在 tmux 里挂。
7. **W&B logging**: project=`evo-math`，每代记录 best/mean/std fitness、
   评估耗时、缓存命中率；track 特有指标见 DESIGN.md。
8. **失败要响**: 答案解析失败率 > 15% 时 fitness 函数必须 raise 而不是静默给 0 分
   （高解析失败率几乎总意味着 prompt 模板或解析器 bug，不是模型真的烂）。

## GSM8K 答案解析规范（高频 bug 区，严格遵守）

- 标准答案: 取 `####` 后的内容，去逗号、去 `$`、strip
- 模型输出: 取生成文本中**最后一个**数值 token 串（正则匹配
  `-?[\d,]*\.?\d+`，去逗号后转 float 比较，容差 1e-4）
- 生成上限 512 new tokens，遇到下一个 `Q:` 截断（base model 会续写新题）
- few-shot prompt: 固定 4-shot，shot 内容硬编码为常量，永不改动
- `tests/test_parse.py` 必须覆盖: 含逗号大数、美元符号、小数、负数、
  答案后还有废话、完全无数字（应判错而非崩溃）

## Milestone 与验收标准

按顺序做，每个 milestone 单独 commit，验收通过前不进入下一个。

- **M0 — harness**:
  - base model 在 500 题 dev 子集上 4-shot greedy 准确率落在 25–42% 区间
    （对齐 Llama 3.2 1B base 的公开报告水平；明显偏离 = 先查解析器）
  - 同 seed 两次评估准确率完全一致
  - loglik proxy 与 accuracy 的相关性检查: 对 base model + 8 个随机 LoRA 扰动个体
    各算两种 fitness，Spearman ρ 报告进 W&B（预期 > 0.5；若不达标，停下来汇报，
    不要自行换 proxy）
  - fitness 缓存: 同一个体第二次评估走缓存，耗时 < 1s
- **M1 — track3 (RMSNorm-ES)**: 最简单，先做。验收: smoke 通过；
  CMA-ES 在 loglik fitness 上 20 代内单调改善（小规模真跑由用户执行）
- **M2 — track1 (PBT-LoRA)**: 验收: smoke 通过；TIES merge 单元测试
  （手算 3 参数小例子，含符号冲突 case）通过；等算力 baseline 脚本
  （单条 SGD、相同总步数）同时交付
- **M3 — track4 (MAP-Elites)**: 验收: smoke 通过；行为描述子在 base model +
  随机扰动个体上的分布有区分度（不塌缩到单一 cell）；档案可视化脚本交付
- **M4 — track2 (curriculum evolution)**: 最后做，依赖 M2 的 LoRA 训练循环。
  验收: smoke 通过；生成器沙箱测试通过（见 DESIGN.md 安全节）

## 工作方式要求

- 动手写代码前先输出实现计划等用户确认（plan 模式）
- 核心逻辑（解析器、merge 算子、ES 更新、MAP-Elites 网格）必须有 pytest 单测
- 不确定的设计决策: 先查 DESIGN.md，DESIGN.md 没写的就停下来问，不要自行发挥
- 禁止为通过验收而修改验收标准本身（包括 eval_indices.json）
