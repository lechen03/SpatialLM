# SpatialLM × ScanNet EXP3 协议：官方 ScanNet-SFT 模型交叉评测（「尺子」判别实验）

> **本文档的主要读者是服务器上的 AI Agent。**
> 背景：EXP2（我方全量微调）macro F1@.25 = 0.4834，官方 README 报 0.656，差 17.3pt。官方用于训练和评测的 ScanNet GT **未公开**（已查证：GitHub 仓库、manycore-research 与一作 ysmao 的 HF 账号均无此数据集），该差距无法排除「评测 GT（尺子）不同」的成分。
> 本实验把**一作发布的官方 ScanNet 微调模型**拉到我们的 val 集上，用**我们的 GT** 评测——用一个零训练成本的对照把「模型能力差距」和「尺子差异」分开。

## 0. 已知信息

| 项 | 值 |
|---|---|
| 测试模型 | `ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT`（一作 Yongsen Mao 发布，2025-11-05 更新，推测即官方 ScanNet 基准模型或同系） |
| 评测数据 | `data/scannet_spatiallm/` 的 312 个 val 场景（EXP1/EXP2 现存产物） |
| 评测 GT | `data/scannet_spatiallm/layout/`（我方自动生成） |
| 对照数字 | EXP1 macro@.25/.50 = 0.4426/0.2836；EXP2 = 0.4834/0.3217；官方 = 0.656/0.526（macro 口径） |

环境沿用 EXP1/EXP2：conda env `spatiallm`、`export HF_HOME=/home/chenle/hf_home_spatiallm`（绕开用户损坏的 HF token）。

## 1. 判读逻辑（写报告时必须对照此表下结论）

| EXP3 macro@.25 结果 | 结论 | 下一步 |
|---|---|---|
| **≥ 0.60** | 我们的 GT/评测没问题；官方模型在我们尺子下依然强 → 差距主要在**训练侧**（bf16 collator 量化、配方） | EXP4 主攻训练侧修复 |
| **0.45 ~ 0.55**（≈EXP2 水平） | 官方 65.6 是用他们私有 GT 量出来的；我方差距大头是**尺子差异** | 停止以 65.6 为目标；以内部基线迭代为主，训练侧修复降为可选 |
| **< 0.40** | 我方尺子更严苛，**或**类别词表失配（见 §3） | 先完成 §3 词表对照与逐类表判读，再下结论 |

**不对称性提醒**：高分 → 可强推论（尺子没问题）；低分有多种解释（能力、GT 约定失配、词表失配、解码设置），必须结合 §3 词表检查和逐类表判读，不能直接断言「官方模型不行」。

残余不可控差异（写进报告 caveat）：官方 65.6 的评测 GT 与解码设置均未知；本实验阈值是带宽不是硬线。

---

## 2. 前置检查（跳过条件）

```bash
conda activate spatiallm && cd ~/SpatialLM && export HF_HOME=/home/chenle/hf_home_spatiallm

# 以下均为 EXP1/EXP2 现存产物，缺任何一样 → 报告用户，停止
ls data/scannet_spatiallm/val.csv data/scannet_spatiallm/layout/scene0011_00.txt
ls -d data/scannet_spatiallm/pcd_val_g0 data/scannet_spatiallm/pcd_val_g1 \
      data/scannet_spatiallm/pcd_val_g2 data/scannet_spatiallm/pcd_val_g3
wc -l data/scannet_spatiallm/val.csv   # 预期 313（表头+312）
```

若 `pcd_val_g*` 缺失，用 EXP1 的分片脚本重建：

```bash
cd data/scannet_spatiallm
i=0
for s in $(awk -F, 'NR>1 && $2=="val"{print $1}' split.csv | tr -d '\r'); do
  mkdir -p pcd_val_g$((i%4)); ln -sf ../pcd/$s.ply pcd_val_g$((i%4))/; i=$((i+1))
done
cd ~/SpatialLM
```

---

## 3. 下载模型 + 单场景冒烟 + **类别词表检查（本实验关键步骤）**

官方 ScanNet GT 转换用的类别词表未知。若他们用 `dining table` 而我们用 `table`、或用下划线形式，直接评估会**假性偏低**，必须先对照。

### 3.1 单场景推理（模型自动下载，~2.5GB）

```bash
python inference.py -d object \
    -p data/scannet_spatiallm/pcd/scene0011_00.ply \
    -o /tmp/exp3_smoke.txt \
    --model_path ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT --seed 42
```

**验证**：`/tmp/exp3_smoke.txt` 存在且含 `bbox` 行。

**失败处理**：
- 模型加载报 `model_type ... not recognized` → 不应发生（`inference.py` import `spatiallm` 时已注册 `spatiallm_qwen`，`spatiallm/model/spatiallm_qwen.py:394`）；若仍报错，把完整 traceback 报告用户，停止
- 下载 401/超时 → 确认已 `export HF_HOME=/home/chenle/hf_home_spatiallm`；仍失败试 `export HF_ENDPOINT=https://hf-mirror.com`
- 生成乱码 / 无 bbox 行 → 保存原始输出报告用户（可能模型与当前代码版本不兼容），停止

### 3.2 词表对照

```bash
grep -ohE 'Bbox\([a-zA-Z_ ]+' /tmp/exp3_smoke.txt | sed 's/Bbox(//' | sort | uniq -c
```

再跑 2~3 个场景（如 scene0050_00、scene0300_00、scene0567_00 中 val 里存在的）合并统计，与我们 18 类（**空格形式**，eval.py 会把下划线替换成空格）对照：

```
cabinet bed chair sofa table door window bookcase painting counter desk
curtain refrigerator shower curtain toilet sink bathtub otherfurniture
```

- **完全一致** → §5 直接用 `benchmark_categories.tsv`
- **有差异**（如同义异名、多出的类）→ 建 `data/scannet_spatiallm/benchmark_categories_exp3.tsv`：tab 分隔，表头 `scannet18\tscannet18`，键 = 官方模型输出的类别名（空格形式），值 = 对应到我们 18 类的名字；无法对应的类留空整行丢弃。**词表差异表必须写进报告**——它本身就是官方 GT 约定的直接证据

---

## 4. 批量推理（312 场景，4 卡分片，约 10 分钟）

```bash
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python inference.py -d object \
    -p data/scannet_spatiallm/pcd_val_g$g -o data/scannet_spatiallm/pred_val_exp3_g$g \
    --model_path ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT --seed 42 \
    > infer_exp3_g$g.log 2>&1 & done; wait

mkdir -p data/scannet_spatiallm/pred_val_exp3
cp data/scannet_spatiallm/pred_val_exp3_g*/*.txt data/scannet_spatiallm/pred_val_exp3/
ls data/scannet_spatiallm/pred_val_exp3/*.txt | wc -l   # 预期 312
```

解码设置与 EXP1/EXP2 完全一致（默认 temperature 0.6 采样、seed 42、逐场景）。

---

## 5. 评估（用我们的 GT）

```bash
python scripts/eval_scannet18.py --metadata data/scannet_spatiallm/val.csv \
    --gt_dir data/scannet_spatiallm/layout \
    --pred_dir data/scannet_spatiallm/pred_val_exp3 \
    --label_mapping data/scannet_spatiallm/benchmark_categories.tsv \
    --label_from scannet18 --label_to scannet18
# 若 §3.2 建了 exp3 词表映射，--label_mapping 换成 benchmark_categories_exp3.tsv

python scripts/analyze_errors.py --pred_dir data/scannet_spatiallm/pred_val_exp3 --tag exp3
```

---

## 6. 报告要求

写 `experiments/FINETUNE_SCANNET_EXP3_REPORT.md`（风格沿用 EXP1/EXP2 报告），**必含**：

1. **汇总数字表**：EXP1 / EXP2 / EXP3 / 官方，macro + micro × @.25/@.50（macro 是与官方可比的口径）
2. **逐类 F1 对照**：EXP3 vs EXP2（并排），标出差异最大的 5 个类并给出一句话归因
3. **词表差异表**（§3.2 结果，即使为空也要写「完全一致」）
4. **结论**：严格对照 §1 判读逻辑三档给出结论与建议的 EXP4 方向
5. **Caveat**：官方模型训练/评测 GT 未公开、其解码设置未知、我方 GT 为自动生成——本实验判读的是「同一把尺子下的相对水平」
6. 复现命令与产物清单

完成后向用户汇报并**停止**（不要自行发起 EXP4）。

## 7. 产物清单约定

| 路径 | 内容 |
|---|---|
| `data/scannet_spatiallm/pred_val_exp3/` | 312 场景预测 |
| `data/scannet_spatiallm/pred_val_exp3_g*/` | 分片输出 |
| `data/scannet_spatiallm/benchmark_categories_exp3.tsv` | （仅当词表有差异）官方词表→我们 18 类映射 |
| `infer_exp3_g*.log` | 分片推理日志 |
| `data/scannet_spatiallm/error_analysis/exp3_*` | 误差分类数据 |
| `experiments/FINETUNE_SCANNET_EXP3_REPORT.md` | 实验报告 |
