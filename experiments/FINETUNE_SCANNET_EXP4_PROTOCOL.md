# SpatialLM × ScanNet EXP4 协议：从官方 ScanNet-SFT 检查点全参数微调（领域引导实验）

> **本文档的主要读者是服务器上的 AI Agent。**
> 背景：EXP3 把 17.3pt 总差距分解为 **8.8pt 同尺能力差**（官方模型在我们 GT 下 0.5715 vs 我方 EXP2 0.4834）+ 6~8pt 尺子差。能力差的主要证据是官方模型的点云特征质量（中心误差约我方一半、薄/小物体 recall 高）。本实验检验：**从官方 ScanNet-SFT 检查点继续训练，能否把这部分特征优势迁移到我方尺子下**。
> 设计：与 EXP2 **单变量对照**——唯一差异是起点模型（`manycore-research/SpatialLM1.1-Qwen-0.5B` → `ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT`），数据/超参/解码/评测逐字段一致。

## 0. 已知信息

| 项 | 值 |
|---|---|
| 起点模型 | `ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT`（EXP3 已下载，缓存在 HF_HOME，无需重新下载） |
| 训练数据 | 我方 `data/scannet_spatiallm/`（1201 train，SVD yaw-OBB GT，我方 18 类词表） |
| 对照 | EXP1 macro@.25/.50 = 0.4426/0.2836；EXP2 = 0.4834/0.3217；EXP3（官方模型，我方尺子）= 0.5715/0.4254；官方 README = 0.656/0.526 |
| 配置 | `configs/scannet_sft_exp4.yaml`（已备好，无需修改） |

注意两点先验事实（写报告时参照）：
1. 官方 ScanNet-SFT 的训练集**大概率也是 ScanNet train 1201 场景**（与我们同一官方 split）——这不是数据泄漏问题：本实验测的是「同一数据、换标签约定下的能力迁移」，val 312 对双方都是干净测试集。
2. 该检查点的 GT 约定是**轴对齐（AABB）+ 官方词表**（garbagebin/picture/bookshelf/showercurtrain，见 EXP3 报告 §2）；我们的 GT 是 SVD yaw-OBB + 我方词表。微调会把模型改写到我方约定（EXP1/EXP2 已证明模型能学会我方约定），但改写过程的摩擦是本实验的观察点之一。

## 1. 判读逻辑（写报告时必须对照此表下结论）

以 EXP4 最终模型 macro F1@.25（我方尺子、312 val）为准：

| 结果 | 结论 | 下一步 |
|---|---|---|
| **≥ 0.53** | 领域引导有效：官方特征优势可经 checkpoint 迁移，且未被约定改写抹掉 | 以此起点为新基线；EXP5 = collator bf16 修复叠加 |
| **0.48 ~ 0.53**（≈EXP2） | 起点不重要：瓶颈在我方训练侧（collator 量化、配方），不在初始化 | EXP5 = collator 修复，起点任选（建议官方 SFT 起点） |
| **< 0.48**（差于 EXP2） | 约定冲突/灾难性遗忘主导（AABB→OBB 改写代价超过特征收益） | 查 eval loss 曲线与逐类表；EXP5 = AABB GT 重生成（消除约定冲突） |

**必做的辅助判读**（不依赖总分档位）：
- **逐类继承分析**：官方在 counter/refrigerator/sink/painting/shower curtain 上领先 EXP2 最多（+11~+33pt）——EXP4 后这些类是否保留部分优势？这是「特征迁移」最直接的证据，即使总分落在中档也要回答
- **几何质量**：matched_geo 的 center_dist 中位数 vs EXP2（EXP2 约 0.10~0.15m 量级）——官方起点的中心精度优势是否保留
- **词表收敛**：推理输出是否已收敛到我方 18 类（残留 garbagebin 等输出为未收敛信号，写入报告）

## 2. 前置检查（跳过条件）

```bash
conda activate spatiallm && cd ~/SpatialLM && export HF_HOME=/home/chenle/hf_home_spatiallm

# 数据与缓存（EXP1-3 现存产物）
ls data/scannet_spatiallm/scannet_train.json data/scannet_spatiallm/val.csv
ls -d saves/scannet_cache   # 预处理缓存，可复用（见 §3 注意）
# 官方模型已在 HF_HOME 缓存（EXP3 下载）
python - <<'EOF'
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained("ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT")
assert c.model_type == "spatiallm_qwen", c.model_type
assert c.point_config["num_bins"] == 1280, c.point_config
t_new = AutoTokenizer.from_pretrained("ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT")
t_old = AutoTokenizer.from_pretrained("manycore-research/SpatialLM1.1-Qwen-0.5B")
s = "<|point_start|>Detect boxes."
assert t_new(s)["input_ids"] == t_old(s)["input_ids"], "tokenizer mismatch"
print("preflight OK")
EOF
```

**预期**：输出 `preflight OK`。
**失败处理**：
- `point_config["num_bins"] != 1280` → 与我方数据预处理不一致，报告用户，停止
- tokenizer 不一致 → 预处理缓存不可复用：把 yaml 里 `save_dir` 改为 `saves/scannet_cache_exp4`（此处允许改此一个字段），并按 §3 的「两次启动」说明操作
- 模型不在缓存（HF_HOME 被清）→ 直接进 §3，训练启动时会自动重新下载

## 3. 训练（约 60 分钟，与 EXP2 同规格）

```bash
# tmux 内启动（会话断开安全；沿用 EXP1/EXP2 的会话名）
tmux send-keys -t spatiallm_train 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate spatiallm && export HF_HOME=/home/chenle/hf_home_spatiallm && python train.py configs/scannet_sft_exp4.yaml 2>&1 | tee train_exp4.log' Enter
```

- 若 tmux 会话不存在：`tmux new-session -d -s spatiallm_train -c ~/SpatialLM` 后再执行上述命令
- **两次启动说明**（EXP1 已知行为）：若 `saves/scannet_cache` 不存在或被换新目录，第一次运行只做预处理后正常退出（日志标志 `Preprocessed dataset is saved at ...`），**同命令再跑一次**才真正训练；缓存已存在则直接训练
- 监控：`tail -f train_exp4.log` + `nvidia-smi`；显存画像应与 EXP2 相当（GPU0/1 贴 22~23.7GB 上限属正常）
- **训练观察点**（写报告）：
  - 首个 100 步 train loss：预期**显著低于 EXP2 同期**（1.66 → 猜测 1.0 上下；模型已见过 ScanNet）——若反而更高，说明约定改写摩擦大，记录之
  - eval loss 曲线 vs EXP2（0.7515→0.7001）：全程序是否领先
- **checkpoint 选择**（照 EXP1 §4.3 做法）：最终模型与 eval loss 最低的 checkpoint **都要评估**，报告里并排
- OOM：与 EXP2 同规格本不应发生；若发生按 EXP2 协议阶梯（ZeRO-3 → cutoff 4096 → 报告停止）

## 4. 批量推理（312 val，4 卡分片，约 10 分钟）

对**最终模型**和 **eval-loss 最低 checkpoint** 各跑一遍（后者输出目录加后缀 `_best`）：

```bash
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python inference.py -d object \
    -p data/scannet_spatiallm/pcd_val_g$g -o data/scannet_spatiallm/pred_val_exp4_g$g \
    --model_path saves/scannet_exp4 --seed 42 > infer_exp4_g$g.log 2>&1 & done; wait
mkdir -p data/scannet_spatiallm/pred_val_exp4
cp data/scannet_spatiallm/pred_val_exp4_g*/*.txt data/scannet_spatiallm/pred_val_exp4/
ls data/scannet_spatiallm/pred_val_exp4/*.txt | wc -l   # 预期 312
```

## 5. 评估（我方尺子，恒等映射）

模型已用我方词表训练，**不需要** EXP3 的跨词表 tsv：

```bash
python scripts/eval_scannet18.py --metadata data/scannet_spatiallm/val.csv \
    --gt_dir data/scannet_spatiallm/layout \
    --pred_dir data/scannet_spatiallm/pred_val_exp4 \
    --label_mapping data/scannet_spatiallm/benchmark_categories.tsv \
    --label_from scannet18 --label_to scannet18

python scripts/analyze_errors.py --pred_dir data/scannet_spatiallm/pred_val_exp4 --tag exp4
```

注意 `benchmark_categories.tsv` 表头为重复列名 `scannet18\tscannet18`，恒等映射下无 EXP3 §5 的 DictReader 陷阱；仅当再做非恒等映射时才需换不同列名。

## 6. 报告要求

写 `experiments/FINETUNE_SCANNET_EXP4_REPORT.md`（风格沿用 EXP1-3 报告），**必含**：

1. **汇总数字表**：EXP1 / EXP2 / **EXP4（最终 + best ckpt 两行）** / EXP3 / 官方，macro + micro × @.25/@50
2. **§1 三档判读**的结论 + **逐类继承分析**（counter/refrigerator/sink/painting/shower curtain 五类的 EXP2→EXP4 变化）
3. **训练过程**：首 100 步 loss vs EXP2、eval loss 曲线对照表（逐 checkpoint）
4. **词表收敛检查**：`grep -ohE 'Bbox\([a-zA-Z_ ]+' pred_val_exp4/*.txt | sed 's/Bbox(//' | sort | uniq -c` 的结果（应只剩我方 18 类）
5. **几何质量**：matched_geo center_dist 中位数（总体 + 分哪些类）vs EXP2
6. Caveat（官方模型训练集疑与我校 train 重叠但结论不受影响、AABB→OBB 约定改写、解码设置同前）
7. 复现命令与产物清单

完成后向用户汇报并**停止**（不要自行发起 EXP5）。

## 7. 产物清单约定

| 路径 | 内容 |
|---|---|
| `saves/scannet_exp4/` | EXP4 最终模型 + checkpoints |
| `train_exp4.log` | 训练日志 |
| `data/scannet_spatiallm/pred_val_exp4/`（及 `_best/`） | 312 场景预测（最终 / best ckpt） |
| `data/scannet_spatiallm/error_analysis/exp4_*` | 误差分类数据 |
| `infer_exp4_g*.log` | 分片推理日志 |
| `experiments/FINETUNE_SCANNET_EXP4_REPORT.md` | 实验报告 |
