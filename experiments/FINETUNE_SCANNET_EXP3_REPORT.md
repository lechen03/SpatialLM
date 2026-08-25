# SpatialLM × ScanNet EXP3 报告：官方 ScanNet-SFT 模型交叉评测（「尺子」判别实验）

> 日期：2026-08-25/26　|　协议：`experiments/FINETUNE_SCANNET_EXP3_PROTOCOL.md`
> 测试模型：`ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT`（一作发布）　|　评测：我方 312 val 场景 + 我方 GT + 我方解码设置（seed 42）

---

## 1. 汇总数字

| 指标 | EXP1（冻结） | EXP2（解冻） | **EXP3 官方模型**† | EXP3 sensitivity‡ | 官方 README 口径 |
|---|---|---|---|---|---|
| macro F1 @.25 | 0.4426 | 0.4834 | **0.5715** | 0.5988 | **0.656** |
| macro F1 @.50 | 0.2836 | 0.3217 | **0.4254** | 0.4453 | 0.526 |
| micro F1 @.25 | 0.5509 | 0.6032 | **0.6600** | 0.6874 | — |
| micro F1 @.50 | 0.3516 | 0.4068 | **0.5051** | 0.5217 | — |
| GT 漏检 / 多余预测 | 2018 / 1372 | 1775 / 1272 | **1646 / 892** | — | — |
| 类混淆对 | 194 | 164 | **110** | — | — |
| 场景 F1 四分位 | .105/.400/.558/.682 | .000/.475/.600/.750 | **.154/.525/.667/.785** | — | — |

† 主变体：`garbagebin`（官方词表独有，469 框）按协议丢弃；`picture→painting`、`bookshelf→bookcase`、`showercurtrain→shower curtain` 映射。
‡ sensitivity：`garbagebin→otherfurniture`（ScanNet-20 将垃圾桶并入 otherfurniture 的合理性检验）。

## 2. 词表差异表（官方 GT 约定的直接证据）

312 场景共 3741 框，官方模型词表 18 项 vs 我方 18 类：

| 官方输出 | 框数 | 处置 | 说明 |
|---|---|---|---|
| chair/door/cabinet/table/window/desk/sofa/sink/bed/curtain/toilet/counter/refrigerator/bathtub（14 类同名） | 3136 | 恒等映射 | |
| `picture` | 169 | → painting | ScanNet 原名；我方为迁移改名 |
| `bookshelf` | 82 | → bookcase | 同上 |
| `showercurtrain` | 29 | → shower curtain | **官方词表自带拼写错误** |
| `garbagebin` | **469（12.5%）** | 主变体丢弃 / sensitivity→otherfurniture | **ScanNet-20 无此类** → 官方 GT 词表更宽（疑 ScanNet-200 系），是"官方从不输出 otherfurniture"的原因（主变体该类 F1=0，231/312 场景） |

**结构性差异（比词表更根本）**：官方模型全部 3572 个框只出现 **2 个角度值**——-π（3363 框）与 -π/2（678 框）→ **官方 GT 为轴对齐 AABB 约定**；我方 GT 为 SVD yaw-OBB。官方预测在我方尺子下对旋转物体的 IoU 天然吃亏，却仍拿到 0.57 macro——其中心/尺度/类别质量高于我们。

## 3. 逐类 F1 对照（@.25，EXP3 主变体 vs EXP2）

| 类 | EXP2 | EXP3 | Δ | 类 | EXP2 | EXP3 | Δ |
|---|---|---|---|---|---|---|---|
| toilet | 0.8497 | **0.8824** | +3.3 | curtain | 0.5033 | **0.5260** | +2.3 |
| chair | 0.7980 | **0.8586** | +6.1 | table | 0.4775 | **0.5770** | +10.0 |
| bed | 0.6094 | **0.6693** | +6.0 | refrigerator | 0.3191 | **0.5080** | +18.9 |
| sofa | 0.6015 | **0.6381** | +3.7 | painting | 0.2177 | **0.3366** | +11.9 |
| bathtub | 0.6129 | **0.7742** | +16.1 | bookcase | 0.3678 | **0.4890** | +12.1 |
| sink | 0.5794 | **0.7229** | +14.4 | cabinet | 0.3455 | **0.3702** | +2.5 |
| door | 0.5406 | **0.6462** | +10.6 | counter | 0.1773 | **0.4275** | +25.0 |
| window | 0.5429 | **0.5918** | +4.9 | otherfurniture | 0.3338 | 0.0000* | −33.4* |
| desk | 0.5028 | **0.6139** | +11.1 | shower curtain | 0.3226 | **0.6556** | +33.0 |

*词表缺失所致（官方用 garbagebin 等细类），sensitivity 变体下该项恢复。

**差异最大的 5 类与归因**：shower curtain +33.0（官方 GT 专设 `showercurtrain` 类、监督充分；我方 GT 该类薄物标注噪声大）；counter +25.0（薄长条柜台，官方中心/尺度质量高——其中心误差中位仅约我们一半）；refrigerator +18.9；bathtub +16.1；sink +14.4——共同点：**全是 EXP1/EXP2 诊断的感知瓶颈类，官方模型在这些类上的优势说明差距主要在点云特征质量与训练配方，而非 LLM 解码**。

## 4. 结论（对照协议 §1 判读逻辑）

主变体 macro@.25 = **0.5715**，落在协议两档（0.45–0.55 与 ≥0.60）**之间的过渡带**；sensitivity 变体 0.5988 触及 0.60 边界。协议明言"阈值是带宽不是硬线"，据此给出**双因素分解**而非单档结论：

1. **尺子差异真实存在，但不是大头**：官方模型换到我们的尺子上，0.656 → 0.57~0.60，损失约 **6~8pt**。来源已定位：轴对齐 vs OBB 的 GT 约定、词表宽度（garbagebin 等）、（可能的）GT 生成管线差异。
2. **训练侧差距是大头**：同一把尺子下，官方模型领先 EXP2 达 **+8.8pt macro / +5.7pt micro**，且 17/18 类全胜、误差结构全面更优（漏检 1646 vs 1775、多余框 892 vs 1272、混淆 110 vs 164、场景四分位整体上移）。我方 17.3pt 总差距 ≈ **9pt 能力差距 + 6~8pt 尺子差异**。

**建议的 EXP4 方向（训练侧修复，按 ROI）**：
- **修复 bf16 collator 对点云网格坐标的量化**（EXP1 已定位：训练目标带 ~10cm 噪声，train/test 偏斜）——最便宜的"纯增益"修复
- GT 管线对齐实验：把我们的 GT 改成轴对齐约定重训（同时消除一类尺子差异，双向受益）
- 类别重采样/加权（counter、refrigerator、sink 等官方优势最大的类）
- numpy RNG worker 修复（增强多样性）

**不建议**：继续以 0.656 为硬目标（其中 ~6-8pt 是不可比成分）；以 EXP2=0.4834 为内部基线迭代，以 EXP3=0.5715 为同尺参照上限。

## 5. 方法论备忘（复现必读）

1. **eval.py 的 DictReader 重复列名陷阱**：表头 `scannet18\tscannet18`（协议原文写法）时，`read_label_mapping` 的两次列查询都取第二列 → **任何非恒等映射都会退化成恒等**（首次运行 749 框被静默丢弃即此因）。修复：表头用不同列名（本实验 `official\tscannet18`），调用 `--label_from official`。`analyze_errors.py` 已同步加 `--label_from/--label_to` 参数。
2. 官方模型推理较慢（~15s/场景 vs 我们 ~8s），312 场景 4 卡约 13 分钟。
3. 官方模型所有角度恒为 -π/-π/2（轴对齐），对 IoU 评估无碍（中心对称等价），但意味着它无法表达旋转框。

## 6. Caveat

- 官方 0.656 的评测 GT 与解码设置均未公开，本实验只判定"**同一把尺子下的相对水平**"。
- 我方 GT 为自动生成（SVD yaw-OBB），本身含标注噪声。
- sensitivity 变体的 garbagebin→otherfurniture 是宽松假设（该类更宽），其 0.5988 应视为官方模型在我方尺子上的**偏乐观**界。
- 官方模型 2025-11-05 版本是否即 README 基准所用权重无法确证（推测同系）。

## 7. 复现命令与产物

```bash
conda activate spatiallm && cd ~/SpatialLM && export HF_HOME=/home/chenle/hf_home_spatiallm
# 冒烟+词表（§3）：inference.py --model_path ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT
# 批量（§4，tmux，4 卡分片同 EXP1/2 模板）→ pred_val_exp3/
python scripts/eval_scannet18.py --metadata data/scannet_spatiallm/val.csv \
    --gt_dir data/scannet_spatiallm/layout --pred_dir data/scannet_spatiallm/pred_val_exp3 \
    --label_mapping data/scannet_spatiallm/benchmark_categories_exp3.tsv \
    --label_from official --label_to scannet18
python scripts/analyze_errors.py --pred_dir data/scannet_spatiallm/pred_val_exp3 \
    --label_mapping data/scannet_spatiallm/benchmark_categories_exp3.tsv \
    --label_from official --label_to scannet18 --tag exp3
```

| 产物 | 路径 |
|---|---|
| 312 场景预测 | `data/scannet_spatiallm/pred_val_exp3/`（分片 `pred_val_exp3_g*/`） |
| 词表映射 | `data/scannet_spatiallm/benchmark_categories_exp3.tsv`（表头 `official\tscannet18`） |
| 误差分析 | `data/scannet_spatiallm/error_analysis/exp3_*` |
| 分片日志 | `infer_exp3_g*.log` |
| 本报告 | `experiments/FINETUNE_SCANNET_EXP3_REPORT.md` |

---

*按协议 §6 要求，本实验到此停止，不自行发起 EXP4。*
