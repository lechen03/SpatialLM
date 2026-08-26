# SpatialLM × ScanNet EXP4 报告：从官方 ScanNet-SFT 检查点全参数微调（领域引导实验）

> 日期：2026-08-26　|　协议：`experiments/FINETUNE_SCANNET_EXP4_PROTOCOL.md`
> 设计：与 EXP2 单变量对照——唯一差异 = 起点模型（`ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT`），数据/超参/解码/评测逐字段一致
> 产物：`saves/scannet_exp4/`（最终 + checkpoint-1500~3010）

---

## 1. 汇总数字（我方尺子，312 val，seed 42）

| 指标 @.25 / @.50 | EXP1（冻结） | EXP2（解冻） | **EXP4 最终** | **EXP4 best(2500)** | EXP3（官方冷评） | 官方自报 |
|---|---|---|---|---|---|---|
| macro F1 | 0.4426 / 0.2836 | 0.4834 / 0.3217 | 0.5791 / 0.4213 | **0.5825 / 0.4258** | 0.5715 / 0.4254 | 0.656 / 0.526 |
| micro F1 | 0.5509 / 0.3516 | 0.6032 / 0.4068 | 0.6788 / 0.4974 | **0.6832 / 0.5021** | 0.6600 / 0.5051 | — |
| GT 漏检 / 多余预测 | 2018 / 1372 | 1775 / 1272 | **1429** / 1089 | — | 1646 / 892 | — |
| 类混淆对 | 194 | 164 | 119 | — | 110 | — |
| 场景 F1 四分位 | .105/.400/.558/.682 | .000/.475/.600/.750 | .133/.540/.714/.800 | — | .154/.525/.667/.785 | — |

**判读（协议 §1 第一档，≥0.53）**：领域引导有效。EXP4 best macro@.25 = **0.5825**，相对 EXP2 **+9.9pt**，且**超过官方模型本身在我方尺子上的冷评表现**（0.5715）——官方特征优势不仅完整迁移，还被我方 yaw-OBB 微调进一步放大。AABB→OBB 的约定改写代价远小于特征收益（协议担心的第三档情形未发生）。

## 2. 逐类继承分析（协议必答：五类的 EXP2→EXP4 变化）

| 类 | EXP2 | EXP4 | Δ | 官方冷评 | 继承判定 |
|---|---|---|---|---|---|
| counter | 0.1773 | 0.3401 | **+16.3** | 0.4275 | 恢复官方优势的 ~59% |
| refrigerator | 0.3191 | 0.5012 | **+18.2** | 0.5080 | **几乎完全追平**（−0.7） |
| sink | 0.5794 | 0.7093 | **+13.0** | 0.7229 | 几乎追平（−1.4） |
| painting | 0.2177 | 0.3410 | **+12.3** | 0.3366 | **反超官方**（+0.4） |
| shower curtain | 0.3226 | 0.5333 | **+21.1** | 0.6556 | 恢复 ~59% |

五类全部大幅继承，其中三类追平/反超官方。**附加发现**：otherfurniture 0.3338 → **0.5179**（官方冷评该类为 0——它只输出 garbagebin 等细类；EXP4 学会了我方词表的同时保住了细粒度检测能力）。其余变化：toilet 0.850→0.843（−0.7）、bookcase +8.7、bathtub +12.0、curtain +6.3、cabinet +4.9、window +4.6、door +7.0；微降：sofa −3.6、bed +0.5、table +6.6、chair +4.8、desk +12.8。

## 3. 训练过程

- 3010 步 / 60 分钟（与 EXP2 同规格，ZeRO-2，GPU1 贴 23.7GB 上限稳定跑完）；缓存复用，单次启动直接训练。
- **首 100 步 train loss 显著低于 EXP2**（协议预测命中）：step10 = 1.5154 vs EXP2 1.7151；step100 = 0.6904 vs ~0.73；终值 0.5521 vs 0.6513。
- **eval loss 逐 checkpoint 领先 EXP2**，但优势随训练收窄（+5.3 → +1.7）——官方先验被逐渐消化，微调本身也在收敛：

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| EXP2 | 0.7515 | 0.7253 | 0.7091 | 0.7034 | 0.7011 | 0.7018（终 0.7001） |
| EXP4 | **0.6988** | **0.6892** | **0.6846** | **0.6829** | **0.6822** | 0.6843（终 0.6845） |

- **checkpoint 选择**：best = checkpoint-2500（0.6822）；best 的 F1 也略优于最终（macro 0.5825 vs 0.5791）——与 EXP1/EXP2 的"最终模型更优"结论相反，本次 eval loss 最低点与 F1 最优点一致，提示官方起点下训练后期存在轻微过拟合。

## 4. 词表收敛检查（协议必答）

```
chair 1289 | door 485 | otherfurniture 464 | cabinet 356 | table 283 | window 246
| painting 178 | desk 132 | sofa 86 | sink 85 | bookcase 82 | bed 65 | curtain 60
| toilet 51 | refrigerator 45 | counter 38 | shower_curtain 26 | bathtub 24
```

**恰好我方 18 类，0 个未映射框**（garbagebin/picture/bookshelf/showercurtrain 全部消失）——词表完全收敛。预测框数 3995（EXP2 3877，GT 4216），欠生成进一步缓解。

## 5. 几何质量（协议必答）

- **中心误差中位数（全类）：0.074 → 0.060m**（EXP2→EXP4），官方模型的中心精度优势被继承并放大。
- 改善最大的类：bathtub 0.085→0.059、cabinet 0.103→0.078、shower curtain 0.096→0.074、door 0.081→0.060、sink 0.081→0.061。
- @.50 口径：macro 0.4258 vs 官方冷评 0.4254（持平）、micro 0.5021 vs 0.5051（−0.3）——EXP4 与官方在我方尺子上 @.50 已无实质差距。

## 6. 结论与下一步

1. **EXP4 = 我方当前最佳模型**（`saves/scannet_exp4/checkpoint-2500` 为最佳权重，macro@.25 0.5825 / micro 0.6832）。交付建议用 checkpoint-2500。
2. 起点模型的选择被证明是**最重要的单一决策**（+9.9pt，超过 EXP2 解冻的 +4.1pt）——特征质量可经 checkpoint 迁移且与约定改写不冲突。
3. 与官方自报 0.656 的残余差距（~7pt）与我方训练侧配方（collator bf16 量化、增强 RNG、lr/epochs）及残余尺子差异相关。
4. **建议 EXP5（协议判读表指定）**：以官方 SFT 为起点，叠加 **collator bf16 修复**（EXP1 定位的 ~10cm 训练目标噪声）；可顺带修 numpy RNG。预期主攻 @.50（当前 0.426，几何精度的上限仍在）。

## 7. Caveat

- 官方 ScanNet-SFT 的训练集大概率为同一官方 train 1201 场景（与本实验训练集重叠）——本实验测的是"同数据、换标签约定的能力迁移"，val 312 对双方均为干净测试集；但"官方起点 + 我方数据"的组合使 EXP4 与 EXP2 的对比包含"见过的数据"成分（协议 §0 已声明此非泄漏）。
- AABB→OBB 约定改写已完成（词表+角度），但改写摩擦无法单独量化。
- 官方自报 0.656 的 GT 与解码设置未知，EXP3 已证 ~6-8pt 为尺子差异。
- 解码采样（temperature 0.6，seed 42）与 EXP1-3 一致。

## 8. 复现命令与产物

```bash
conda activate spatiallm && cd ~/SpatialLM && export HF_HOME=/home/chenle/hf_home_spatiallm
# 训练（tmux；缓存复用单次启动）
tmux send-keys -t spatiallm_train 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate spatiallm && export HF_HOME=/home/chenle/hf_home_spatiallm && python train.py configs/scannet_sft_exp4.yaml 2>&1 | tee train_exp4.log' Enter
# 双模型推理与评估：--model_path saves/scannet_exp4 与 saves/scannet_exp4/checkpoint-2500
#   输出 pred_val_exp4/ 与 pred_val_exp4_best/；eval/analyze 同 EXP2 流程（恒等映射）
```

| 产物 | 路径 |
|---|---|
| 模型 | `saves/scannet_exp4/`（最终）+ `checkpoint-2500`（最佳） |
| 训练日志 | `train_exp4.log` |
| 预测 | `data/scannet_spatiallm/pred_val_exp4/`、`pred_val_exp4_best/`（分片 `_g*/`） |
| 误差分析 | `data/scannet_spatiallm/error_analysis/exp4_*` |
| 分片日志 | `infer_exp4_g*.log`、`infer_exp4_best_g*.log` |
| 本报告 | `experiments/FINETUNE_SCANNET_EXP4_REPORT.md` |

---

*按协议 §6 要求，本实验到此停止，不自行发起 EXP5。*
