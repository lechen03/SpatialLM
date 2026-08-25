# SpatialLM × ScanNet 微调实验报告（EXP2：解冻点云塔）

> 日期：2026-08-25　|　服务器：4× RTX 4090（24GB）　|　基线：EXP1（`experiments/FINETUNE_SCANNET_EXP1_REPORT.md`）
> 配置：`configs/scannet_sft_full_ds.yaml`（runbook 方案 B）　|　输出：`saves/scannet_full/`

---

## 1. 摘要

**单一变量对照**：相对 EXP1（方案 A，`freeze_point_tower: true`），EXP2 解冻 Sonata 点云编码器做全量微调，其余超参完全一致（lr 1e-5、10 epochs、batch 1、同数据同缓存）。为在 24GB 卡上容纳全量训练，按 runbook 方案 B 启用 **DeepSpeed ZeRO-2**（优化器状态+梯度 4 卡分片，与 DDP 数学等价，不引入实验变量）。

**结果：全指标显著提升**——micro F1@.25 **0.5509 → 0.6032（+5.2pt）**、@.50 +5.5pt、macro +4.1/+3.8pt。增益集中在 EXP1 诊断出的感知瓶颈类（window +11.1、sink +10.4、otherfurniture +9.3），验证了"漏检薄/小/壁挂物体是点云编码器分辨率问题"的归因。**EXP2 模型（`saves/scannet_full/`）成为新的交付模型。**

## 2. 实验设置

### 2.1 相对 EXP1 的全部差异

| 项 | EXP1 | EXP2 |
|---|---|---|
| freeze_point_tower | true | **false** |
| deepspeed | 无 | **configs/ds_zero2.json（ZeRO-2）** |
| output_dir | saves/scannet_frozen | saves/scannet_full |

其余（lr 1e-5 cosine、warmup 0.03、10 epochs、per-device batch 1、bf16 混合精度、cutoff 8192、num_bins 1280、eval/save 500 步、数据集与预处理缓存 `saves/scannet_cache` 复用）逐字段一致。

### 2.2 启动波折：裸解冻 OOM → ZeRO-2

首次尝试直接解冻（无 ZeRO-2）在 **step 24 OOM**：GPU1 已用 22.32/23.64GB，backward 需再分配 2.15GB。原因：可训练参数 494.75M → 603.5M（静态 +1.3GB）+ Sonata 反向激活（随场景点数 8 万~18.6 万波动）。日志建议的 `expandable_segments` 本机不支持（驱动限制）。

处置：改装 deepspeed 0.19.5 并启用 ZeRO-2（梯度+优化器状态分片，每卡省 ~5.4GB）。OOM 日志存档于 `train_exp2_oom.log`。

## 3. 训练过程

- **3010 步 / 60 分钟**（EXP1 57 分钟，ZeRO-2 通信开销仅 +5%），1.16s/it，干净退出。
- 显存：GPU2/3 约 17–20GB，GPU0/1 高达 22.2/23.7GB（贴上限但全程稳定，含 6 次 eval + checkpoint）。
- train loss：1.7151 → **0.6513**（最低 0.6302；EXP1 终值 0.7014）。
- eval loss 逐点对比（每个 checkpoint 均领先，优势随训练扩大）：

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| EXP1 | 0.7772 | 0.7568 | 0.7416 | 0.7368 | 0.7353 | 0.7356 |
| EXP2 | 0.7515 | 0.7253 | 0.7091 | 0.7034 | **0.7011** | 0.7018（终评 0.7001） |

## 4. 评估结果（312 val 场景，18 类口径，seed 42）

### 4.1 汇总

| 指标 | EXP1 | EXP2 | Δ |
|---|---|---|---|
| micro F1 @.25 | 0.5509 | **0.6032** | +5.2pt |
| micro F1 @.50 | 0.3516 | **0.4068** | +5.5pt |
| macro F1 @.25 | 0.4426 | **0.4834** | +4.1pt |
| macro F1 @.50 | 0.2836 | **0.3217** | +3.8pt |
| GT 漏检总数 | 2018 | **1775** | −243 |
| 多余预测总数 | 1372 | **1272** | −100 |
| 类别幻觉 | 1/3765 | 2/3879 | 仍≈0 |
| 场景 F1 四分位 | .105/.400/.558/.682/1.0 | .000/.475/.600/.750/1.0 | 下四分位 +7.5pt |

### 4.2 逐类（F1@.25，EXP1 → EXP2）

**提升**：window 0.432→**0.543**（+11.1）、sink 0.475→**0.579**（+10.4）、otherfurniture 0.240→**0.334**（+9.3）、bookcase 0.287→**0.368**（+8.1）、door 0.462→**0.541**（+7.9）、cabinet 0.277→**0.346**（+6.9）、sofa 0.533→**0.602**（+6.9）、curtain 0.443→**0.503**（+6.0）、toilet 0.803→**0.850**（+4.6）、chair 0.756→**0.798**（+4.2）、refrigerator 0.270→0.319、table +1.0、bathtub +1.1。
**回退**：bed 0.640→0.609（−3.0）、counter 0.203→0.177（−2.6）、painting 0.241→0.218（−2.4）、desk −0.9；shower curtain 持平。

> 解冻的增益方向与 EXP1 失败模式诊断精确吻合：感知瓶颈类（薄/小/壁挂）受益最大；回退类多为标注噪声敏感或小样本类（counter 仅 47 场景、painting 的平板 GT 约定问题依旧）。

### 4.3 误差结构变化（EXP1 → EXP2）

| 误差类 | GT 侧 | 预测侧 |
|---|---|---|
| 纯漏检 undetected | 1462 → **1320**（−142） | — |
| 定位差 loc_miss / loc_fp | 362 → 291 | 370 → 305 |
| 类混淆 confusion | 194 → **164**（−30） | 17 → 7 |
| 虚构 spurious | — | 948 → 920 |
| 重复 duplicate | — | 37 → 40 |

改善是全面的（漏检、定位、混淆三项都降），其中**漏检降幅最大**——正是"点云特征不够好"的直接证据。

### 4.4 匹配对几何质量（中位数，EXP1 → EXP2）

- counter：中心误差 **0.227 → 0.114m（减半）**，高度偏差 0.672 → 0.766
- shower curtain：宽度膨胀 1.392 → 1.260（缓解）
- desk：IoU 0.629 → 0.675，中心 0.138 → 0.107
- 其余各类普遍：中心误差 ↓、IoU ↑（chair 0.579→0.605、sofa 0.705→0.743、window 0.450→0.497）
- 轻微回退：toilet IoU 0.567→0.553、painting 0.490→0.464（90° 翻转与平板问题依旧，见 EXP1 模式②）

## 5. 结论与后续

1. **解冻点云塔是正确投资**：+5.2pt micro@.25，且精准修复感知瓶颈类。ZeRO-2 以 +5% 时长代价解决 24GB 显存问题。**新交付模型：`saves/scannet_full/`**（checkpoint-1500~3010 保留）。
2. 残余问题排序（EXP1 报告的模式在 EXP2 后的状态）：
   - 模式①薄/壁挂漏检：**部分缓解**（window/curtain/sink 明显改善；painting/counter 反而略退——GT 约定问题未解）
   - 模式③otherfurniture 汇聚盆：**仍是最大单项**（263 漏 + 误聚，占比升高）→ 下一优先级
   - 模式②@.5 中心/尺度精度：**普遍改善**但反事实结构不变（中心仍是首要杠杆）
   - 模式④desk↔table 等语义混淆：小幅下降（194→164），仍是标签先验问题
   - 模式⑤door 重复：持平（37→40）
3. 建议的 EXP3 候选（按预期 ROI）：
   - **otherfurniture 类别治理**（重映射/拆分/过采样）——当前最大错误源
   - painting/counter 的 GT 平板约定审计（EXP1 发现 sz<0.25m 框 recall 极低）
   - door NMS 去重后处理（无需重训）
   - 更长训练 / 更大 lr 只作用于点云塔的分层解冻实验

## 6. 复现命令

```bash
conda activate spatiallm && cd ~/SpatialLM && export HF_HOME=/home/chenle/hf_home_spatiallm
pip install deepspeed   # 已装 0.19.5

tmux send-keys -t spatiallm_train 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate spatiallm && export HF_HOME=/home/chenle/hf_home_spatiallm && python train.py configs/scannet_sft_full_ds.yaml 2>&1 | tee train_exp2.log' Enter

# 评估同 EXP1 流程，--model_path saves/scannet_full，输出 pred_val_exp2
```

## 7. 产物清单

| 路径 | 内容 |
|---|---|
| `saves/scannet_full/` | EXP2 交付模型 + checkpoint-1500/2000/2500/3000/3010 |
| `train_exp2.log` / `train_exp2_oom.log` | ZeRO-2 训练日志 / 首次裸解冻 OOM 日志 |
| `data/scannet_spatiallm/pred_val_exp2/` | 312 场景预测 |
| `data/scannet_spatiallm/error_analysis/exp2_*` | 误差分类数据（analyze_errors.py --tag exp2） |
| `infer_exp2_g*.log` | 分片推理日志 |
