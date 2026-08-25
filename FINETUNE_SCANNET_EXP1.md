# SpatialLM × ScanNet 微调实验报告（EXP1）

> 日期：2026-08-24 ~ 08-25　|　服务器：4× RTX 4090（24GB），64 核 / 503GB RAM
> 代码：`~/SpatialLM`（branch `scannet-finetune`，commit `c3970f1`）
> 方案：**A**（`configs/scannet_sft_4090.yaml`，冻结点云塔）　|　Runbook：`FINETUNE_SCANNET.md`

---

## 1. 摘要

在 1201 个 ScanNet train 场景上以方案 A（冻结 Sonata 点云编码器，只训 LLM + 投影层）微调 SpatialLM1.1-Qwen-0.5B，10 epochs / 57 分钟，4 卡 DDP 每卡 15–18GB。在 312 个 val 场景上的**18 类完整口径**结果：

| 指标 | @0.25 | @0.50 |
|---|---|---|
| micro F1 | **0.5509** | 0.3516 |
| macro F1（官方口径风格） | 0.4426 | 0.2836 |

类别幻觉几乎为零（3765 个预测框中仅 1 个类别外框）。失败模式高度结构化：**漏检集中于薄/小/壁挂类、otherfurniture 汇聚盆占 20% 错误量、@.50 失败由中心+尺度主导（角度多为对称性伪影）、语义邻居标签翻转**。checkpoint-2500（eval loss 最低）对比最终模型：micro 略差 → **交付最终模型**。

---

## 2. 环境与数据

### 2.1 环境（conda env `spatiallm`）

| 组件 | 版本 | 验证 |
|---|---|---|
| Python | 3.11.15 | ✓ |
| torch | 2.4.1+cu124，4 卡可见 | ✓ |
| flash-attn | 2.8.3.post1（源码编译，MAX_JOBS=32） | ✓ |
| torch-scatter / spconv / timm | 2.1.2+pt24cu124 / cu120 / 1.0.28 | ✓ |
| transformers / datasets / accelerate | 4.46.1 / 3.6.0 / 1.7.0（均为上界版本） | ✓ |

cuda-toolkit 12.4.99（conda nvidia/label 渠道）+ sparsehash 由 conda 安装；其余经 `poetry install` + `poe install-sonata` + `poe install-training`。

### 2.2 数据

- 源：`data/scannet`（Pointcept 预处理，18GB）：train 1201 + val 312 个 .pth + `scannet_axis_align_matrix_trainval.pkl`，键 `coord/color/semantic_gt20/instance_gt` 已实测兼容。
- 转换：`prepare_scannet.py` → 1513 PLY + GT layout txt（yaw-OBB、18 类、排除 wall/floor/<30 点实例），78 秒完成。
- 质量抽查：`check_conversion.py scene0000_00` → **mean containment 1.000，>99% 26/26，class match 26/26**（与本地验证一致）。
- ShareGPT：`scannet_train.json` 1201 / `scannet_val.json` 312 样本 + dataset_info.json 注册。

---

## 3. 训练

### 3.1 配置（未改动 runbook 默认）

`freeze_point_tower: true`（方案 A 核心），`freeze_language_tower/train_proj_only: false`，`cutoff_len 8192`，`per_device_train_batch_size 1`，`lr 1e-5` cosine + warmup 0.03，`bf16`，`num_train_epochs 10`，`eval_steps/save_steps 500`，`save_total_limit 5`，`report_to none`，`num_bins 1280`。

### 3.2 过程中的问题与处置

| 问题 | 处置 |
|---|---|
| 首次启动 401 Unauthorized：`~/.cache/huggingface/token` 中存在损坏 token（825 字节，"signature verification failed"），污染公开模型的匿名下载 | **未动用户文件**，给训练进程单独 `export HF_HOME=/home/chenle/hf_home_spatiallm` → 匿名下载成功，模型缓存在此。（建议用户择机 `huggingface-cli logout && login` 重建） |
| runbook 说一次启动即可，实际 **train.py 需要启动两次**：第一次只做 tokenizer 下载 + 数据预处理（存 `saves/scannet_cache`）后所有 rank `sys.exit(0)`；第二次同命令才真正训练（`loader.py:450-459`） | 按两次启动执行；日志标志 `Preprocessed dataset is saved at saves/scannet_cache` |
| 用户要求训练置于 tmux（远程会话要断开） | 先等 checkpoint-500 落盘 → 干净杀进程树 → `tmux spatiallm_train` 内重启（自动从 ckpt-500 续训，无损）；随后用户要求全新训练，删 `saves/scannet_frozen` + 旧日志存档 `train.log.old` 后重启 |
| `split.csv` 是 Python csv.writer 写的 `\r\n` 行尾，awk `$2=="val"` 匹配失败 | 提取时加 `tr -d '\r'` |

### 3.3 训练结果

- **3010 步 / 10 epochs / 57 分 01 秒**（3.51 样本/秒，1.10s/it），干净退出。
- train loss：**1.6633 → 0.7014**（前 100 步降到 0.84，之后长尾，~step 2000 饱和）。
- eval loss（312 val，每 500 步）：0.7772 → 0.7568 → 0.7416 → 0.7368 → **0.7353(@2500 最低)** → 0.7356(@3000)；train/eval 差距 ~0.035 且 eval 无回升 → **无过拟合**，10 epochs 剂量合适（~7 epochs 即可拿到近似结果）。
- 资源：4 卡 100% 利用率，每卡 15.4–18.3GB（方案 A 预估 10–20GB 内）。
- 产物：`saves/scannet_frozen/`（最终模型 model.safetensors 2.41GB fp32 + tokenizer，可直接作 `--model_path`）+ checkpoint-1500/2000/2500/3000/3010。

---

## 4. 评估

### 4.1 协议说明（重要）

- 推理：`inference.py -d object --seed 42`（逐场景设种子，可复现；采样解码 temperature 0.6）。312 场景按 4 卡轮转分片并行（与串行结果完全一致），~10 分钟。
- **stock `eval.py` 的坑**：其 OBJECTS 列表是 SpatialLM 官方词表硬编码，18 个 ScanNet 类只覆盖 7 类（table/desk/toilet/sink/bookcase/counter/bathtub/shower curtain/otherfurniture/door/window 被静默丢弃），Layouts 表全 masked。其输出与官方 65.6@.25 **不可比**。
- 本实验补充脚本 **`eval_scannet18.py`**：import eval.py 原函数，仅将类列表换成完整 18 类；两脚本共有 7 类数字完全一致（协议等价性验证通过）。

### 4.2 逐类结果（18 类，312 val 场景）

| 类 | F1@.25 | F1@.50 | 出现场景数 | | 类 | F1@.25 | F1@.50 | 出现场景数 |
|---|---|---|---|---|---|---|---|---|
| toilet | **0.8033** | 0.5373 | 51 | | curtain | 0.4434 | 0.2318 | 52 |
| chair | **0.7560** | 0.5851 | 223 | | table | 0.4679 | 0.4101 | 201 |
| bed | 0.6398 | 0.5645 | 62 | | door | 0.4616 | 0.1813 | 243 |
| bathtub | 0.6022 | 0.5054 | 31 | | sink | 0.4754 | 0.1383 | 88 |
| sofa | 0.5329 | 0.4757 | 70 | | window | 0.4323 | 0.1807 | 179 |
| desk | 0.5114 | 0.4016 | 96 | | shower curtain | 0.3226 | 0.0000 | 31 |
| | | | | | bookcase | 0.2871 | 0.2199 | 61 |
| | | | | | refrigerator | 0.2704 | 0.1981 | 53 |
| | | | | | cabinet | 0.2765 | 0.1915 | 172 |
| | | | | | painting | 0.2412 | 0.1055 | 101 |
| | | | | | otherfurniture | 0.2404 | 0.1150 | 258 |
| | | | | | counter | 0.2028 | 0.0625 | 48 |

**汇总**：macro 0.4426 / 0.2836；micro 0.5509 / 0.3516。GT 4216 框 vs 预测 3764 框（欠生成 ~11%）；幻觉类别 1 个（cushion）。

### 4.3 checkpoint-2500（eval loss 最低点）对比

| 指标 | 最终模型(3010) | ckpt-2500 |
|---|---|---|
| micro F1 @.25 / @.50 | **0.5509 / 0.3516** | 0.5424 / 0.3384 |
| macro F1 @.25 / @.50 | 0.4426 / **0.2836** | 0.4468 / 0.2754 |
| GT 漏检 / 多余框 | **2018 / 1372** | 2036 / 1449 |
| 场景 F1 四分位 | 0.105/0.400/0.558/0.682/1.0 | 几乎相同 |

**结论：eval loss 最低 ≠ 检测最优；交付最终模型（`saves/scannet_frozen/` 根目录）。**

---

## 5. 逐场景误差分析（`analyze_errors.py` + 4 路并行深挖）

误差总体：**2018 漏检 + 1372 多余框**。GT 侧：72.4% 纯漏检（最佳跨类 IoU<0.1）、18.0% 定位差漏检、9.6% 类混淆（194 对）。

### 模式 ①：薄/小/壁挂物体系统性漏检（最大错误源）

painting recall **0.19**（156 漏检，中位 0.44×0.10×0.41m 挂高 1.79m）、counter 0.27（73% 的 GT 为 ≥2m×0.27m 长薄条）、refrigerator 0.29（漏的多为矮款 sz 1.30 vs 1.51）、window 0.40（漏检:虚构=3.8:1）。对照组 toilet R 0.82 / bed 0.74 / chair 0.73。**失败集中在点云塔分辨率边缘的长尾几何**。另发现 **238 个 sz<0.25m 平板 GT 框 recall 仅 5.9%**（平板 table 0/47、平板 painting 0/38）——训练见过但推理几乎不输出如此扁的框。

### 模式 ②：@.50 失败 = 中心(50%) + 尺度(38%)，角度基本是伪影

对 814 个 IoU∈[0.25,0.5) 匹配对的反事实归因：对齐中心可救 52.2%（F1@.5 oracle 0.347→0.453），对齐尺度 42.5%（→0.434），对齐角度仅 14.9%（→0.377）；任两因子对齐 → 0.549（≈当前 @.25 水平）。
**角度**：朴素中位差 102°，但 36.1% 是 180° 翻转（同一盒子）、11.4% 是近正方形底的 90° 翻转；mod-180 后中位仅 15.6°。真角度病：**toilet**（中位 71°，干净 90° 翻转；马桶纵横比 1.44 无法靠对称蒙混，角度正是其 @.5 及格/不及格的分界）、**chair**（981 对，方差大）。壁挂类角度近乎完美（door 4.5°、painting 2.9°）。
**尺度系统偏差**：counter 高度 ×0.672、shower curtain 宽度 ×1.39、window/bookcase y ×0.87、sink 高度不稳定 [0.66,1.51]（预测尺度落在 1/32m 格网上）。

### 模式 ③：otherfurniture 汇聚盆（占总错误量 20.1%）

308 漏检 + 261 虚构 + 混淆双通道（34 出 / 25 入）。最差四分位场景中占 GT 框 20.6%（其他场景 12.2%），并吸收 cabinet/refrigerator/chair 的边界不确定性。

### 模式 ④：语义邻居标签翻转（几何对、标签错）

194 对混淆中 **59.8% IoU≥0.5**：desk↔table 36 对（18.6%）、cabinet→otherfurniture 14、refrigerator→otherfurniture/cabinet 16、otherfurniture→table 10、window→door 7（同为墙面薄板）。模型局部化正确但无法仅凭点云选定标签。

### 模式 ⑤：door 过度触发

166 spurious + **29 同墙重复**（最重一幕 7 预测 vs 2 GT），唯一 P<R 的类。

### 场景级规律

- f1 与场景框数**零相关**（r=−0.008）、与点云大小弱相关（~3% 方差）→ **难度由类别组成决定**；结构类（door/window/painting/curtain）占比与 f1 相关 r=−0.35。
- 房型：bathroom 0.580 > living 0.487 ≈ bedroom 0.480 > **kitchen 0.470**（KW p=2.4e-6）。
- 最差四分位（74 场景）recall 0.301 且 precision 0.413——**全面崩坏非偏科**；完美场景仅 5 个（全是 3–6 件 chair+table 小房间）。
- 61% 场景欠生成（R 0.521 vs P 0.635），但欠/过方向与 f1 无关（r=+0.014），失配幅度才相关（|Δ| r=−0.237）。

---

## 6. 已知上游缺陷（影响解读，受 runbook 约束未修）

1. **训练 collator 将点云张量转 bf16**（含整数网格坐标，≥512 的坐标最高 ~4 voxel≈10cm 量化误差），推理却是精确整数 → 训练目标带噪声、train/test 偏斜（`collator.py:196-198`）。
2. **numpy RNG 在 32 个 dataloader worker 间相关**（`seed_worker` 只重置 torch）→ 数据增强多样性打折。
3. GT 本身为 SVD 拟合 yaw-OBB（含标注噪声），与官方数据管线或有差异。
4. 本报告 micro F1@.50（0.3516）来自独立 0.5 阈值匹配；`analyze_errors.py` 的 tp50 嵌套在 0.25 匹配内（0.3469，略保守）。

## 7. 改进建议（按 ROI）

1. **otherfurniture 治理**（重审类别映射 / 重采样 / 拆分类）→ ~+3.5 micro pt 潜力
2. **painting/薄壁挂类 recall**（解码阈值、场景过采样、验证 0.1m 厚度过 Sonata 体素化的存活率）→ ~+2.0 pt
3. **door NMS + 去重**（同墙段保最优、无墙缝证据降权）→ ~+2.0 pt
4. **desk/table 尺寸先验后处理**（36 对干净翻转，footprint 面积可分）
5. toilet 朝向后处理（贴墙约束 + 局部点密度选向）；counter 等类的尺度校准
6. 训练侧：解冻点云塔（`freeze_point_tower: false`）或方案 B 全量微调，主攻中心精度（@.5 最大杠杆）与欠生成
7. 不建议投入泛化角度修复（oracle 仅 +3pt）

## 8. 复现命令速查

```bash
conda activate spatiallm && cd ~/SpatialLM && export HF_HOME=/home/chenle/hf_home_spatiallm

# 训练（预处理缓存 saves/scannet_cache 已存在时直接进入训练；删除则先走预处理+退出再跑一次）
tmux new-session -d -s spatiallm_train -c ~/SpatialLM
tmux send-keys -t spatiallm_train 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate spatiallm && export HF_HOME=/home/chenle/hf_home_spatiallm && python train.py configs/scannet_sft_4090.yaml 2>&1 | tee train.log' Enter

# 批量推理（4 卡分片，逐场景 seed=42 与串行等价，~10 分钟）
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python inference.py -d object \
    -p data/scannet_spatiallm/pcd_val_g$g -o data/scannet_spatiallm/pred_val_g$g \
    --model_path saves/scannet_frozen --seed 42 > infer_g$g.log 2>&1 & done; wait
cp data/scannet_spatiallm/pred_val_g*/*.txt data/scannet_spatiallm/pred_val/

# 评估（18 类完整口径）与误差分析
python eval_scannet18.py --metadata data/scannet_spatiallm/val.csv \
    --gt_dir data/scannet_spatiallm/layout --pred_dir data/scannet_spatiallm/pred_val \
    --label_mapping data/scannet_spatiallm/benchmark_categories.tsv \
    --label_from scannet18 --label_to scannet18
python analyze_errors.py --pred_dir data/scannet_spatiallm/pred_val --tag final
```

## 9. 产物清单

| 路径 | 内容 |
|---|---|
| `saves/scannet_frozen/` | 最终交付模型（+ checkpoint-1500/2000/2500/3000/3010） |
| `train.log` / `train.log.old` | 全新训练日志 / 之前中断运行的存档 |
| `data/scannet_spatiallm/pred_val/` | 最终模型 312 场景预测 |
| `data/scannet_spatiallm/pred_val_2500/` | ckpt-2500 预测（对比用） |
| `data/scannet_spatiallm/error_analysis/` | final_* / ckpt2500_* 逐场景 CSV、匹配对几何、混淆对、summary JSON |
| `eval_scannet18.py` / `analyze_errors.py` | 18 类评估 / 逐场景误差分类工具（本实验新增） |
| `infer_g*.log`, `infer_2500_g*.log` | 分片推理日志 |
