# SpatialLM × ScanNet 服务器部署与训练 Runbook（供 AI Agent 执行）

> **本文档的主要读者是服务器上的 AI Agent。** 任务：从零把 SpatialLM 在 ScanNet 上的微调训练跑起来。
> 人类用户只需完成 §1（本地打包上传），其余 §3–§6 由 Agent 执行。

---

## 给 Agent 的执行规则（必读）

1. **任务边界**：从 §3 开始执行，到 §6 训练健康运行为止。**不要**自行执行 §7（推理/评估）——那是用户明确指示后才做的后续步骤。
2. **逐步执行，不得跳步**：每个步骤都遵循「命令 → 验证 → 失败处理」三段。验证不通过时先按"失败处理"操作；仍无法解决就**停下**，把完整报错、你已尝试的操作、当前状态报告给用户，等待指示。
3. **先查后做（幂等）**：每个步骤开头有"跳过条件"。如果该步骤的产物已存在且验证通过，跳过并在最终报告中注明，不要重复执行。
4. **只改允许改的东西**：唯一允许编辑的是 `configs/*.yaml` 中本文档明确指出的字段。不要修改 `scripts/`、`train.py`、`spatiallm/` 框架源码。
5. **如实报告**：向用户汇报时粘贴真实输出，不要转述或臆测。验证命令的预期输出写了具体值的，必须实际比对。
6. **长任务**：环境安装中 flash-attn 编译可能需要 10–30 分钟，训练需要数小时。启动长任务后用日志轮询确认进展，不要提前判定失败或反复重启。
7. **用户没有告诉你答案的事情（如 sudo 密码、找不到数据集）**：报告并询问，不要自行猜测绕过。

---

## 0. 已知环境信息

| 项 | 值 |
|---|---|
| 服务器 | Linux，4× RTX 4090（每卡 24GB），已装 conda |
| 代码包 | `~/spatiallm_code.tar.gz`（用户已上传，几百 KB） |
| 数据集 | 服务器上已有 Pointcept 预处理格式的 ScanNet（.pth），**路径未知**，§5 先定位 |
| 训练方案 | 默认**方案 A**（`configs/scannet_sft_4090.yaml`，冻结点云塔）；方案 B 仅在用户明确要求时使用 |

**关键路径速查**（执行中统一使用这些变量）：

```bash
REPO=~/SpatialLM                                  # 代码根目录
DATA=$REPO/data/scannet_spatiallm                 # 生成的训练数据
TRAIN_CFG=$REPO/configs/scannet_sft_4090.yaml     # 方案 A 配置
OUTPUT_DIR=$REPO/saves/scannet_frozen             # 训练输出
SCANNET_SRC=<§5 中定位，例如 /data/scannet>       # 服务器上的原始 ScanNet
```

## 任务完成标准（达成即停）

- [ ] §3 代码解压完整
- [ ] §4 环境验证通过（torch 2.4.1+cu124、4 卡可见、flash-attn 可导入）
- [ ] §5 数据生成完成，`check_conversion.py` 输出 `mean containment: 1.000`
- [ ] §6 训练进程存活、4 卡有占用、日志出现 loss 且数值合理（首次 <10，逐渐下降）
- [ ] 已向用户报告：训练日志路径、checkpoint 目录、预计完成时间

---

## 1. 用户本地步骤（Windows PowerShell，Agent 不执行，仅供参照）

```powershell
cd C:\Users\cl759\Projects\SpatialLM
tar --exclude data --exclude .git --exclude saves `
    -czf spatiallm_code.tar.gz -C .. SpatialLM
scp .\spatiallm_code.tar.gz user@server:~/
```

## 2. 背景（为什么这么做）

官方称全量微调需 ~60GB 显存；4×4090 每卡 24GB，DDP 不分摊显存，因此默认配置跑不了。方案 A 冻结点云编码器（Sonata 本身已在 ScanNet 类数据上预训练，冻结不影响领域迁移的主要收益），单卡 24GB 可行。数据转换流水线（`prepare_scannet.py` → PLY+GT → `create_scannet_sharegpt.py` → ShareGPT）已在本地全量 1513 场景验证过：实例点 100% 落在生成的框内，类别 100% 匹配。

---

## 3. 解压代码

**跳过条件**：`~/SpatialLM/scripts/prepare_scannet.py` 已存在且 §3.1 验证通过。

### 3.1 命令

```bash
mkdir -p ~/SpatialLM && cd ~/SpatialLM
tar -xzf ~/spatiallm_code.tar.gz --strip-components=1
```

### 3.2 验证

```bash
ls ~/SpatialLM/scripts/prepare_scannet.py \
   ~/SpatialLM/scripts/create_scannet_sharegpt.py \
   ~/SpatialLM/scripts/check_conversion.py \
   ~/SpatialLM/configs/scannet_sft_4090.yaml \
   ~/SpatialLM/configs/scannet_sft_full_ds.yaml \
   ~/SpatialLM/configs/ds_zero2.json \
   ~/SpatialLM/train.py ~/SpatialLM/inference.py ~/SpatialLM/pyproject.toml
```

**预期**：所有文件存在，无 "No such file"。

### 失败处理

- `~/spatiallm_code.tar.gz` 不存在 → 报告用户"代码包未上传"，停止。
- tar 报损坏 → `gzip -t ~/spatiallm_code.tar.gz` 确认后报告用户重新上传。

---

## 4. 安装环境

**跳过条件**：`conda activate spatiallm` 成功，且 §4.4 验证全部通过。

### 4.1 创建 conda 环境

```bash
conda create -n spatiallm python=3.11 -y
conda activate spatiallm
# cuda-toolkit 供 flash-attn 编译；若后续 pip 能装预编译 wheel，此步其实可跳过，
# 但为稳妥先装（装过会命中缓存，重复执行无副作用）
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash
```

### 4.2 安装 Python 依赖

```bash
cd ~/SpatialLM
pip install poetry && poetry config virtualenvs.create false --local
poetry install          # 解析安装 torch 2.4.1+cu124、transformers、open3d 等
```

**预期**：`poetry install` 结束无 ERROR（warning 可忽略；下载 torch 约 2.5GB，耐心等）。

### 4.3 安装 Sonata 编码器依赖 + 训练栈

```bash
poe install-sonata     # flash-attn / torch-scatter / spconv；flash-attn 无预编译 wheel 时编译 10-30 分钟
poe install-training   # omegaconf / datasets / accelerate / wandb
```

**预期**：两条命令退出码 0。
**失败处理（flash-attn 编译失败）**：
1. `nvcc --version` 确认 12.4；不是则回到 4.1 确认 cuda-toolkit 装好；
2. 去 https://github.com/Dao-AILab/flash-attention/releases 找 `torch2.4+cu12` + `py311` 的预编译 wheel，`pip install <wheel_url>`；
3. 仍失败 → 报告用户，附编译日志最后 50 行。

> 方案 B 才需要 `pip install deepspeed`。默认方案 A **跳过**。

### 4.4 环境验证（必须全部通过才继续）

```bash
conda activate spatiallm
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# 预期：2.4.1+cu124 4

python -c "import flash_attn; print(flash_attn.__version__)"
# 预期：版本号，无 ImportError

python -c "import transformers, datasets, accelerate; print(transformers.__version__)"
# 预期：4.x（<=4.46.1）
```

---

## 5. 定位数据集并生成训练数据

**跳过条件**：`$DATA/scannet_train.json` 与 `$DATA/pcd/` 下 .ply 数量一致且 §5.4 验证通过。

### 5.1 定位服务器上的 ScanNet（解析 SCANNET_SRC）

```bash
# 先试常见位置
ls -d /data/scannet /dataset/scannet ~/scannet ~/data/scannet /mnt/*/scannet 2>/dev/null

# 找不到再全盘找标志文件（maxdepth 限制避免扫太久）
find / -maxdepth 5 -name "scannet_axis_align_matrix_trainval.pkl" 2>/dev/null | head -5
```

**确定 SCANNET_SRC 后验证结构**（该目录下必须三样齐全）：

```bash
export SCANNET_SRC=/找到的路径
ls $SCANNET_SRC/train/*.pth | wc -l   # 预期 ~1201（≥1000 即可）
ls $SCANNET_SRC/val/*.pth | wc -l     # 预期 ~312（≥100 即可）
ls $SCANNET_SRC/scannet_axis_align_matrix_trainval.pkl
```

**失败处理**：三样缺任一或数量差距大 → 把实际目录结构（`ls` 输出）报告用户，停止。数量略不同没关系（split 按实际文件生成）。

### 5.2 转换（约 3-10 分钟，CPU 任务）

```bash
cd ~/SpatialLM && conda activate spatiallm
python scripts/prepare_scannet.py -s $SCANNET_SRC -d data/scannet_spatiallm
```

**预期**：进度条走完后输出 `Done. 15xx scenes -> data/scannet_spatiallm`，且：

```bash
ls data/scannet_spatiallm/pcd/*.ply | wc -l      # 与源 .pth 总数一致
ls data/scannet_spatiallm/layout/*.txt | wc -l   # 同上
head -2 data/scannet_spatiallm/split.csv         # id,split 表头 + 场景行
head -3 data/scannet_spatiallm/benchmark_categories.tsv  # 18 类标签映射
```

### 5.3 抽查转换质量（必须通过）

```bash
python scripts/check_conversion.py scene0000_00 $SCANNET_SRC
```

**预期输出末尾两行**：

```
mean containment: 1.000, >99%: 26/26
class match: 26/26
```

（`26/26` 是 scene0000_00 的实例数；不同数据副本可能略有差异，**关键是 mean containment 必须为 1.000（或 ≥0.999），class match 必须满分**。）

**失败处理**：containment < 0.999 → 说明服务器数据副本与本地验证过的格式有出入，报告用户（附 check 输出），停止。

### 5.4 生成 ShareGPT 训练数据（约 1 分钟）

```bash
python scripts/create_scannet_sharegpt.py
```

**预期**：

```
data/scannet_spatiallm/scannet_train.json: 1201 samples
data/scannet_spatiallm/scannet_val.json: 312 samples
data/scannet_spatiallm/dataset_info.json updated
```

（样本数与你的 train/val 场景数一致即可。）

---

## 6. 启动训练（方案 A）

**跳过条件**：训练已在运行（`pgrep -f "train.py"` 有进程且日志正常）——此时只做 §6.4 健康检查。

### 6.1 启动前检查

```bash
nvidia-smi   # 确认 4 卡空闲（显存占用接近 0）。有其他任务占卡 → 报告用户，询问是否等待
```

### 6.2 后台启动训练

```bash
cd ~/SpatialLM && conda activate spatiallm
# 国内服务器下载 HuggingFace 模型慢时取消下一行注释
# export HF_ENDPOINT=https://hf-mirror.com

nohup python train.py configs/scannet_sft_4090.yaml > train.log 2>&1 &
echo $! > train.pid
```

`train.py` 检测到 4 卡会自动 torchrun 起 4 个 DDP 进程，无需手动设置 MASTER_ADDR 等变量。

### 6.3 启动阶段监控（前 20 分钟）

首次启动依次经历：下载基座模型 SpatialLM1.1-Qwen-0.5B（~2GB）→ 数据预处理（存入 `saves/scannet_cache`）→ 开始训练。每 2 分钟看一次日志：

```bash
tail -50 train.log
nvidia-smi --query-gpu=index,memory.used --format=csv
```

**健康标志**（出现即进入 §6.4）：
- 日志出现 `loss` 字样的训练指标行（`logging_steps: 10`，每 10 步一行），首个 loss 通常在 1~10 之间
- 4 张卡都有显存占用（方案 A 预计每卡 10~20GB）

**启动阶段失败处理**：

| 症状 | 处理 |
|---|---|
| 模型下载卡住/超时 | `pkill -f train.py` 等待 10 秒确认进程清空；`export HF_ENDPOINT=https://hf-mirror.com` 后重新 §6.2 |
| `CUDA out of memory`（训练刚开始就炸） | 按下方 OOM 阶梯逐项降级，每改一项重启训练观察 10 分钟 |
| 报 `Output directory already exists...` 且不想续训 | `rm -rf saves/scannet_frozen` 后重启（想续训则什么都不用做，框架自动从最新 checkpoint 续） |
| dataloader/预处理 OOM（CPU 内存） | `preprocessing_num_workers: 8` 降到 2，`dataloader_num_workers: 8` 降到 2 后重启 |

**OOM 降级阶梯**（只改 `configs/scannet_sft_4090.yaml`，改一项试一次）：
1. `cutoff_len: 8192` → `4096`
2. `freeze_language_tower: false` → `true`（只训投影层，最省）
3. 仍 OOM → 报告用户（附 train.log 中 OOM 前后各 30 行），停止

### 6.4 训练健康确认（任务完成的判定）

满足以下全部条件后，**任务完成**：

```bash
pgrep -f train.py >/dev/null && echo "进程存活"
grep -c "'loss'" train.log    # >0，且最新一行的 loss 比首个 loss 明显下降（可在启动 30-60 分钟后复核一次）
nvidia-smi                    # 4 卡均有占用
```

**向用户报告**（然后停止，不要继续 §7）：

```
✅ 训练已启动并运行健康
- 配置：方案 A（configs/scannet_sft_4090.yaml），数据 <N> 个训练场景
- 日志：tail -f ~/SpatialLM/train.log
- checkpoint：~/SpatialLM/saves/scannet_frozen/checkpoint-XXXX（每 500 步，最多 5 个）
- 最终模型：训练结束后存到 ~/SpatialLM/saves/scannet_frozen/
- 预计时长：3~8 小时（约 3000 步）
- 中断恢复：重跑 python train.py configs/scannet_sft_4090.yaml 会自动从最新 checkpoint 续训
```

---

## 7. 训练完成后（用户指示后才执行）

### 7.1 单场景推理

```bash
python inference.py -d object \
    -p data/scannet_spatiallm/pcd/scene0011_00.ply \
    -o pred.txt --model_path saves/scannet_frozen --seed 42
```

### 7.2 val 批量推理（312 场景）

```bash
cd data/scannet_spatiallm
mkdir -p pcd_val
awk -F, 'NR>1 && $2=="val"{print $1}' split.csv | while read s; do ln -sf ../pcd/$s.ply pcd_val/; done
echo "id" > val.csv
awk -F, 'NR>1 && $2=="val"{print $1}' split.csv >> val.csv
cd ~/SpatialLM
python inference.py -d object -p data/scannet_spatiallm/pcd_val \
    -o data/scannet_spatiallm/pred_val --model_path saves/scannet_frozen --seed 42
```

### 7.3 F1@IoU 评估

```bash
python eval.py \
    --metadata data/scannet_spatiallm/val.csv \
    --gt_dir data/scannet_spatiallm/layout \
    --pred_dir data/scannet_spatiallm/pred_val \
    --label_mapping data/scannet_spatiallm/benchmark_categories.tsv \
    --label_from scannet18 --label_to scannet18
```

参考锚点（来源 `README.md` Benchmark Results → 3D Object Detection）：官方 SpatialLM1.1-Qwen-0.5B 在**同协议**（SpatialLM-Dataset 预训练 checkpoint + ScanNet 1201 场景微调 + val 312 评测）下 F1@.25 = 65.6。该数字是全量微调 + 官方未公开的训练配方与 GT 生成方式取得的；我们方案 B（全量微调）与之基本同量级，方案 A（冻结点云塔）预期略低。评测用的 GT 由本流水线自动生成，与官方评测 GT 未必逐框一致，数字只作量级参照，不作达标线。

### 7.4 可视化（rerun）

```bash
python visualize.py -p data/scannet_spatiallm/pcd/scene0011_00.ply -l pred.txt --save scene0011_00.rrd
```

### 7.5 方案 B（全量微调，仅用户要求时）

```bash
pip install deepspeed   # 方案 B 额外依赖
python train.py configs/scannet_sft_full_ds.yaml > train_full.log 2>&1 &
```

OOM 时：ZeRO-2 → ZeRO-3（`configs/ds_zero2.json` 改 `"stage": 3`；yaml 中 `pure_bf16` 必须为 false）→ `cutoff_len` 降 4096 → 放弃转方案 A。

### 7.6 用户新采集点云的推理要求

PLY 带 RGB、米制、z 轴朝上（xy 无需对齐）、单场景 <32m×32m×25.6m。

---

## 8. 常见问题速查

| 问题 | 处理 |
|---|---|
| CUDA OOM | §6.3 降级阶梯 |
| flash-attn 编译失败 | §4.3 失败处理 |
| HF 下载慢/失败 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 想看训练曲线 | yaml `report_to: wandb` + `wandb login`（默认 none 不影响训练） |
| 推理出现训练外类别名 | LLM 幻觉，正常；评估时被 label_mapping 过滤 |
| 数据集路径结构不同 | `-s` 目录下有 `train/`、`val/`、pkl 三样即可；场景数不同也能跑 |

---

## 附：新增文件一览

| 文件 | 用途 |
|---|---|
| `scripts/prepare_scannet.py` | ScanNet .pth → PLY + GT layout（轴对齐、实例 OBB、18 类），顺带生成 split.csv 与 benchmark_categories.tsv |
| `scripts/create_scannet_sharegpt.py` | 生成 ShareGPT 训练数据（免训练栈依赖） |
| `scripts/check_conversion.py` | 转换质量抽查：`python scripts/check_conversion.py <scene> <src路径>` |
| `scripts/eval_scannet18.py` | 18 类完整口径 F1 评估（stock eval.py 只覆盖 7/18 类，见 experiments/ 报告） |
| `scripts/analyze_errors.py` | 逐场景误差分类与匹配对几何分析 |
| `configs/scannet_sft_4090.yaml` | 方案 A：冻结点云塔微调（默认） |
| `configs/scannet_sft_full_ds.yaml` | 方案 B：全量微调 + DeepSpeed ZeRO-2 |
| `configs/ds_zero2.json` | DeepSpeed ZeRO-2 配置 |
| `experiments/FINETUNE_SCANNET_EXP1.md` | 实验报告 EXP1（方案 A） |
| `experiments/FINETUNE_SCANNET_EXP2.md` | 实验报告 EXP2（方案 B，解冻点云塔） |
| `data/scannet_spatiallm/` | 转换后的数据集（服务器上由 §5 生成，gitignore 不入库） |
