# SpatialLM × ScanNet 微调操作手册（4×4090 服务器）

本手册覆盖从本地数据打包到服务器训练、推理、评估的完整流程。

## 0. 背景与现状

**目标**：在 ScanNet 数据集上微调 SpatialLM1.1-Qwen-0.5B，使其对自己的点云输出结构化场景理解（18 类物体检测，SceneScript 语言格式）。

**已在本机完成**（产物在 `data/scannet_spatiallm/`，共 3.2GB）：

| 文件 | 说明 |
|---|---|
| `pcd/scene*.ply` | 1513 个轴对齐点云（已应用 axis-align 矩阵，z 轴朝上） |
| `layout/scene*.txt` | GT 标注（Bbox 格式，米制连续坐标，由实例分割+语义标签自动生成） |
| `split.csv` | 1201 train / 312 val |
| `scannet_train.json` / `scannet_val.json` / `dataset_info.json` | 训练用 ShareGPT 格式数据 |
| `benchmark_categories.tsv` | 评估用 18 类标签映射 |

数据转换脚本：`prepare_scannet.py`（.pth→PLY+GT）、`create_scannet_sharegpt.py`（→ShareGPT）、`check_conversion.py`（验证，已通过：实例点 100% 落在对应框内）。

**硬件约束**：官方称全量微调需约 60GB 显存，而 4×4090 每张只有 24GB。DDP 不分摊显存（每卡放完整模型+优化器），所以默认配置跑不了，必须选下文方案 A 或 B。

---

## 1. 本地（Windows）操作

### 1.1 打包

在 **PowerShell** 中执行（打包代码 + 转换好的数据，约 3.3GB，排除 12GB 原始数据）：

```powershell
cd C:\Users\cl759\Projects\SpatialLM

tar --exclude data/scannet --exclude .git --exclude saves `
    -czf spatiallm_scannet.tar.gz -C .. SpatialLM
```

### 1.2 传输到服务器

```powershell
# 替换 user@server 为实际地址；确认服务器目标目录存在
scp .\spatiallm_scannet.tar.gz user@server:~/

# 若 ssh 走非标准端口：
# scp -P 22222 .\spatiallm_scannet.tar.gz user@server:~/
```

> 大文件传输慢可先压缩分卷或用 rsync（如服务器允许）。3.3GB 一般几分钟内。

### 1.3 （可选）回传结果到本地

训练完成后，把**最终模型**（不含优化器状态，约 3GB）和可视化文件传回：

```powershell
scp -r user@server:~/SpatialLM/saves/scannet_frozen .\saves\
scp user@server:~/SpatialLM/*.rrd .
```

本地查看 `.rrd` 可视化：`pip install rerun-sdk`，然后 `rerun xxx.rrd`。
本地推理不推荐（需要装 CUDA 版 torch + open3d），建议都在服务器做。

---

## 2. 服务器（Linux, 4×4090）操作

### 2.1 解压

```bash
mkdir -p ~/SpatialLM && cd ~/SpatialLM
tar -xzf ~/spatiallm_scannet.tar.gz --strip-components=1
ls data/scannet_spatiallm/   # 应看到 pcd layout split.csv *.json benchmark_categories.tsv
```

### 2.2 安装环境

```bash
# conda 环境（Python 3.11 + CUDA 12.4 工具链）
conda create -n spatiallm python=3.11 -y
conda activate spatiallm
# cuda-toolkit 供 flash-attn 编译用；若 pip 能装上预编译 wheel 则可跳过
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash

# 依赖
cd ~/SpatialLM
pip install poetry && poetry config virtualenvs.create false --local
poetry install

# SpatialLM1.1 依赖（Sonata 编码器：flash-attn / torch-scatter / spconv）
poe install-sonata      # flash-attn 若无预编译 wheel，编译约需 10-30 分钟

# 训练栈
poe install-training

# 仅方案 B（全量微调）需要 DeepSpeed：
pip install deepspeed
```

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# 应输出 2.4.1+cu124 和 4
```

**网络提示**（国内服务器下载 HuggingFace 模型慢/失败时）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.3 显存方案选择（必读）

| 方案 | 配置文件 | 说明 | 适用 |
|---|---|---|---|
| **A（推荐先用）** | `configs/scannet_sft_4090.yaml` | 冻结点云塔（`freeze_point_tower: true`）+ bf16 混合精度 + 梯度检查点（默认开）。Sonata 编码器本身在 ScanNet 类数据上预训练过，冻结它只训 LLM+投影层，显存和速度都友好 | 24GB/卡 |
| **B（全量微调）** | `configs/scannet_sft_full_ds.yaml` | 点云塔+语言塔全开，用 DeepSpeed ZeRO-2 把优化器状态和梯度分片到 4 卡 | 追求上限，24GB/卡 |

两份配置都改好了（含 `eval_dataset: scannet_val`、`report_to: none`、独立的 `output_dir`），无需再改。

**OOM 降级阶梯**（哪个方案 OOM 都按此顺序尝试）：
1. `per_device_train_batch_size` 已是 1，不动；先确认报错是训练 OOM 还是数据预处理 OOM
2. 方案 B 下 ZeRO-2 换 ZeRO-3（`"stage": 3`，同时把 yaml 里 `pure_bf16` 保持 false——它和 ZeRO-3 互斥）
3. `cutoff_len: 8192` 降到 4096（会截断超长 GT，牺牲一点大场景标注）
4. 方案 B 放弃，转方案 A；方案 A 再 OOM 则 `freeze_language_tower: true`（只训投影层，效果最差但最省）

### 2.4 启动训练

```bash
cd ~/SpatialLM
conda activate spatiallm

# 方案 A
python train.py configs/scannet_sft_4090.yaml

# 方案 B（全量微调 + ZeRO-2）
python train.py configs/scannet_sft_full_ds.yaml
```

`train.py` 检测到 4 张卡会自动用 torchrun 起 4 个 DDP 进程，**无需手动设置** MASTER_ADDR 等环境变量（单机多卡场景）。

- 首次启动会先下载基座模型 `manycore-research/SpatialLM1.1-Qwen-0.5B`（约 2GB）并预处理数据（存入 `save_dir`），然后开始训练
- 数据规模：1201 样本 × 10 epoch ÷ 4 卡 ≈ 3000 步；4090 上预计 **3~8 小时**（方案 A 快于方案 B）
- 每 500 步存一个 checkpoint（`output_dir/checkpoint-XXXX`，最多保留 5 个）并跑一次验证
- 训练结束 `trainer.save_model()` 把**最终模型**存到 `output_dir` 根目录（如 `saves/scannet_frozen/`）

### 2.5 监控与中断恢复

```bash
# 另开终端看显存（方案 A 单卡建议 <20GB，方案 B 单卡 <22GB）
watch -n 5 nvidia-smi

# 看训练日志（loss 应从 ~1x 降到 0.x 并持续走低）
tail -f saves/scannet_frozen/trainer_log.jsonl 2>/dev/null || ls saves/scannet_frozen/
```

**中断恢复**：直接重跑同一条训练命令即可，框架会自动从 `output_dir` 里最新的 checkpoint 续训（`overwrite_output_dir: false` 时这是默认行为）。

### 2.6 推理

单场景：

```bash
python inference.py -d object \
    -p data/scannet_spatiallm/pcd/scene0011_00.ply \
    -o pred.txt \
    --model_path saves/scannet_frozen \
    --seed 42
```

- `-d object` = 只检测物体（与训练任务一致）
- 不传 `-c` 类别时检测全部 18 类；`inference.py` 里 `--category` 的 choices 白名单是官方 59 类，要按类别筛选时需自行把 choices 改成这 18 类
- `--seed` 固定可复现；追求精度可加 `--num_beams 3`（更慢）

val 集批量推理（312 个场景，为评估做准备）：

```bash
# 建 val 场景的软链接目录（避免推理到 train 场景）
cd data/scannet_spatiallm
mkdir -p pcd_val
awk -F, 'NR>1 && $2=="val"{print $1}' split.csv | while read s; do ln -sf ../pcd/$s.ply pcd_val/; done
# 生成评估用元数据
echo "id" > val.csv
awk -F, 'NR>1 && $2=="val"{print $1}' split.csv >> val.csv
cd ~/SpatialLM

python inference.py -d object \
    -p data/scannet_spatiallm/pcd_val \
    -o data/scannet_spatiallm/pred_val \
    --model_path saves/scannet_frozen \
    --seed 42
```

### 2.7 定量评估（F1@IoU）

```bash
python eval.py \
    --metadata data/scannet_spatiallm/val.csv \
    --gt_dir data/scannet_spatiallm/layout \
    --pred_dir data/scannet_spatiallm/pred_val \
    --label_mapping data/scannet_spatiallm/benchmark_categories.tsv \
    --label_from scannet18 --label_to scannet18
```

会输出逐类 + 总体的 F1@0.25/0.50 IoU 表。参考：官方 SpatialLM1.1 在 ScanNet 18 类上 F1@.25 = 65.6（先在官方数据上训练再微调的数字）；你的设置（仅 1201 个场景直接微调）达到 50~60@.25 即属正常。

对比不同 checkpoint：把 `--model_path` / `--pred_dir` 换成对应 checkpoint 重跑 2.6/2.7 即可。

### 2.8 可视化（rerun）

```bash
pip install rerun-sdk   # 若未装
python visualize.py \
    -p data/scannet_spatiallm/pcd/scene0011_00.ply \
    -l pred.txt \
    --save scene0011_00.rrd

# 服务器有桌面环境可直接：rerun scene0011_00.rrd
# 无桌面：把 .rrd 传回本地看（见 1.3）
```

---

## 3. 用自己新采集的点云做推理

训练好的模型对**任意同源点云**可用，只要满足：

1. **PLY 格式带 RGB**，米制单位
2. **z 轴朝上**（重力对齐即可，xy 不需要对齐——训练时 `random_rotation: true` 已增强）
3. 单场景范围原则上 <32m×32m×25.6m（超大场景需先缩放或分块）

```bash
python inference.py -d object -p your_scene.ply -o your_scene.txt --model_path saves/scannet_frozen
```

输出即结构化场景理解结果（`Bbox(类别,x,y,z,朝向角,长宽高)` 逐行）。

---

## 4. 常见问题

| 问题 | 处理 |
|---|---|
| CUDA OOM | 按 2.3 的降级阶梯逐项尝试 |
| flash-attn 编译失败 | 确认 `nvcc --version` 是 12.4；或去 [flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases) 找 torch2.4+cu12+py311 的预编译 wheel 直接 pip 安装 |
| 下载基座模型卡住 | `export HF_ENDPOINT=https://hf-mirror.com` 后重试；或本地 `huggingface-cli download` 后 scp 上去 |
| 训练时想看曲线 | yaml 里 `report_to: wandb`，先 `pip install wandb && wandb login`；不想用就保持 `none` |
| 二次训练报 "Output directory already exists" | 换 `output_dir`，或删掉旧目录，或确认是否想让它自动续训（有 checkpoint 时会自动续） |
| 想更快收敛/效果更好 | `learning_rate: 5.0e-5`；多 epoch 会过拟合，关注 eval loss 挑最优 checkpoint |
| 推理结果里出现训练没见过的类别名 | LLM 偶发幻觉，正常现象；评估时会被 label_mapping 过滤掉 |

---

## 附：本仓库新增文件一览

| 文件 | 用途 |
|---|---|
| `prepare_scannet.py` | ScanNet .pth → PLY + GT layout 转换（含轴对齐、实例 OBB 提取） |
| `create_scannet_sharegpt.py` | 生成 ShareGPT 训练数据（免训练栈依赖版本） |
| `check_conversion.py` | 转换质量验证脚本 |
| `configs/scannet_sft_4090.yaml` | 方案 A：冻结点云塔微调 |
| `configs/scannet_sft_full_ds.yaml` | 方案 B：全量微调 + DeepSpeed ZeRO-2 |
| `configs/ds_zero2.json` | DeepSpeed ZeRO-2 配置 |
| `data/scannet_spatiallm/` | 转换后的数据集（gitignore，不入库） |
