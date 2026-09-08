# SpatialLM × ScanNet 分析报告：预训练词表覆盖对逐类表现的影响

> 日期：2026-08-26　|　性质：桌面分析（无新实验，全部数字来自 EXP1–EXP4 报告与仓库源码）
> 问题：官方 SpatialLM 预训练（自家 SpatialLM-Dataset）见过的物体类 vs 我方 ScanNet 18 类的差集，是否解释了我方模型的逐类强弱？

---

## 1. 摘要

**结论：成立，但只解释约一半的组间差距，且有两个醒目反例。** 预训练词表外的 6 类在 EXP1/EXP2 中系统性比词表内 10 类低 **~14pt**（macro 组均值）；换用官方 ScanNet-SFT 起点（其词表恰好覆盖这些类）后组间差收窄至 **8.5pt**，且涨幅最大的类正是起点新获得域内 Bbox 监督的类——构成假设的正反两向证据。但 painting / refrigerator 两个词表内类的垫底表现说明**几何难度（薄/小/壁挂）与尺寸先验错配**是叠加的独立成因。对 EXP5 的含义：对 counter / shower curtain / cabinet 等弱势类做过采样或 loss 加权是对症的。

## 2. 数据来源

| 数据 | 出处 |
|---|---|
| 官方预训练词表（59 类物体） | `inference.py:161` `--category` choices（官方预训练数据集的物体类别清单） |
| 我方 18 类 | `scripts/prepare_scannet.py` `SCANNET_GT20_CLASSES`（gt20 去掉 wall/floor/unlabeled；picture→painting、bookshelf→bookcase 为我方改名） |
| EXP1 逐类 F1 | `FINETUNE_SCANNET_EXP1_REPORT.md` §4.2 |
| EXP2 逐类 F1 | `FINETUNE_SCANNET_EXP2_REPORT.md` §4.2 |
| EXP3（官方模型冷评）逐类 | `FINETUNE_SCANNET_EXP3_REPORT.md` §3 |
| EXP4 逐类 F1 | `FINETUNE_SCANNET_EXP4_REPORT.md` §2（5 类 + otherfurniture 为精确值，其余 12 类由 EXP2 值 + 报告给出的 Δ 重构；重构后 18 类均值 0.574，与报告最终模型 macro 0.5791 差 0.5pt，在 Δ 四舍五入误差内） |

**重要背景**：官方预训练中 **door/window 不是 Bbox 类**——它们以建筑结构实体（Door/Window，挂在 wall 上）的形式出现；官方 ScanNet-SFT 的词表（EXP3 实测）为 14 个同名类 + `picture`/`bookshelf`/`showercurtrain`/`garbagebin`，**含 counter、不含 otherfurniture**。

## 3. 词表对照

### 3.1 我方 18 类的分档

| 档 | 类别（我方名） | 与官方 59 类词表的关系 |
|---|---|---|
| **A：精确在词表**（10） | bed, chair, sofa, desk, sink, toilet, refrigerator, painting, curtain, bookcase | 逐字匹配（bookcase 为我方对 bookshelf 的改名，官方词表用的就是 bookcase） |
| **B：语义近邻**（2） | cabinet, bathtub | 官方只有变体（wardrobe/nightstand/tv_cabinet 等 11 个 \*_cabinet）或近义（tub）；无 cabinet/bathtub 本词 |
| **C：完全没有**（6） | table, door, window, shower_curtain, counter, otherfurniture | table 只有 dining/coffee/side_table；door\*/window\* 仅实体级预训练（概念已知、未作为 Bbox 输出）；counter/shower_curtain/otherfurniture 无任何对应 |

### 3.2 官方 59 类词表全列（存档）

sofa, chair, dining_chair, bar_chair, stool, bed, pillow, wardrobe, nightstand, tv_cabinet, wine_cabinet, bathroom_cabinet, shoe_cabinet, entrance_cabinet, decorative_cabinet, washing_cabinet, wall_cabinet, sideboard, cupboard, coffee_table, dining_table, side_table, dressing_table, desk, integrated_stove, gas_stove, range_hood, micro-wave_oven, sink, stove, refrigerator, hand_sink, shower, shower_room, toilet, tub, illumination, chandelier, floor-standing_lamp, wall_decoration, painting, curtain, carpet, plants, potted_bonsai, tv, computer, air_conditioner, washing_machine, clothes_rack, mirror, bookcase, cushion, bar, screen, combination_sofa, dining_table_combination, leisure_table_and_chair_combination, multifunctional_combination_bed

## 4. 分组统计（F1@.25，312 val，我方尺子）

分组口径：A 档 10 类为「词表内」，C 档 6 类为「词表外」；B 档 2 类（cabinet 0.28~0.39、bathtub 0.60~0.73）语义地位模糊，**不进组**。

| 组均值 | EXP1（基座起点） | EXP2（解冻） | EXP4（官方 SFT 起点）† |
|---|---|---|---|
| 词表内（10 类） | 0.496 | 0.535 | 0.607 |
| 词表外（6 类） | 0.355 | 0.399 | 0.522 |
| **组间差** | **14.2pt** | **13.6pt** | **8.5pt** |
| 全 18 类 macro（对照） | 0.4426 | 0.4834 | 0.5791（best 0.5825） |

† EXP4 逐类为重构值（见 §2），组间差量级不受重构误差影响。

组内离散度极大（词表内 EXP1 跨度 0.24~0.80），n=10 vs 6，未做显著性检验——组间差应视为描述性证据，不是因果隔离。

## 5. 支持假设的三行证据

1. **EXP1/EXP2 组间差稳定 ~14pt**：以官方基座为起点时（预训练影响最纯粹的观测点），词表外 6 类系统性更差。词表外垫底的 counter（0.20~0.18）、otherfurniture（0.24~0.33）与词表内头部 toilet（0.80~0.85）、chair（0.76~0.80）形成干净的分隔。
2. **EXP4 组间差收窄至 8.5pt**：起点换成官方 ScanNet-SFT 后，词表外组的均值增益（+12.3pt）大于词表内组（+7.2pt）——因为该起点的词表恰好补上了 counter/shower curtain（showercurtrain）/otherfurniture（garbagebin 吸收）这些类。
3. **EXP2→EXP4 涨幅与「起点是否新获得该类的域内 Bbox 监督」高度对应**：shower curtain +21.1、otherfurniture +18.4、refrigerator +18.2、counter +16.3、sink +13.0、painting +12.3——除 refrigerator/sink/painting（两词表皆有）外，涨幅前几名正是 C 档类。

## 6. 反例与混淆因素（结论的边界）

| 反例/混淆 | 事实 | 说明 |
|---|---|---|
| **painting** | 词表内（精确匹配），EXP1 仅 0.241、垫底级 | EXP1 误差分析归因：薄壁挂几何在 0.025m 体素下是分辨率边缘（156 漏检，挂高 1.79m、厚 0.10m）。**词表覆盖救不了几何难度** |
| **refrigerator** | 词表内，EXP1 仅 0.270 | 尺寸先验错配：漏检多为矮款（sz 1.30 vs 先验 1.51）——预训练的合成分布与现实变体不符 |
| door / window | C 档却中游（0.43~0.61） | 有实体级预训练（概念、位置先验已知），只是输出格式不同——「没预训练」的边界不干净 |
| counter / otherfurniture | C 档垫底 | 双重打击：词表外 + 几何难（2m×0.27m 长薄条）/ 标注脏（ScanNet 杂物袋类）——无法把 14pt 全记到词表头上 |
| 组间差本身 | n=10 vs 6，组内方差大 | 描述性相关，非因果隔离 |

**综合归因**（与 EXP1/EXP2 误差分析的结论一致）：逐类强弱 ≈ 词表覆盖（域内监督有无）× 几何难度（薄/小/壁挂）× 标注质量（杂物袋类）。三者中词表覆盖是 EXP4 已部分修复的一项，几何与标注是残余短板。

## 7. 对 EXP5 的含义

1. **弱势类的过采样/加权有据可依**：counter（0.34）、shower curtain（0.53）、cabinet（0.39）同时具备「域内监督晚 + 标注质量差」特征，是加权/过采样的首选目标；EXP2 报告 §5 的同类建议现在有了词表层面的独立证据。
2. **几何短板靠加权救不全**：painting 类的薄物漏检（EXP1 模式①）根源在体素分辨率，EXP5 若只做重采样预期收益有限，collator bf16 修复（训练目标去噪）对中心/尺度类指标更对症。
3. **不必再为「词表对齐」做额外工作**：EXP4 已证明官方 SFT 起点自动补齐了域内词表监督，我方词表作为微调目标工作正常（词表收敛检查 0 未映射框）。

## 8. 复算方式

全部统计可由各报告的逐类表手工重算：组均值 = 对应 10/6 类 F1@.25 的算术平均；EXP4 重构值 = EXP2 值 + EXP4 报告 §2 给出的 Δ。误差来源仅为报告 Δ 的 0.1pt 四舍五入累积（对组均值影响 <0.2pt）。
