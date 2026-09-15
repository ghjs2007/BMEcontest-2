# 基于智能手表传感器的进食检测算法

第十一届全国大学生生物医学工程创新设计竞赛 · 智能穿戴与运动健康赛道 · 赛题二
（本 README 按试题作品 7 大要素组织，供总结报告写作参考）

---

## 1. 选题背景及意义

- **场景**：智能手表全天候采集 IMU（加速度计+陀螺仪 ~105Hz，raw ADC）与 PPG（44 通道，
  有效采样 ~2Hz）多源传感器数据，为无感自动识别进食行为提供基础。
- **挑战**：进食动作与日常动作（喝水、打电话、刷牙）高度相似；两种佩戴场景检测原理
  本质不同——**惯用手**（手部动作特征明显，IMU 主导）与**非惯用手**（动作弱，需
  上下文/生理信号）。
- **意义**：自由生活进食监测服务于饮食行为量化、慢病管理、营养干预等健康场景；
  全天多传感器数据下的鲁棒事件检测是穿戴健康的核心共性技术。
- **评估**（Resources/试题.txt 锁定）：预测 Episode 与 GT Episode `IoU ≥ 0.25` 判 TP；
  全局灵敏度 = TP/真实事件数、PPV = TP/预测事件数、F1 = 2·Sens·PPV/(Sens+PPV)；
  次要指标：正确匹配事件的起止时间 MAE。组委会对提交的可执行文件在**独立测试集**
  上客观评分。

## 2. 研究目标

1. 惯用手/非惯用手场景下的进食事件（Episode 级）检测，官方全局口径 F1 最大化；
2. 解决进食检测的两大结构瓶颈：**候选覆盖**（活动连通域提案仅覆盖 50-68% 餐）与
   **假阳性控制**（单级排序头无法同时兼顾召回与精度）；
3. 全程使用真实传感器时间戳（包级恢复 + 缺口切段），杜绝时间轴漂移类评估伪影；
4. 数据驱动的两阶段方案：全覆盖滑窗粗召回 → 事件级复核精控精度。

## 3. 设计原理及方案（event-stack 当前发布系统）

```
240s 窗 / 15s 步长全覆盖滑窗（真实时间戳网格，窗不跨缺口，覆盖率 ≥0.8）
  → 62 维 ACC 特征（三轴+幅值+jerk 稳健统计 + 1s 活动包络时间特征）
  → HistGradientBoosting 窗分类器（正：与餐重叠>50%；严格负：距餐≥300s；
     边界窗 -1 不训练；负样本每会话 ≤3× 正窗）
  → macro 候选 ∪ 15s/7.5s ACC+GYRO micro 候选
  → 同会话稳定 NMS + subject admission（阈值/IoU/cap 只由 train OOF 选择）
  → LogisticRegression 与受限 LightGBM 概率 blend（56 维基础复核 + Context-v1 60 维上下文 = 116 维）
  → 冻结 event policy（阈值、事件几何与 subject budget）
  → 官方评估（IoU≥0.25 一对一匹配；eligible 质量审计分母）
```

**设计要点**（当前发布契约；历史对照见 §5）：

1. **全覆盖滑窗替代活动提案**：proposal 依赖"餐时段有 ≥3.8min 连续活动段"（几何
   上仅 47-58% 餐可达 IoU≥0.25），滑窗使每餐必然被多窗覆盖（正窗 1112/折 vs 17），
   候选层 recall 从 0.25 → 0.60-0.84；
2. **两级精度架构**：窗口模型只需"宁滥勿缺"（候选层 PPV 仅 0.1-0.18 也可接受），
   假阳性由事件级复核器压制（PPV 0.1 → 0.4-0.7）——单级排序头试图同时解决
   覆盖与 FP 是旧方案（检测即排序）的根因瓶颈；
3. **事件框 = 支持窗跨度**：自适应宽框天然容忍 GT 时间偏移（用户自报标注中位偏移
   10-13.5min 为真实数据性质，修复时间轴后复测不变）；
4. **标签三态**：重叠 >50% 正 / 距餐 ≥300s 严格负 / 其余 -1 剔除——消除旧方案
   IoU≥0.25 模糊边缘窗的训练噪声；
5. **质量审计（eligible 分母）**：每折 5-20% 餐数据不可达（会话缺失/时段缺口/
   时间错位）——评估分母用可评估餐集合，避免系统性低估（旧方案分母含不可达餐）。
6. **候选控制与 stacking**：macro/micro 并集先做同 `sid` 的确定性 NMS，再按稳定
   `subject_id` 做准入；两个事件复核器在 subject-disjoint OOF 上各打分一次，融合权重、
   准入配置、事件阈值和预算均在 outer-train 内冻结。
7. **发布与追溯**：每次晋级同时保留 canonical 五折 summary、5 个 outer-fold evidence、
   full-target deployment bundle 和 attestation；manifest 记录模型/输入指纹与 SHA-256。
8. **发布运行时 ABI**：event-stack 的 `requirements.txt` 由 deployment `manifest.json` 的
   `dependency_versions` 原样生成，顺序固定为 numpy、joblib、scikit-learn、lightgbm，且全部
   使用 `==` 精确 pin。当前 deployment manifest 为 Python `3.11.15`、numpy `2.4.6`、
   joblib `1.5.3`、scikit-learn `1.9.0`、lightgbm `4.7.0`。Python 不由 requirements 安装，
   但运行环境必须为 Python `3.11.x`；推理会在任一 joblib 模型反序列化前核验 Python 主/次版本
   与上述四个包的精确版本，不匹配即拒绝，不降级为 warning 或尝试加载。

### 当前发布结果（Context-v1，2026-09-15）

当前 release 为 `160afaf81debf1ee`。在预注册的严格五折、四路 inner
subject-disjoint OOF 协议下，唯一改动是给事件复核器追加确定性的 Context-v1：同一 `sid`
内 macro/micro 概率流的前/候选/后 20 分钟统计、固定阈值 run 形态、coverage 和时序集中度。
该模块不读标签、不跨会话、不拟合参数；60 列由 schema-v2 固定，最终复核器宽度为 116。

| fold | config hash | TP/eligible/pred | F1 |
|---|---|---:|---:|
| 0 | `afcf9609a2f6925a` | 16/23/35 | 0.5517241379 |
| 1 | `8b3e27bd2796d038` | 30/31/46 | 0.7792207792 |
| 2 | `b463226db1e6073a` | 15/27/33 | 0.5000000000 |
| 3 | `3f26fcb2172cd882` | 25/32/42 | 0.6756756757 |
| 4 | `0199041f39a09f5d` | 28/40/41 | 0.6913580247 |
| **aggregate** | `160afaf81debf1ee` | **114/153/197** | **0.6514285714** |

PPV=`0.5786802030`、recall=`0.7450980392`、FP=`83`、短餐最终 recall=`20/39=0.5128205128`、
candidate recall=`0.7712418301`、短餐 candidate recall=`0.5384615385`。相对前一已发布
`035644cf0889a5dd` 的 F1=`0.5589743590`，提升 `+0.0924542125`；39 个受试者的 F1
中位数/IQR/p10 分别为 `0.6667` / `[0.5357, 0.7273]` / `0.2667`，配对结果为 24 改善、5
不变、10 变差。它满足技术晋级及推荐门（ΔF1≥0.005 且至少 3/5 folds 不差），但仍是反复
开发后的 CV 证据，不能表述为独立测试集泛化保证。

**对照系统（检测即排序 v2 + FD 预训练）**：多参数提案 + LGBM/TCN 深度双排序 +
会话门控 + 形态学后处理，全局 F1 均值 ~0.27（eligible 校正）。保留作为对照与
消融基准；其 TCN 深度分（AUC 0.83-0.90 窗级）与滑窗 HGB 分相关性仅 0.26，
作为复核器增强特征有融合潜力（实验进行中）。

## 4. 主要算法程序

| 脚本 | 功能 |
|---|---|
| scripts/slide_features.py | 滑窗 62 特征提取（train 采样/含餐会话全窗/无餐负样本/val 全窗；多进程并行） |
| scripts/slide_verifier.py | 窗模型训练 → 密度候选 → 33 特征复核器 → 官方评估（--fold k） |
| scripts/tcn_slide_score.py | TCN 深度模型对滑窗打分（批量 interp + 双缓冲 GPU 流水线） |
| scripts/validate_baselines.py | 1s 活动包络缓存（真实时间戳）+ V1 启发式基线 |
| scripts/build_candidate_windows.py | proposal 候选窗口缓存（对照系统） |
| scripts/train_ranker.py | MM-Ranker TCN 训练（--init-from FD 权重 / NEG_RATIO / AUG_POS 环境变量） |
| scripts/rank_events.py / rank_events_v2.py | 对照系统提案 + 解码 |
| scripts/official_iou_eval.py | 官方评估 + 后处理 + 对比表 |
| scripts/pretrain_fd.py | FD 数据集 Episode 级预训练（checkpoints/fd_pretrained_s1.pt） |
| scripts/slide_features.py --mode train / meal_train / no_meal_train / val | 各模式窗口表 |

运行流程见 §复现。

## 5. 代码得分及结果说明

### 5.1 当前最优（v6 干净协议——受试者互斥、零信息 CV）

第二轮 peer review 指出 wbag 泄漏：评估折 k 时 bag 里的 fold m≠k 模型训练过
折 k val 受试者的会话（受试者互斥划分本身已审计干净——每个受试者全部会话只在
单折 val、各折 train/val 受试者交集 = 0——但跨折模型平均使 bag 见过 val 受试者
的其他会话）。**v5.1 的 0.617/0.632 因此作废**（含约 0.08-0.13 泄漏虚增，见 5.2
对照）。v6 起评估协议：窗模型只用本折 train（∪_{j≠k}S_j，val 受试者完全未见）
训练并打分本折 val；复核器只用本折 train 候选训练（同口径）。

**主配置（TCN 特征开，全折统一——无逐折架构选择）**：

| fold | F1 | sens | ppv | TP/eligible | pred |
|---|---|---|---|---|---|
| 0 | 0.426 | 0.435 | 0.417 | 10/23 | 24 |
| 1 | **0.655** | 0.613 | 0.704 | 19/31 | 27 |
| 2 | 0.418 | 0.519 | 0.350 | 14/27 | 40 |
| 3 | 0.436 | 0.375 | 0.522 | 12/32 | 23 |
| 4 | 0.590 | 0.575 | 0.605 | 23/40 | 38 |
| **均值** | **0.505** | | | | |
| 全局聚合 | **0.512** | 0.510 | 0.513 | 78/153 | 152 |

**部署 CPU 口径（无 TCN，dist 同构）**：单模型均值 **0.449**、聚合 **0.458**
（7+20+9+12+17 = 65/153，pred 131）；5 模型负样本重采样 bag 均值 **0.473**
（bag 仅对 no-TCN 有 +0.024 净增益；TCN 下 bag 反而 -0.015——见 5.3 探索记录）。
严格 LOSO（每留一受试者重训窗模型 + 复核器，固定部署阈值 0.717，CPU 口径）：
38 受试者聚合 **F1 0.416**（67/153，sens 0.438 ppv 0.396；均值受试者 F1 0.388）——
与旧泄漏版 0.407 持平（verifier 泄漏贡献微小），且严格口径 153 vs 旧 180。

与组内平行方案完整嵌套 OOF 0.5191 同量级（其方案无 TCN 特征）。干净协议下
TCN 每折均增益（+0.05~0.12，fold4 +0.075）——v5.1"fold4 TCN 有害"系泄漏伪影。

### 5.2 泄漏审计（第二轮 peer review 回应要点）

| 指控 | 审计结果 |
|---|---|
| LOSO 用全数据 verifier（见过留出受试者） | **成立**——严格 LOSO 已修复（0.407→0.416 @153 口径） |
| 5 折 bag 跨折模型见过 val 受试者 | **成立**——v5.1 0.617 作废；干净单模型 0.505 |
| "fold4 NO_TCN、密度参数、no_meal 数量逐折选择"为验证折适配 | 成立（fold4 NO_TCN 已在干净协议证伪——TCN 全折更优）；no_meal 150 为全局统一设置 |
| 阈值逐折在 val 上选（乐观） | 已测：全局 pooled 阈值 vs 逐折 ≈0.00-0.01（干净协议候选上复核） |
| 口径修正 0.44 / 同协议 0.53 / 部署 0.60 为推算 | 承认——v5.1 与 dist "~0.60" 均为泄漏协议产物，作废；诚实估计：研究配置 ~0.51、CPU 部署 ~0.45-0.47 |

### 5.3 v6 探索记录（干净协议，2026-09-07）

**误差结构**（153 eligible）：TP 78（51%）；**候选层漏 28（18%）——短餐为主**
（漏检餐中位 7.8min vs TP 16.2min，61% <10min）；**复核层拒 42（27%）**——
其中"弱窗口证据"餐（窗概率 pmean 0.44 vs TP 0.66）；复核误报 74——**窗口证据与
真餐完全同强度**（fr.45 0.87 vs 0.85，TCN 分同）且聚集在餐时（本地 12-13/17-19
点）——部分为未记录真实进食，构成 PPV 底噪；候选级 PR 曲线上限 ~0.54。

**窗级 AUC（干净协议，62+TCN+prior）**：fold0 0.880 / fold1 0.927 / **fold2
0.785**（窗口层最弱折）/ fold3 0.899 / fold4 0.897——fold0/3 损失在事件层
（密度/复核）而非窗层；fold2 损失在窗层。

**已测无增益（干净协议统一参数）**：①窗阈值 0.20-0.29 × 密度 min_pos 6-10 网格
（基线 0.288/mp10 最优）——弱餐窗概率根本不越阈，放松只加 FP；②复核器
LR→HGB（0.483）与 +显式上下文计数特征（0.486）；③折内负样本重采样 bag+TCN
（0.490）。

**发现并修复的密度 bug（默认关，BME_DENS_COVFIX=1 可开）**：覆盖率卷积 'same'
段边缘零填充把邻近数据缺口（>75s）的餐段 cov 稀释（31 个越阈窗 cov 仅 0.78
被误杀）——候选层漏 28→16 真实修复，但缺口邻域活动 run 变候选 → 复核 FP 增，
净 -0.01（0.496 vs 0.505）。缺口邻域"未佩戴时间"不应稀释覆盖率——需要复核器
学会接受缺口邻域强证据餐后启用（后续工作）。

### 5.4 版本历史（数值为当时协议下记录；5.1 之前均为 wbag 泄漏协议，仅内部参考）

v5.1（wbag，泄漏协议）：均值 0.617/聚合 0.632——作废，泄漏幅度见 5.2。
v5：per-fold TCN 模式选择（fold4 NO_TCN +0.10）——干净协议证伪，作废。
v4.2：eligible 收紧 + no_meal 150——评估口径与复核负样本选择，保持。
v4：窗模型 5 折 bag（+0.06~0.09）——增益含泄漏成分，bag 结构保留在 dist。
v3/v2/v1：TCN 分融合、复核负样本、时刻先验等演进。

对照系统（FD 预训练微调 + proposal 解码，eligible 校正）均值 ~0.27。
该对照系统的逐折诊断产物已在发布清理中移除；结果作为历史基线保留，当前正式证据以
`outputs/crossfit/summary_035644cf0889a5dd.json` 及其 5 个逐折 JSON 为准。

### 5.5 正式 locked nested 基线（2026-09-08）

`scripts/crossfit_event_stack.py` 现在执行两层受试者互斥交叉拟合：inner 窗模型只给
未见受试者生成 OOF 候选，inner verifier 再给未见受试者候选生成 OOF 分数并选择
单一阈值；阈值冻结后才训练最终 outer-train 模型并评估 untouched outer-val。
因此下表是当前正式 locked 结果；`slide_verifier.py` 的 val 最优阈值仅保留为
`diagnostic_per_fold_optimum`，不得与下表混用。

| fold | config hash | inner F1 | 冻结阈值 | outer TP/eligible/pred | outer F1 | 候选 recall |
|---|---|---:|---:|---:|---:|---:|
| 0 | `bfc9da3529e41979` | 0.580 | 0.578967 | 11/23/35 | 0.379 | 0.652 |
| 1 | `fd13740c8bb6dcb2` | 0.532 | 0.640244 | 19/31/35 | 0.576 | 0.871 |
| 2 | `07c5cbd5ea427303` | 0.568 | 0.487595 | 13/27/66 | 0.280 | 0.741 |
| 3 | `5163cfde46f49819` | 0.559 | 0.543717 | 21/32/81 | 0.372 | 0.812 |
| 4 | `1f4ed6da69c03265` | 0.565 | 0.525790 | 25/40/58 | 0.510 | 0.825 |
| **聚合** | — | — | — | **89/153/275** | **0.416** | **121/153 = 0.791** |

CPU/no-TCN baseline 聚合 sensitivity 0.582、PPV 0.324；短餐（<10min）recall
16/39=0.410，非惯用手 recall 42/90=0.467。5 fold/4 inner splits 在 5 个受限
CPU 进程下墙钟约 24s（各折阶段耗时合计 42.2s）；相同单折二次运行命中内容寻址
缓存，从 13.3s 降至 3.0s。当前 locked 证据位于 `outputs/crossfit/` 并纳入版本控制；
可重建的试验缓存位于 `cache/crossfit/`，按精确规则忽略。

coverage-fix 配套 42 维 verifier 的 nested 消融：候选 recall 从 0.791 升至
0.863（132/153），最终 TP 从 89 升至 103，但 pred 从 275 增至 324，PPV
0.324→0.318，聚合 F1 0.416→0.432。按预注册规则（候选漏下降、PPV 不下降、
F1 至少 +0.01）因 PPV 下降而**拒绝直接启用**；保留为后续 hard-negative/
短餐专用复核实验。当前正式 locked F1 距 0.65 仍差 0.234，下一阶段必须优先
解决 fold2/3 的阈值迁移与餐时高分 FP，而不是继续放宽密度参数。

### 5.6 Subject-budget decoder 消融（2026-09-08）

事件上限 K 仅从每个 outer-train 的 verifier OOF 候选上联合选择，预注册网格固定为
2,3,4,5,6；outer 标签不参与 K 或阈值选择。baseline 五折选择 K=6/6/6/4/6，
TP/eligible/pred=71/153/184，sensitivity 0.464、PPV 0.386、F1 0.421。对应 config
hash 为 `d4ef023ebb47ef13` / `8d666bae3affcb47` / `e9f77082215bd8ee` /
`bb7d6ff2cd686f76` / `261e63e3d8bb5e6e`。

coverage+budget 五折均选择 K=6，TP/eligible/pred=86/153/217，sensitivity 0.562、
PPV 0.396、F1 0.465；hash 为 `5a01228a4082bab1` / `699b1f0b4ac323c4` /
`652d2b9f7cd64b05` / `967809d9818bf81a` / `dfa90527cc8ff5c0`。它相对 uncapped
coverage 提升 F1 +0.033、PPV +0.078，但 sensitivity 下降 0.111，超过预注册允许
的 0.05，故不采纳为默认。短餐 recall 仅 10/39=0.256、非惯用手 41/90=0.456。
最佳结果仍低于 0.55 决策门槛，证明瓶颈主要是 verifier 组内排序信息不足，而非单纯
阈值漂移；下一阶段转入候选内/上下文原始 62 维动作特征聚合与 hard-negative 建模。

### 5.7 Raw-summary verifier 消融（2026-09-08）

将候选内 62 维窗口特征的 mean/std/P10/P90、相对前后 20min 上下文差与窗口计数
拼接到原 verifier，共 349 维（coverage 为 354 维）；LogisticRegression 的 C 从
inner OOF 固定网格 0.001/0.01/0.1 选择。三组 locked 聚合结果：legacy raw
92/153/295、F1 0.411；coverage raw 98/153/306、F1 0.427；coverage raw + budget
77/153/207、F1 0.428。均未达到相对 matching probability 配置 +0.02 的采纳门槛。

raw coverage 的短餐 recall 17/39、非惯用手 53/90，说明原始统计能找回部分弱餐，
但同时放大跨受试者绝对尺度差异和 FP；加 K 后短餐又降至 10/39。该 349/354 维
表示不作为默认，但聚合器保留供后续特征筛选。下一阶段采用事件序列约束（同一
受试者局部候选簇只保留最高分、最小餐间隔由 inner OOF 选择），目标是在不裁掉
其他时段真餐的前提下降低 fold2/3 重复误报。

### 5.8 ACC+GYRO 微窗口并集消融（2026-09-10，拒绝设为默认）

预注册配置在不改动 outer 标签、阈值网格或 macro baseline 分母的前提下，先以
15s/7.5s 的 47 维 ACC+GYRO 微窗口模型生成候选，再与原 240s macro 候选取并集，
由 56 维事件复核器评分。所有选择仅来自 outer-train 的 subject-disjoint OOF；下表
的 `短餐 final recall` 是 `<10min` outer 事件的最终检出率，而非候选率。
由于已对同一 outer-CV 进行重复开发比较，本次 0.478632 仅属于开发证据，不是最终
untouched 泛化分数；最终报告仍需新的未触碰测试。

| fold | config hash | micro 阈值 | TP/eligible/pred | F1 | union 候选 recall | micro-only recall | 短餐 final recall | union 候选数 | 运行 s* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `b66281c96383ccee` | 0.10 | 15/23/35 | 0.517 | 0.826 | 0.783 | 0.143 | 713 | 26.8 |
| 1 | `670a483d50e59c48` | 0.20 | 28/31/88 | 0.471 | 1.000 | 1.000 | 1.000 | 726 | 25.9 |
| 2 | `16834e85ac78c41f` | 0.10 | 17/27/60 | 0.391 | 0.926 | 0.741 | 0.500 | 644 | 26.1 |
| 3 | `b0fddb538990ad28` | 0.10 | 25/32/73 | 0.476 | 0.969 | 0.625 | 0.615 | 710 | 25.7 |
| 4 | `42082593914869d9` | 0.10 | 27/40/59 | 0.545 | 0.975 | 0.900 | 0.429 | 620 | 24.0 |
| **聚合** | `fabfba0da8d8dc91` | — | **112/153/315** | **0.479** | **0.948** | **0.817** | **20/39 = 0.513** | **3,413** | **128.5** |

\* 运行时间是各折已训练阶段之和，排除了 20 个可复用 micro cache 的一次性提取；
五折命令实测墙钟约 41.3s。缓存提取累计为 2,316.1s（3,552,881 个 47 维窗口）。

该消融通过了分数、短餐、PPV 和运行时间门槛：F1 0.479 ≥ 0.436、短餐 0.513 ≥
0.510、PPV 0.356 ≥ 0.294、五折墙钟 <600s；相对 locked macro baseline F1=0.416
提高 0.063。但它**未通过候选体积门槛**：3,413 个 union 候选远高于 `4 × 153 =
612`，故不得将 `micro_enabled` 设为默认。它距最终 F1 0.65 仍差 **0.171**，仅保留
可复用的微窗口表示与紧凑 JSON 证据，下一步应在训练内解决候选去重/精度，而不是按
outer 结果调阈值。FD-I/FD-II 外部数据在竞赛规则下可用，但迁移实验仍推迟到单独的
门控计划；使用其 CC BY-NC-ND 4.0 数据前仍须遵守许可条款。

### 5.9 Event-stack artifact 与发布契约（2026-09-11）

候选控制/stacking 的 deployment bundle 不信任自报指标。合法晋级必须在一个原子 run 内保留
canonical aggregate summary、五个 outer-fold evidence bundle、一个 deployment bundle 及
`promotion_attestation.json`；attestation 绑定实验 key、严格五折、门槛/F1 和每个 manifest 的
SHA-256。打包器只接受这一完整结构，且只原子替换仓库 `dist/event_stack`。

晋级脚本还会在写入前重新验证证据，而不是信任 summary 的自报字段：五份 `run_config`
必须重现锁定的 experiment key；每折以当前 `FilesystemDataSource` 输入指纹和
`(macro, verifier, micro)=(63,56,47)` 重算 cache key；outer/inner 指标、候选/短餐
召回、切片、计时和 fold 清单均从五份 evidence 依 runner 的同一聚合公式复算。full-target
训练只拼接经过校验的 outer-validation 分区：macro 原始特征为 62 列并在 time prior 后为
63 列，micro 为 47 列。与训练 pipeline 一致，特征可含由 `SimpleImputer(median)` 处理的
`NaN`，但拒绝 `+/-Inf` 和整列缺失（默认插补器会丢列）；标签必须是有限三态
`-1/0/1`，其中 `-1` 在拟合前剔除。候选 verifier 特征遵循同一插补/列宽契约；最终写入
bundle 的 63/47/56 schema 来自实际拟合 estimator 的 `n_features_in_`。任一不一致都会在
创建 `models/` 或 `dist/` 前拒绝晋级。

deployment bundle 的输入状态覆盖定义 full-target 并集的五个 macro validation cache、五个
micro validation cache、五份 fold manifest、index/meals manifest，以及每份 validation cache
实际引用或相应 split manifest 列出的 validation session cache。必需的 cache/manifest/index/meals
缺失会拒绝晋级；会话 cache 缺失则以稳定的 `{path, missing: true}` 标记写入 provenance，符合
`_eligible_truths` 对不可用会话的合法跳过规则。已存在文件记录 canonical absolute path、size、
mtime_ns 和 SHA-256，故内容变更或缺失会话随后出现都会改变部署 manifest。所有
train、meal_train 和 no_meal_train cache 同样不属于 deployment 指纹。

部署特征输入必须同时携带稳定 `subject_id` 与会话 `sid`：前者用于冻结的 candidate admission
预算和 event budget，后者只用于同会话 NMS 与输出事件几何；同一 payload 中的 `sid` 必须全局唯一，
即使它们属于同一受试者也会被拒绝。CPU 是当前唯一可发布后端；`auto` 因此解析为 CPU，强制 CUDA
失败。当前 CUDA adapter registry 为空，打包/加载阶段会拒绝 `cuda_adapter.py` 及任何 CUDA/component
声明。未来只能通过代码内显式、审计过的注册协议添加设备实现；CPU/CUDA 输出相近本身不能证明实际
在 CUDA 上执行。

### 5.10 历史严格最佳：候选控制 + Logistic/LightGBM blend（2026-09-14）

实验 key 为 `035644cf0889a5dd`。这是一次严格五折、四路 inner subject-disjoint OOF 选择后的
开发证据；相对上一版 `0.5432098765`，唯一注册改动是把 inner OOF admission 最低候选
召回约束从 0.88 降至 0.80，使门控可以拒绝更多低质量候选。它仍不是 untouched 测试集
泛化估计。raw union 为 3,413，现版本 admission 后为 299，全目标域 deployment 使用五个
outer-validation 分区的合法并集（不是把 outer 标签回灌到选择过程）。

| fold | config hash | micro 阈值 | blend | admission (IoU/阈值/cap) | raw/admitted | TP/eligible/pred | F1 | candidate recall | final short recall |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | `7e16f906c79cb8a6` | 0.10 | 0.75 | 0.3/0.35/8 | 713/60 | 14/23/26 | 0.5714285714 | 16/23 | 1/7 |
| 1 | `1f690edb16c70372` | 0.20 | 0.50 | 0.3/0.2/8 | 726/56 | 27/31/56 | 0.6206896552 | 27/31 | 4/4 |
| 2 | `da02d2581f5b4f95` | 0.10 | 0.75 | 0.3/0.5/8 | 644/55 | 18/27/52 | 0.4556962025 | 18/27 | 4/8 |
| 3 | `e165f585d47eba6e` | 0.10 | 0.25 | 0.3/0.2/8 | 710/64 | 24/32/57 | 0.5393258427 | 27/32 | 8/13 |
| 4 | `85cd9c33d4f0e9b9` | 0.10 | 0.50 | 0.3/0.2/8 | 620/64 | 26/40/46 | 0.6046511628 | 30/40 | 2/7 |
| **聚合** | — | — | — | — | **3413/299** | **109/153/237** | **0.5589743590** | **118/153=0.7712418301** | **19/39=0.4871794872** |

聚合 PPV 为 `0.4599156118`（FP=128），recall 为 `0.7124183007`；micro candidates 为
2,773。相比上一版 `0.5432098765`（TP=110、pred=252、FP=142），F1 提升
`+0.0157644824`，以 1 个 TP 换取 14 个 FP 的减少。它未通过推荐默认门：短餐最终 recall
仅 `19/39=0.4871794872`（推荐门 `≥0.65`）；candidate short recall 为 `0.5641025641`，
且项目目标 `F1≥0.65` 尚未达到。因此状态为“已固化的严格改进、
暂不推荐为默认”。

模型路径为 `models/event_stack/035644cf0889a5dd/`（5 个 outer-fold、deployment、
`promotion_summary.json`、`promotion_attestation.json`），发布包为 `dist/event_stack/`。
attestation 绑定 canonical summary、严格五折和每个 manifest 的 SHA-256；deployment manifest
另记录 full-target provenance。输入仍是 63/47/56 维的预计算特征 JSON，不是原始会话；raw-session
adapter 尚未完成。当前 CPU-only：`auto` 解析 CPU，强制 `gpu/cuda` 在无 CUDA adapter 时明确失败。

## 6. 结果分析与评价

### 6.1 两方案瓶颈分解对比

| 环节 | proposal 体系 | 滑窗管线（干净协议） |
|---|---|---|
| 候选覆盖 | 47-58% 餐（活动连通域几何限制） | ~82%（几何全覆盖，28/153 漏为窗证据不足或缺口邻域） |
| 候选层 recall | 0.25 | ~0.82（153 餐中 125 有 IoU≥0.25 候选） |
| 窗级正样本 | 17/折 | 1112/折 |
| FP 控制 | 单级排序硬扛 | 密度候选 + 复核器（PPV 0.35-0.70；FP 窗口证据与真餐同强——信息极限） |
| 5 折 F1 均值 | ~0.27 | **~0.51（TCN）/ ~0.45-0.47（CPU 部署）** |

### 6.2 关键实验与消融（时间轴修复后、严格口径）

| 实验 | 结果 | 结论 |
|---|---|---|
| 时间轴修复（peer review 审计） | 旧 0.319/0.333 作废 | WINDOW_MS=525000 bug（事件右扩 8.75min 伪膨胀）+ 行号轴漂移 2.9-5.7min/会话；修复后真实基线 ~0.2 |
| wbag 跨折泄漏审计（peer review 2） | v5.1 0.617→干净 0.505 | 跨折 bag 模型见过 val 受试者会话——泄漏约 0.08-0.13；每折 -0.09~-0.24（fold4 0.750→0.590 干净 TCN） |
| 负样本子采样 NEG_RATIO=100 | 恢复训练（valAUC 0.80→0.83+） | 真实时间轴候选 1:2183 失衡 → 全负塌缩 |
| 正样本增强 AUG×4 | 0.202→0.249（对照系统） | 时间抖动 ±5s + 噪声；AUG8 过强 |
| 滑窗管线移植（组内方案借鉴） | 0.27→0.41 | 覆盖 + 两级架构为结构性增益 |
| 事件框越阈窗收缩 | fold0 候选匹配 17/25 可达 | 密度 run 膨胀 ±600s 缓冲过长，收缩贴合餐 |
| TCN 深度分（滑窗上） | 干净协议窗级 AUC +0.01-0.025（fold2 0.785 仍最弱） | 全折增益；fold4 NO_TCN 结论为泄漏伪影 |
| 窗阈值×密度网格（干净协议） | 6 组全局参数无一组超基线 | 弱餐窗概率不越阈，密度参数不是瓶颈 |
| 密度覆盖率语义（BME_DENS_COVFIX） | 候选漏 28→16，净 F1 -0.01 | 缺口邻域餐真实修复但复核 FP 增——缺"缺口邻域"复核适配 |
| 复核器 LR→HGB / +上下文计数 | 0.483 / 0.486（<0.505） | 复核瓶颈在窗口证据本身非模型容量 |
| 折内多样 bag（负样本重采样） | no-TCN +0.024；TCN -0.015 | sklearn HGB 确定性 → 种子无效；多样 bag 仅 CPU 口径有价值 |
| 标签偏移（修复后复测） | 中位 10-13.5min 不变 | 真实数据性质；大膨胀/支持窗跨度框可部分桥接；"F1 上限"结论作废（recall≠F1 上限 + 检测框形态可补偿） |

### 6.3 与组内平行方案/文献对比

| | 组内平行方案（0.5143@40 测试标注） | 本项目滑窗管线（干净协议 CV） |
|---|---|---|
| 架构 | 全覆盖滑窗 + 密度 + 33 特征复核（纯 ACC62） | 同架构 + TCN 深度分融合（62+2） |
| 评估 | 6 名独立受试者测试集 26 餐/208h；嵌套 OOF 0.5191 | 5 折受试者级零信息 CV（153 eligible 餐） |
| 结果 | F1 0.5385（测试）/ 0.4907（OOF）/ 0.5191（嵌套） | **均值 0.505 / 聚合 0.512（TCN）**；CPU 口径 0.449-0.473 |

两系统在同一协议（受试者互斥零信息）下数值同量级（0.49-0.52），交叉验证了两级
架构 + 滑窗为有效结构；本项目 TCN 特征在干净协议下提供 +0.05~0.06 增益（fold4
+0.075）。剩余差距主要来自短餐候选层漏检与餐时误报（见 5.3 误差结构）。

## 7. 总结与应用展望

**总结**：当前严格最佳为 event-stack 候选控制/stacking 版本——240s macro 与 15s
ACC+GYRO micro 全覆盖候选，经同会话 NMS、subject admission，再由 LogisticRegression
与 LightGBM blend 复核并执行冻结 event policy；严格五折聚合为 **F1 0.5589743590
（109/153/237，PPV 0.4599156118，recall 0.7124183007）**。严格修复了时间轴类评估伪影
与 wbag 跨折受试者泄漏（0.617 作废），并以 eligible 质量审计分母和 nested OOF 保证
选择隔离。该结果已固化并优于上一版 0.5432098765，但最终短餐 recall 0.487179（candidate
short recall 0.564103）未达推荐门
0.65，F1 也未达项目目标 0.65，故仍标记为开发证据而非最终泛化承诺。

**展望**：
1. 复核层结构改进：缺口邻域强证据餐的接受（密度覆盖率语义已修复，需复核适配，
   见 5.3）；短餐候选召回（窗证据弱是主因）；
2. 窗口层（fold2 AUC 0.785 最弱）：新特征源（GYRO/PPG）或长上下文窗表示；
3. 餐时误报（PPV 底噪）：部分为未记录进食，需事件级上下文/行为模式判别；
4. raw-session adapter：目前 dist 只接受预计算 63/47/56 维特征 JSON；补齐适配器后再
   重新做无训练数据的 CPU smoke/parity 验证。FD-I/FD-II 迁移仅在该目标域基线稳定后
   启动，保留随机初始化和 external_weight=0 对照并遵守数据许可证。

---

## 复现

```bash
conda activate bme
# 1. 会话缓存（真实时间戳 raw，单次）
#    （cache/sessions/ 已存在可跳过；重建见 scripts/compress_sessions2.py）
# 2. 滑窗特征（多进程并行；train/meal_train/no_meal_train/val 四种表）
python scripts/slide_features.py --fold {0..4} --mode train
python scripts/slide_features.py --fold {0..4} --mode meal_train
python scripts/slide_features.py --fold {0..4} --mode no_meal_train
python scripts/slide_features.py --fold {0..4} --mode val
# 3. 正式 locked nested event-stack 评估（当前严格最佳；CPU-only，重复运行复用缓存）
python scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 5 \
    --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80
# 3a. 晋级与发布（当前注册门要求 aggregate F1 严格高于 0.5432098765）
python scripts/promote_event_stack.py --summary outputs/crossfit/summary_035644cf0889a5dd.json
python scripts/package_event_stack.py --bundle models/event_stack/035644cf0889a5dd/deployment \
    --destination dist/event_stack
# 3b. ACC+GYRO 微窗口候选并集消融（当前因候选体积门槛未采纳为默认）
D:/Anaconda3/envs/bme/python.exe scripts/build_micro_features.py --fold all --split all --workers 8
D:/Anaconda3/envs/bme/python.exe scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0 --micro-enabled
# coverage 召回消融（当前未采纳为默认）
python scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 0 --coverage-fix
# 旧逐折 val 最优阈值脚本仅供诊断
python scripts/slide_verifier.py --fold {0..4}
# 4. （对照系统）FD 预训练微调 5 折
python scripts/train_ranker.py --fold {0..4} --no-ppg --init-from checkpoints/fd_pretrained_s1.pt \
    #  环境：BME_BATCH=256 BME_NEG_RATIO=100 BME_AUG_POS=4（当前最优对照配置）
python scripts/rank_events_v2.py --fold {0..4} --prior-grid 15m
python scripts/official_iou_eval.py --all
```

## 项目结构

```
src/            # 核心库（config/data/eval/infer/models/pipeline/event_stack）
scripts/        # event-stack、滑窗管线、对照系统、FD 预训练与官方评估
docs/           # 正式架构、数据处理、组内/外部数据审计与发布说明
checkpoints/    # 当前仍使用的模型训练检查点
cache/          # 当前管线缓存（sessions/slide/micro15/splits/crossfit）
outputs/        # 当前 locked crossfit 汇总与 5 个逐折证据
dist/           # event_stack 发布包（当前预计算特征输入）+ 遗留对照推理包
FDdatasets/     # FD-I/FD-II（KU Leuven 外部数据）
ReferenceDocs/  # 文献综述（报告引用素材）
Archieves/  Data/   # 历史与原始数据（保留）
```

设计文档：docs/三阶段重构设计.md（当前架构、严格结果、审计时间线与迁移边界）。

## 开发过程与决策时间线、协议审计与迁移路线

- **时间轴审计**：修复 `WINDOW_MS=525000` 的毫秒/行数混用，以及按行号而非包级时间戳
  定位窗口的问题；旧 0.319/0.333 结果作废。评估统一为 IoU≥0.25、eligible 153-event
  分母和真实时间戳。
- **第二轮 peer review**：确认 v5.1 跨折 wbag 泄漏（0.617/0.632 作废），确认 LOSO
  verifier 曾见留出受试者并已重训修复；严格 LOSO 为 0.416（67/153）。TCN 在干净协议
  全折有增益，但 CPU 发布仍采用 no-TCN。组内 0.652/0.499 等数字因阈值、正类提纯或
  纯负会话口径未嵌套，不能直接横比。
- **架构决策**：活动提案覆盖不足，切换为 240s/15s 全覆盖 macro；组内审计确认
  15s ACC+GYRO 与旋转稳健特征值得迁移，于是加入 micro 候选。微窗并集曾达到
  0.478632（112/153/315），但 raw union 3,413 超过 612 候选体积门，保留为历史证据。
- **当前晋级**：candidate-control spec 固定同会话 NMS、subject admission、LR/LGBM
  blend 和事件策略只在 outer-train 的四路 subject-disjoint OOF 选择；实验 key
  `035644cf0889a5dd` 得到 F1 0.5589743590（109/153/237，FP 128）。随后预注册的
  Context-v1 将确定性 session-local 上下文追加到 116 维 verifier，run
  `160afaf81debf1ee` 达到 F1 0.6514285714（114/153/197，FP 83），已生成新的 5+1 bundle、
  provenance、attestation 与 `dist/`。该数据仍是 CV development evidence，不宣称独立测试集最优。
- **迁移学习路线**：先补齐 raw-session adapter 和短餐/餐时 hard-negative，再在同一
  15s 表示上评估 FD-I/FD-II；必须保留随机初始化、`external_weight=0` 控制，未经许可
  确认不纳入 WIMID/CAD，遵守 FD 的 CC BY-NC-ND 4.0 条款。
