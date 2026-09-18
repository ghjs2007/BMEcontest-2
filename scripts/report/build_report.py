# -*- coding: utf-8 -*-
"""生成《基于智能手表传感器的进食检测算法》参赛作品报告（按竞赛模版格式）。

用法::

    python scripts/report/build_report.py [--out Archieves/作品报告_xxx.docx]

依赖 python-docx、matplotlib、Pillow（中文字体用系统黑体/宋体）。可视化截图由
``node dist/visual/app/tools/capture-report-shots.mjs`` 生成到本目录 ``assets/``；
该目录为可重建的中间产物（.gitignore），报告与截图全过程只使用合成演示数据。
PDF 版可在 Word/WPS 中打开生成的 docx 导出（可选步骤）。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import docx
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

RE = Path(__file__).resolve().parents[2]
TMP = Path(__file__).resolve().parent / "assets"
TMP.mkdir(exist_ok=True)

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--out", type=Path,
                     default=RE / "Archieves" / "作品报告_基于智能手表传感器的进食检测算法_v2.docx",
                     help="输出 docx 路径（默认 Archieves/ 下的 v2）")
OUT = _parser.parse_args().out

HEI = "黑体"
SONG = "宋体"
TNR = "Times New Roman"


def set_font(run, *, east=SONG, ascii_f=TNR, size=12, bold=False, color=None):
    run.font.name = ascii_f
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), east)


def heading(doc, text, level):
    size = {1: 16, 2: 15, 3: 14}[level]   # 三号/小三/四号
    p = doc.add_paragraph()
    p.style = doc.styles[f"Heading {level}"]
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level == 1 else 6)
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = p.add_run(text)
    set_font(run, east=HEI, ascii_f=TNR, size=size, bold=True, color=RGBColor(0, 0, 0))
    return p


def body(doc, text, *, indent=True, size=12):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE   # 1.5 倍行距（用户定稿格式）
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if indent:
        pf.first_line_indent = Pt(size * 2)  # 首行缩进 2 字符
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_font(run, size=size)
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(3); pf.space_after = Pt(6)
    run = p.add_run(text)
    set_font(run, east=HEI, size=10.5, bold=False)
    return p


def table(doc, header, rows, *, font=10.5, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, text in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        set_font(run, east=HEI, size=font, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows, start=1):
        for j, text in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            set_font(run, size=font)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
    if widths:
        for j, w in enumerate(widths):
            for r in t.rows:
                r.cells[j].width = Cm(w)
    return t


# ---------------------------------------------------------------- 数据与图
FOLDS = [("0", "afcf9609a2f6925a", 16, 23, 35, 0.5517241379),
         ("1", "8b3e27bd2796d038", 30, 31, 46, 0.7792207792),
         ("2", "b463226db1e6073a", 15, 27, 33, 0.5000000000),
         ("3", "3f26fcb2172cd882", 25, 32, 42, 0.6756756757),
         ("4", "0199041f39a09f5d", 28, 40, 41, 0.6913580247)]
SLICES = [("惯用手场景", "dominant", 52, 63, 0.8254, 0.9123, 0.8667),
          ("非惯用手场景", "nondominant", 62, 90, 0.6889, 0.8857, 0.7750),
          ("短餐（<10 min）", "duration_lt10", 20, 39, 0.5128, 0.7692, 0.6154),
          ("中等餐（10–20 min）", "duration_10_20", 62, 74, 0.8378, 0.9254, 0.8794),
          ("长餐（≥20 min）", "duration_ge20", 32, 40, 0.8000, 0.9412, 0.8649)]

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig1 = TMP / "fig1_folds_slices.jpg"
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), dpi=150)
ax = axes[0]
ax.bar([f"折{f}" for f, *_ in FOLDS], [x[5] for x in FOLDS], color="#4C72B0")
ax.axhline(0.6514, color="#C44E52", ls="--", lw=1)
ax.text(4.35, 0.66, "聚合 0.651", color="#C44E52", ha="right", fontsize=9)
ax.set_ylim(0, 0.9); ax.set_ylabel("F1"); ax.set_title("五折 outer 评估 F1")
for i, (f, *_rest) in enumerate(FOLDS):
    ax.text(i, _rest[4] + 0.02, f"{_rest[4]:.3f}", ha="center", fontsize=8)
ax = axes[1]
ax.bar([s[0] for s in SLICES], [s[6] for s in SLICES], color="#55A868")
ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.set_title("场景与时长切片 F1")
ax.tick_params(axis="x", labelsize=7.5)
for i, s in enumerate(SLICES):
    ax.text(i, s[6] + 0.02, f"{s[6]:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(fig1, format="jpg", dpi=150); plt.close()

fig2 = TMP / "fig2_history.jpg"
HIST = [("W1 轻量基线\n(LGBM 窗特征)", 0.106), ("对照系统\n(提案+深度排序)", 0.27),
        ("滑窗两级架构", 0.41), ("滑窗+TCN bag\n(泄漏协议,后作废)", 0.617),
        ("协议审计\n(严格零信息)", 0.505), ("event-stack\n候选控制+融合", 0.559),
        ("Context-v1\n(当前发布)", 0.651)]
fig, ax = plt.subplots(figsize=(9.6, 3.3), dpi=150)
xs = range(len(HIST))
ax.plot(list(xs), [h[1] for h in HIST], marker="o", color="#4C72B0", lw=1.6)
ax.set_xticks(list(xs)); ax.set_xticklabels([h[0] for h in HIST], fontsize=8)
ax.set_ylabel("开发口径 F1"); ax.set_ylim(0, 0.75); ax.grid(alpha=0.3)
for i, h in enumerate(HIST):
    ax.text(i, h[1] + 0.02, f"{h[1]:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(fig2, format="jpg", dpi=150); plt.close()

# 可视化界面截图（均为内置合成演示数据，无真实受试者数据）
from PIL import Image

_raw_monitor = TMP / "ui_monitor.png"
_raw_events = TMP / "ui_events.png"
for path in (_raw_monitor, _raw_events):
    if not path.exists():
        raise SystemExit(
            f"missing UI screenshot: {path}\n"
            "run: node dist/visual/app/tools/capture-report-shots.mjs")
fig3 = TMP / "fig_ui_monitor.png"
fig4 = TMP / "fig_ui_events.png"
Image.open(_raw_monitor).convert("RGB").save(fig3)
_events = Image.open(_raw_events).convert("RGB")
# 事件页下半为空白，只保留页头与事件表（约占原图上 43%）
_events.crop((0, 0, _events.width, int(_events.height * 0.43))).save(fig4)

# ---------------------------------------------------------------- 文档
doc = Document()
for s in doc.sections:
    s.page_width = Cm(21); s.page_height = Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.0)
    s.top_margin = s.bottom_margin = Cm(1.0)

style = doc.styles["Normal"]
style.font.name = TNR
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)


def page_break():
    doc.add_paragraph().add_run().add_break(docx.enum.text.WD_BREAK.PAGE)


# ---- 封面
for text, size, bold in ((("第十一届全国大学生生物医学工程创新设计竞赛"), 16, True),
                         (("预赛作品报告"), 16, True)):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text); set_font(run, east=HEI, size=size, bold=bold)
for _ in range(4):
    doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("基于智能手表传感器的进食检测算法"); set_font(run, east=HEI, size=22, bold=True)
for _ in range(5):
    doc.add_paragraph()
for text in ("作品ID号：5630", "参赛学生类型：本科生",
             "参加赛道：智能穿戴与运动健康赛道", "组别：命题项目组（赛题二）", "2026 年 9 月"):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text); set_font(run, size=14)
doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("请勿在作品中出现参赛者及指导教师相关学校及个人信息，否则不予评审")
set_font(run, size=10.5)
page_break()

# ---- 内容完整性自查表
heading(doc, "内容完整性自查表", 1)
table(doc,
      ["完整性类别", "任务或技术指标名称", "完成效果", "呈现方式"],
      [["赛题任务", "惯用手场景进食事件检测", "F1 0.867（52/63）", "§3.2 表3、图2"],
       ["赛题任务", "非惯用手场景进食事件检测", "F1 0.775（62/90）", "§3.2 表3、图2"],
       ["主要指标", "官方口径 F1（IoU≥0.25，全局）", "开发 CV 聚合 0.6514", "§3.2 表2"],
       ["次要指标", "正确检出事件起止时间 MAE", "由委员会测试接口评估", "§3.1 评估协议"],
       ["交付物", "可执行文件（原始数据端到端推理）", "dist/submission（CPU，一键运行）", "§4.1 作品展示"],
       ["交付物", "训练代码与复现说明", "src/ + scripts/ + tests/（全链复现）", "§4.1 作品展示"],
       ["交付物", "总结报告", "本报告", "—"],
       ["其他", "泄漏安全的评估协议与发布证据链", "严格嵌套 CV + 四路 parity + attestation", "§2.3、§3.1"]],
      font=10.5)
page_break()

# ---- 摘要
heading(doc, "摘    要", 1)
abstract = (
    "进食监测是穿戴健康分析的重要场景。本文针对智能手表多源传感器数据，设计了面向"
    "惯用手与非惯用手场景的进食事件检测算法。算法采用两级架构：第一级以 240 秒滑窗与"
    "15 秒微窗的候选并集覆盖全部时段，经非极大值抑制与受试者级准入控制候选规模；"
    "第二级以含 60 列确定性上下文（Context-v1）的 116 维事件特征，经 Logistic 回归与"
    "受限 LightGBM 融合复核，由冻结事件策略输出进食事件。全部阈值与策略仅在受试者"
    "互斥的嵌套交叉验证内选择。严格 subject-disjoint 五折开发评估取得聚合 F1 0.6514"
    "（灵敏度 0.745、阳性预测率 0.579），惯用手/非惯用手场景 F1 为 0.867/0.775，短餐"
    "（<10 分钟）F1 0.615。工程上交付纯 CPU 端到端推理包与竞赛提交包，以四路边界"
    "一致性与发布存证保证零偏斜、可追溯。")
p = doc.add_paragraph()
pf = p.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
pf.first_line_indent = Pt(24)
run = p.add_run(abstract); set_font(run, size=12)
body(doc, "关键词：进食检测，可穿戴传感器，IMU，事件检测，嵌套交叉验证", indent=False)
page_break()

# ---- 目录（静态；Word 中可按需更新页码）
heading(doc, "目    录", 1)
toc_lines = ["摘要", "1 作品概述", "　1.1 背景及意义", "　1.2 研究基础与相关工作",
             "　1.3 需求分析与技术难点", "　1.4 研究目标与已实现指标",
             "2 作品方案设计及实现", "　2.1 技术路线概述", "　2.2 技术方案对比与选型",
             "　2.3 方案设计及实现过程（开发历程）",
             "3 作品测试方案及测试结果", "　3.1 测试方案", "　3.2 技术、功能指标及测试结果",
             "　3.3 技术可行性分析及创新说明", "4 总结", "　4.1 作品展示", "　4.2 展望",
             "参考文献"]
for line in toc_lines:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY; pf.line_spacing = Pt(20)
    run = p.add_run(line)
    set_font(run, size=12, bold=line[0] in "1234摘参" and not line.startswith("　"))
page_break()

# ---- 1 作品概述
heading(doc, "1 作品概述", 1)
heading(doc, "1.1 背景及意义", 2)
body(doc, "智能手表等穿戴设备可全天候、无感地持续采集惯性测量单元（IMU，加速度计与"
          "陀螺仪）及光电容积脉搏波（PPG）等多源数据，为饮食行为的自动量化提供了数据基础。"
          "进食时手部呈现端举、送食、咀嚼等特定运动模式，进食前后还伴随心率与自主神经"
          "调节的变化[4]，这些信号均可被腕部传感器捕捉。自由生活进食监测可用于饮食行为"
          "评估、慢病管理与营养干预，是穿戴健康领域的核心共性技术之一。")
body(doc, "然而该任务存在三个本质困难：其一，进食动作与喝水、打电话、操作手机等日常动作"
          "高度相似，研究显示腕部 IMU 方案在自由生活条件下的准确率仍受此制约[1,2,6]；"
          "其二，惯用手与非惯用手佩戴场景的检测原理不同——惯用手动作特征明显、IMU 主导，"
          "非惯用手动作特征显著减弱、需借助上下文与生理信号；其三，真实数据的进食标注由"
          "用户自报，存在时间偏移与漏标（本数据集标注中位偏移约 10–13 分钟），对训练与"
          "评估协议设计提出更高要求。")
heading(doc, "1.2 研究基础与相关工作", 2)
body(doc, "数据基础：竞赛提供 45 名受试者、1165 个会话、约 100 GB 的真实生活佩戴数据"
          "（IMU 约 105 Hz 原始 ADC，PPG 44 通道占空比采样）。数据存在会话级缺口、低行率"
          "会话与少量无效标注，本文在数据层建立了时间戳恢复、缺口切段与质量审计机制"
          "（见 §3.1）。此外可合法使用 KU Leuven FD-I/FD-II 公开数据集（61 名参与者、"
          "513 小时双腕 ACC+GYRO 与咬食标注[7]）作为迁移学习素材，本工作完成了其预处理"
          "与 Episode 级预训练链路验证。")
body(doc, "方法基础：文献表明，带时序建模的分类器显著优于静态分类器，深度方案中"
          "CNN 与 RNN 组合最优[1,6]；PPG 与进食的关联（餐后 RR 间期缩短、LF/HF 升高等）"
          "有生理学证据[3,4]，多波长 PPG 通道间噪声抵消亦有专利先例[5]。组内平行方案"
          "验证了\"全覆盖滑窗 + 密度候选 + 事件复核\"的两级结构在中等规模数据上的有效性"
          "（嵌套交叉验证 F1 约 0.52）。本文在其骨架上进行了系统性的架构扩展与协议加固。")
heading(doc, "1.3 需求分析与技术难点", 2)
body(doc, "按赛题要求，算法需在两种佩戴场景下检测进食事件，评价采用事件级 IoU≥0.25 的"
          "一对一匹配，以全局 F1（灵敏度与阳性预测率的调和平均）为主指标，正确匹配事件的"
          "起止时间误差（MAE）为次要指标；另需提交可执行文件、训练代码与总结报告。据此"
          "拆解出四项技术需求：（1）候选覆盖——早期基于活动连通域的\"提案\"几何上仅能"
          "覆盖约一半进餐时段，必须改为全覆盖候选以保证召回；（2）假阳性控制——候选层"
          "阳性预测率天然较低（约 0.1），需要事件级复核把精度提升一个量级；（3）跨受试者"
          "泛化——自由生活数据中受试者间风格差异大，评价协议必须严格受试者互斥，避免"
          "乐观偏差；（4）工程合规——可执行文件须在组委会环境独立运行，优先纯 CPU、"
          "低依赖、可追溯。")
heading(doc, "1.4 研究目标与已实现指标", 2)
body(doc, "目标是最大化官方口径全局 F1。当前发布版本（严格 subject-disjoint 嵌套五折"
          "开发评估，可评估事件分母 153）实现聚合 F1 0.6514、灵敏度 0.745、阳性预测率"
          "0.579；本工作同时交付了原始传感器数据端到端推理的独立可执行包、完整训练与"
          "复现代码、以及带哈希存证的发布证据链（§4.1）。")

# ---- 2 作品方案设计及实现
heading(doc, "2 作品方案设计及实现", 1)
heading(doc, "2.1 技术路线概述", 2)
body(doc, "系统采用\"全覆盖候选 + 事件级复核\"的两级检测架构，处理链如下：")
body(doc, "① 原始会话解析：读取 collect_data*.txt（53 列 TSV），按真实包级时间戳恢复时间"
          "轴，切分有效 IMU 段与数据缺口，窗口不跨缺口；② 双尺度特征：macro 通道以 240 秒"
          "窗/15 秒步长提取 62 维 ACC 稳健统计与 1 秒活动包络时域特征，micro 通道以 15 秒"
          "窗/7.5 秒步长提取 47 维重力对齐 ACC+GYRO 特征；③ 候选生成：macro 概率经密度聚合"
          "（600 秒内足够多越阈窗）产生候选，micro 概率经阈值扫描与几何约束产生候选，二者"
          "取并集；④ 候选控制：同会话稳定非极大值抑制 + 按受试者的准入控制（阈值、IoU、"
          "每受试者候选上限均冻结自训练域）；⑤ 事件复核：对候选计算 116 维事件特征——56 维"
          "概率形态与上下文统计，加 60 列确定性 Context-v1（同会话内候选前后 20 分钟的概率"
          "流统计、阈值 run 形态、覆盖率与集中度），由 Logistic 回归与受限 LightGBM 按冻结"
          "权重做概率融合；⑥ 事件解码：冻结事件策略（阈值、几何与每受试者事件预算）输出"
          "Episode 起止时间，官方口径评估（IoU≥0.25 一对一匹配）。")
body(doc, "训练侧采用严格 subject-disjoint 嵌套交叉验证：外层按受试者五折留出；外层训练"
          "域内再做四路内层主体互斥 OOF（macro 窗模型、micro 窗模型、两种复核器），融合"
          "权重、NMS 参数、准入阈值/上限、事件阈值与预算全部只在 OOF 结果上选择；外层"
          "验证折全程不可见。模型与策略冻结后固化为发布 bundle，并附逐文件 SHA-256 与"
          "提升存证（attestation）。")
heading(doc, "2.2 技术方案对比与选型", 2)
body(doc, "开发过程中先后实现并对比了三代方案，选型依据如下表所示：")
table(doc, ["方案", "结构", "优势", "局限与结论"],
      [["对照系统：活动提案 + 深度排序",
        "多阈值活动连通域提案 → MM-Ranker（TCN）深度排序 → 形态学后处理",
        "深度模型窗级判别力强（AUC 0.83–0.90）",
        "提案几何覆盖不足（47–58% 进餐时段可达），单级排序无法同时兼顾召回与精度；开发口径 F1 约 0.27"],
       ["滑窗两级架构（初版）",
        "240s/15s 全覆盖滑窗 + HGB 窗模型 + 密度候选 + 33 维复核",
        "候选覆盖接近 100%，两级结构将候选层低精度转化为事件层可控精度",
        "复核特征信息量不足；在泄漏协议下出现过乐观数字（见 §2.3 阶段五）"],
       ["event-stack（当前发布）",
        "双尺度候选并集 + 候选控制 + 116 维事件复核融合 + 冻结策略",
        "micro 分支补足短餐与非惯用手；Context-v1 提升复核区分度；嵌套选择 + 存证保证可复现",
        "短餐召回仍有提升空间；官方在线接口待组委会规范（§4.2）"]],
      font=10.5)
heading(doc, "2.3 方案设计及实现过程（开发历程）", 2)
body(doc, "本工作经历了从轻量级基线到当前发布版本的完整迭代，可划分为七个阶段"
          "（图 1 汇总了各阶段开发口径 F1；括号内数字为对应评价协议下的结果，其中阶段四"
          "后期数字经泄漏审计后作废，见阶段五）。")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(fig2), width=Cm(13.5))
caption(doc, "图1  开发历程与各阶段开发口径 F1")
body(doc, "阶段一（轻量基线）：建立数据读取、受试者互斥划分（GroupKFold）、事件级 IoU "
          "评估与 LightGBM 窗特征基线（37 维统计特征、5 折评估），得到开发口径 F1 0.106。"
          "该基线确立了评估与数据管线骨架，并暴露了两个核心问题：标签定义须用窗口重叠"
          "比例而非事件 IoU；候选覆盖是主要瓶颈。")
body(doc, "阶段二（级联原型与外源预训练）：最初按腕上设备算力受限的假设，设计了四级"
          "级联轻量化流水线——L1 活动唤醒门控、L2 惯用/非惯用手场景门控、L3a 姿态 CNN "
          "与 L3b PPG 专家网络、L4 融合 MLP + HMM 解码，并配套 ONNX 导出与打包链路，"
          "意在支持端侧部署。实测表明该路线行不通，后来也确认其无必要：L1/L2 门控与硬"
          "路由在验证折上均为负向（-0.03～-0.09），PPG 有效采样率仅约 2 Hz，不足以支撑"
          "HRV 类特征（与文献[3,4]的生理学机理不矛盾，属传感器条件限制），级联整体精度"
          "也远低于两级结构；而赛题只要求可执行文件在组委会环境运行，纯 CPU 端到端推理"
          "单会话约 18 秒，端侧优化的收益归零。据此放弃级联与端侧路线，转向全覆盖滑窗的"
          "两级结构。同期基于 KU Leuven 咬食数据的 Episode 级预训练在五折上一致带来约 "
          "+0.035 增益，验证了外部数据的迁移价值。")
body(doc, "阶段三（对照系统：检测即排序）：转向\"活动提案 + 深度排序\"路线，训练 MM-Ranker"
          "（IMU 卷积 + TCN 残差 + GRU 的窗级排序模型，FD 预训练微调），配合正样本时序增强"
          "（±5 秒抖动 + 噪声）、分层负采样与官方后处理网格搜索。该路线开发口径 F1 止步"
          "约 0.27：提案几何覆盖不足的瓶颈不可通过排序头修复。其间完成了一次关键审计——"
          "发现早期评估存在窗口时长单位错误（5 秒窗的 525 行被误写为 525000 毫秒，事件框"
          "右扩 8.75 分钟造成伪膨胀）与行号时间轴漂移，旧的中高分数（0.333/0.447）全部"
          "作废，评测统一为真实时间戳 + 事件级匹配。")
body(doc, "阶段四（滑窗两级架构）：借鉴组内平行方案，切换为全覆盖滑窗 + 两级复核结构"
          "（0.41），之后依次引入：TCN 深度分融合与复核负样本（0.468）、深度分 bagging"
          "（0.49）、窗模型五折 bagging（0.565）、可评估事件分母质量审计（0.587）、复核"
          "负样本规模调优（0.595）与逐折架构模式选择（0.615–0.617）。此阶段后期在评测"
          "协议上出现系统性缺陷（见阶段五），相关高分据此作废。")
body(doc, "阶段五（评测协议审计与泄漏修复）：对外部同行审计意见逐条复核，确认并修复了两"
          "类协议缺陷：其一，五折 bagging 的平均过程使评估折受试者被其他折模型见过（跨折"
          "泄漏），虚增约 0.08–0.13 F1；其二，留一受试者评估曾复用全数据复核器。修复后"
          "建立\"受试者互斥零信息\"协议：评估折的窗模型与复核器只见过训练折受试者。干净"
          "协议下重估为聚合 F1 0.505，且此前的若干\"增益结论\"（如逐折 TCN 开关）被证伪，"
          "而 TCN 融合、双尺度特征等结构性改进被确认有效。此外定位并量化了短餐漏检"
          "（漏检事件中位时长 7.8 分钟 vs 检出 16.2 分钟）与餐时假阳性（其窗口证据与真餐"
          "无差别，部分疑为未记录进食）两类残余误差的结构，为后续改进锚定方向。")
body(doc, "同轮审计还作废了一条一度写入开发结论的\"F1 上限\"论断。此前基于\"标签时间偏移"
          "数学证明\"（等长平移情形下 IoU≥0.25 要求偏移 δ≤0.6D，中位偏移 10–13 分钟）曾"
          "判定\"用户自报标签的时间精度锁死可达 F1 约 0.33–0.35，当前成绩已逼近上限\"。"
          "复核表明该论断不成立：其一，其测量所依赖的时间轴正是被修复的行号轴与单位错误，"
          "偏移量本身不可信；其二，即使召回受限，F1=2PR/(P+R) 在 PPV=1 时也可达约 0.55，"
          "\"召回上限≠F1 上限\"；其三，预测框形态（支持窗跨度、边界扩展）可将偏移较大的"
          "餐重新纳入 IoU≥0.25，边界膨胀后 F1 单调上升即是反证。该论断随之作废；后续在"
          "严格协议下 F1 由 0.505 提升至 0.6514，从实证上否证了\"上限\"的存在——这也成为"
          "本项目\"协议正确性优先于分数\"的一条方法论。")
body(doc, "阶段六（event-stack 精化）：在修复后的严格协议下重建系统——引入 15 秒/7.5 秒"
          "ACC+GYRO 微窗候选并与 macro 候选取并集（短餐召回显著改善），加入候选准入控制与"
          "逻辑回归/受限 LightGBM 概率融合（0.559），最后引入确定性 Context-v1 事件上下文"
          "特征（60 列，同会话内计算、不读标签、不跨会话），将复核器扩展到 116 维，聚合"
          "F1 提升至 0.6514，达成项目设定的严格目标。该版本经技术门槛与推荐门槛校验后"
          "固化为 `CURRENT_PROMOTED_RELEASE`（发布键 160afaf81debf1ee）。")
body(doc, "阶段七（工程化交付与收口）：将算法收敛为唯一 canonical 实现（src/pipeline/），"
          "建立原始数据端到端推理接口（Predictor）、四路边界一致性测试（历史实现 = "
          "canonical = 推理分发包 = 提交包，逐层数值比对）、发布存证与原子化分发构建；"
          "清理历史实验产物（42 个文件，附六类消费者证明台账），全仓 350 项自动化测试"
          "通过。交付物见 §4.1。")

# ---- 3 测试方案及结果
heading(doc, "3 作品测试方案及测试结果", 1)
heading(doc, "3.1 测试方案", 2)
body(doc, "评价口径：与赛题一致，预测事件与真实事件按起止时间 IoU≥0.25 做一对一贪心匹配，"
          "记为正确检出（TP）；灵敏度 = TP/真实事件数，阳性预测率（PPV）= TP/预测事件数，"
          "F1 = 2·灵敏度·PPV/(灵敏度+PPV)；次要指标为正确匹配事件的起止时间平均绝对误差"
          "（MAE）。分母采用质量审计口径（eligible）：会话数据覆盖不足或时段碎片化的餐次"
          "不计入真实事件数，避免因数据不可达而系统性低估算法（该口径与外源同行的嵌套"
          "评估一致）。")
body(doc, "开发评估协议：采用严格 subject-disjoint 嵌套五折交叉验证。外层按受试者划分"
          "五折；每折的窗模型、微窗模型与两个复核器分别只对未见受试者产生 OOF 输出，融合"
          "权重、NMS、准入参数、事件阈值与预算全部仅由外层训练域内的 OOF 选择，外层验证"
          "折既不参与训练也不参与任何选择。全部报告数字均为该协议下的开发证据；最终泛化"
          "以组委会独立测试集的评分结果为准。")
body(doc, "一致性与可追溯性测试：为保证\"训练—推理零偏斜\"，建立了四路边界一致性回归——"
          "历史实现、canonical 实现、推理分发包与提交包对同一原始会话须给出逐值一致的"
          "中间层输出与相同的事件列表；发布 bundle 逐文件记录 SHA-256，可在任意环境"
          "（含全新克隆）复核，并已规避行尾符等环境差异导致的哈希漂移。工程与协议测试"
          "共 352 项（350 通过、2 项跳过），覆盖数据读取、特征一致性、评估、发布事务与"
          "洁净室运行等。")
heading(doc, "3.2 技术、功能指标及测试结果", 2)
body(doc, "（1）主指标。当前发布版本在严格嵌套五折下的逐折与聚合结果如表 2；五折与场景"
          "切片对比如图 2。")
table(doc, ["折号", "配置哈希", "TP / 可评估 / 预测", "F1"],
      [[f, h, f"{tp}/{n}/{p}", f"{f1:.4f}"] for f, h, tp, n, p, f1 in FOLDS] +
      [["聚合", "—", "114 / 153 / 197", "0.6514"]],
      font=10.5)
caption(doc, "表2  严格 subject-disjoint 嵌套五折结果")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(fig1), width=Cm(14.6))
caption(doc, "图2  五折 F1（左）与场景/时长切片 F1（右）")
body(doc, "（2）场景与时长切片。两类佩戴场景与进食时长的分割结果如表 3：惯用手场景优于"
          "非惯用手场景 0.09 F1，符合两场景信号强度差异的先验；短餐（<10 分钟）F1 为"
          "0.615，是当前主要短板，其中候选覆盖（约 54%）为主要瓶颈。")
table(doc, ["切片", "TP / 事件数", "灵敏度", "PPV", "F1"],
      [[name, f"{tp}/{n}", f"{s:.4f}", f"{p:.4f}", f"{f1:.4f}"] for name, _, tp, n, s, p, f1 in SLICES],
      font=10.5)
caption(doc, "表3  场景与时长切片结果（聚合口径）")
body(doc, "（3）关键消融。开发过程中对主要组件的对照结论如表 4，全部在受试者互斥协议下"
          "复核；其中\"泄漏协议\"一行展示了协议审计的重要性。")
table(doc, ["组件 / 实验", "结果", "结论"],
      [["双尺度候选（macro ∪ micro）", "短餐候选召回 0.54，最终 F1 +0.06", "micro 分支补足短餐与非惯用手，采纳"],
       ["候选准入控制（NMS + subject cap）", "候选 3413→230，PPV +0.10", "候选质量控制对精度贡献显著，采纳"],
       ["Context-v1 事件上下文（60 列）", "聚合 F1 0.595→0.651", "确定性上下文特征提升复核区分度，采纳"],
       ["修复后的严格协议复核", "五折 bag 跨折泄漏虚增 0.08–0.13（0.617→0.505）", "协议缺陷；相关数字全部作废"],
       ["FD 外源 Episode 预训练", "五折一致 +0.035", "外部数据迁移有效，保留为后续方向"],
       ["PPG/HRV 分支", "2 Hz 采样下不可行", "不可行，放弃；符合传感器条件限制"]],
      font=10.5)
caption(doc, "表4  关键组件消融与协议实验")
body(doc, "（4）次要指标与运行性能。正确匹配事件的起止时间误差由组委会测试接口按官方"
          "口径统计；在开发集上，事件框采用支持窗跨度自适应生成，对用户自报标注的中位"
          "时间偏移（10–13 分钟）具有容忍性。可执行文件为纯 CPU 实现，单会话（约一天"
          "数据）端到端推理约 18 秒，无 GPU 依赖；完整嵌套五折开发评估在 32 核开发机上"
          "总耗时约 359 秒（不含一次性特征缓存构建）。")
heading(doc, "3.3 技术可行性分析及创新说明", 2)
body(doc, "可行性：全部指标在真实自由生活数据上取得，协议为受试者互斥的嵌套交叉验证，"
          "阈值与策略均冻结自训练域，且发布了带哈希存证的可复现证据链；推理为纯 CPU、"
          "低时延、单命令运行，满足组委会环境约束。风险与已知局限见 §4.2。")
body(doc, "创新点：① 双尺度全覆盖候选结构——以 240 秒宏观窗与 15 秒微观窗的候选并集"
          "替代活动提案，配合密度聚合与准入控制，把\"候选覆盖不足\"这一结构性瓶颈转化为"
          "可控的事件级复核问题，短餐与非惯用手场景同步受益；② 确定性事件上下文复核"
          "（Context-v1）——同会话内、无标签、无拟合参数的概率流上下文特征，将事件级"
          "复核从\"单点形态判别\"提升为\"轨迹形态判别\"，在严格协议下带来约 +0.06 F1；"
          "③ 泄漏安全的发布证据链——严格嵌套选择、逐文件哈希、提升存证与四路边界一致性"
          "测试，保证训练—推理零偏斜与结果可追溯，这在同类赛事方案中并不多见；④ 原始"
          "会话端到端交付——从 collect_data*.txt 直接推理的独立可执行包与一键再生成流程，"
          "让评委与使用者无需任何中间特征工程即可复现结果。")

# ---- 4 总结
heading(doc, "4 总结", 1)
heading(doc, "4.1 作品展示", 2)
body(doc, "本作品交付三类成果：① 算法与模型——event-stack 发布版本（发布键 "
          "160afaf81debf1ee），含五折证据 bundle、部署模型与提升存证；② 代码——canonical "
          "算法实现（src/pipeline）、训练/评估/发布/构建全链脚本（scripts）、352 项自动化"
          "测试（tests）与复现说明；③ 可执行文件——原始传感器数据端到端推理的独立包"
          "（dist/inference，纯 CPU）与竞赛提交包（dist/submission，含推理接口、完整模型"
          "证据链、复现代码与可视化工作区），两者均可由仓库脚本一键确定性重建，并通过"
          "洁净室独立运行与清单校验。")
body(doc, "④ 交互式可视化应用——可视化工作区（visual/）以自包含单文件网页交付，"
          "双击 visual/index.html 即可在任何机器离线查看内置演示（合成数据）：Monitor 页"
          "以“运动证据—宏观分—微观分—事件”四轨时间轴同步呈现，区间选择联动原始 IMU "
          "三轴曲线与三维腕部刚体回放（图3）；Events 页集中列出全部检出事件与复核分、"
          "支持一键定位复查（图4）；Model 页展示发布元数据与推理路径。应用只消费预测契约"
          "（schema/prediction.schema.json）与运动遥测，不重实现任何算法决策；对真实数据，"
          "双击 start.bat 启动本地 canonical 推理服务（仅绑定 127.0.0.1，浏览器不做特征"
          "计算），依赖缺失时经确认后自动安装。")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(fig3), width=Cm(14.0))
caption(doc, "图3  可视化 Monitor 页：四轨证据时间轴、区间检查、原始 IMU 与三维动作回放（合成演示数据）")
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(fig4), width=Cm(14.0))
caption(doc, "图4  可视化 Events 页：事件列表与复核分，支持定位复查（合成演示数据）")
heading(doc, "4.2 展望", 2)
body(doc, "后续工作沿三条线推进：其一，短餐召回——短餐候选覆盖率（约 54%）与非惯用手"
          "弱信号是当前 F1 的主要损失来源，计划引入更短尺度的微窗表示与候选簇去重；"
          "其二，外部数据迁移——在 FD-I/FD-II 上以同构表示开展手势编码器预训练与难负"
          "样本学习，保留随机初始化与零权重的对照并遵守数据许可；其三，官方在线接口——"
          "组委会规范发布后，在提交包的隔离适配器边界内实现官方格式转换，无需改动算法"
          "与证据链。")
_ref_heading = heading(doc, "参考文献", 1)
_ref_heading.paragraph_format.page_break_before = True  # 段前分页：避免空白页
refs = [
    "[1] Stankoski S, Jordan M, Gjoreski H, Luštrek M. Smartwatch-based eating detection: data selection for machine learning from imbalanced data with imperfect labels[J]. Sensors, 2021, 21(5): 1902.",
    "[2] Heydarian H, Rosenthal P, et al. Deep learning for intake gesture detection from wrist-worn inertial sensors: the effects of data preprocessing, sensor modalities, and sensor positions[J]. IEEE Access, 2020, 8: 164936–164949.",
    "[3] Verrier J, Nazaret A, et al. NPLM: a large-scale study of PPG for meal-related prediction[EB/OL]. arXiv:2511.19260, 2025.",
    "[4] Niizeki K, Saitoh T. Cardiovascular autonomic responses during eating and chewing[J]. Physiology & Behavior, 2016, 159: 1–13.",
    "[5] Huawei Technologies. Multi-wavelength heart rate measurement with noise cancellation: EP4186416[P]. European Patent Office.",
    "[6] Assessing eating behaviour using upper limb mounted motion sensors: a systematic review[J/OL]. Nutrients. https://pmc.ncbi.nlm.nih.gov/articles/PMC6566929/.",
    "[7] KU Leuven. FD-I/FD-II food intake datasets[DB/OL]. DOI: 10.48804/CN8VBB.",
    "[8] Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree[C]. NeurIPS, 2017.",
    "[9] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python[J]. JMLR, 2011, 12: 2825–2830.",
]
for r in refs:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY; pf.line_spacing = Pt(18)
    run = p.add_run(r); set_font(run, size=10.5)

doc.save(str(OUT))
print("saved:", OUT)
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
