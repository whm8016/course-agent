"""Journal-style architecture figure for 实验技术与管理.

Outputs (does not touch docs/assets/course_agent_architecture.*):
  docs/figures/fig1_system_architecture_journal.pdf
  docs/figures/fig1_system_architecture_journal.png
  docs/figures/fig1_system_architecture_journal.pptx
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "fig1_system_architecture_journal"

# Printed size: full-column 160 mm x 102 mm
W_MM, H_MM = 160.0, 102.0
NAVY = "#2F4A6D"
EDGE = "#2F4A6D"
FILL_RAIL = "#E8EEF4"
FILL_LAYER = "#F4F7FA"
FILL_BOX = "#FFFFFF"
FILL_HEAD = "#2F4A6D"
TEXT = "#1A1A1A"
TEXT_W = "#FFFFFF"

SIMSUN = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
SIMHEI = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")


def box(ax, x, y, w, h, facecolor, edgecolor=EDGE, lw=0.5, z=2):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            joinstyle="miter",
            zorder=z,
        )
    )


def label(ax, x, y, text, *, size=8, bold=False, color=TEXT, ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        fontproperties=SIMHEI if bold else SIMSUN,
        fontsize=size,
        color=color,
        ha=ha,
        va=va,
        zorder=6,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.5,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def draw_matplotlib() -> None:
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), dpi=300, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_MM)
    ax.set_ylim(H_MM, 0)
    ax.set_axis_off()
    ax.set_facecolor("white")

    # Outer frame
    ox, oy, ow, oh = 4.0, 4.0, 152.0, 94.0
    box(ax, ox, oy, ow, oh, "white", lw=0.8, z=1)

    rail_w = 13.0
    cx = ox + rail_w
    cw = ow - rail_w

    layers = [
        (oy, 16.0, "交\n互\n层"),
        (20.0, 25.0, "调\n度\n层"),
        (45.0, 38.0, "子\n系\n统\n层"),
        (83.0, 15.0, "数\n据\n层"),
    ]
    for y, h, name in layers:
        box(ax, ox, y, rail_w, h, FILL_RAIL, z=2)
        box(ax, cx, y, cw, h, FILL_LAYER, z=2)
        label(ax, ox + rail_w / 2, y + h / 2, name, size=8, bold=True)

    # --- 交互层 ---
    t_box = (22.0, 7.0, 62.0, 10.0)
    s_box = (90.0, 7.0, 62.0, 10.0)
    box(ax, *t_box, FILL_BOX)
    box(ax, *s_box, FILL_BOX)
    label(ax, 53.0, 12.0, "教师", size=8, bold=True)
    label(ax, 121.0, 12.0, "学生", size=8, bold=True)

    # --- 调度层 ---
    kx, ky, kw, kh = 22.0, 23.0, 130.0, 20.0
    box(ax, kx, ky, kw, kh, FILL_BOX)
    label(ax, 87.0, 27.2, "统一调度内核", size=8, bold=True)
    label(ax, 87.0, 31.4, "工具调用 · 推理—行动循环", size=7.5)
    cell_w, cell_h, gap = 29.5, 7.2, 2.0
    cell_y = 34.2
    cell_x0 = 24.5
    for i, name in enumerate(("问答", "解题", "出题", "研究")):
        x = cell_x0 + i * (cell_w + gap)
        box(ax, x, cell_y, cell_w, cell_h, FILL_RAIL)
        label(ax, x + cell_w / 2, cell_y + cell_h / 2, name, size=8)

    # --- 能力层：三列 ---
    col_w, col_gap = 42.0, 2.0
    col_x = [22.0, 22.0 + col_w + col_gap, 22.0 + 2 * (col_w + col_gap)]
    col_y, col_h = 47.5, 33.0
    headers = ("① 索引子系统", "② 上下文治理", "③ 记忆子系统")
    bodies = (
        ("标题层级结构化切分", "拼接上下文摘要", "知识图谱后端", "向量索引后端"),
        ("提示词按稳定性分层", "有效上下文窗口", "轮前预算规划", "轮内优先级驱逐"),
        ("短期记忆", "中期记忆", "长期记忆", "按学生个性化读写"),
    )
    for x, head, rows in zip(col_x, headers, bodies, strict=True):
        box(ax, x, col_y, col_w, col_h, FILL_BOX)
        box(ax, x, col_y, col_w, 6.4, FILL_HEAD, edgecolor=NAVY)
        label(ax, x + col_w / 2, col_y + 3.2, head, size=8, bold=True, color=TEXT_W)
        row_h = (col_h - 6.4) / 4
        for i, row in enumerate(rows):
            ry = col_y + 6.4 + i * row_h
            if i:
                ax.plot(
                    [x + 1.2, x + col_w - 1.2],
                    [ry, ry],
                    color="#C5CDD6",
                    lw=0.4,
                    zorder=5,
                )
            label(ax, x + col_w / 2, ry + row_h / 2, row, size=7.5)

    # --- 数据层 ---
    box(ax, 22.0, 86.0, 130.0, 9.5, FILL_BOX)
    label(ax, 87.0, 90.75, "PostgreSQL　Redis　外部模型接口（本地无 GPU）", size=8)

    # Orthogonal arrows: actors -> kernel -> three columns
    arrow(ax, 53.0, 17.0, 53.0, 23.0)
    arrow(ax, 121.0, 17.0, 121.0, 23.0)
    hub_y = 43.5
    xs = [x + col_w / 2 for x in col_x]
    ax.plot([xs[0], xs[-1]], [hub_y, hub_y], color=NAVY, lw=0.5, zorder=4)
    ax.plot([87.0, 87.0], [43.0, hub_y], color=NAVY, lw=0.5, zorder=4)
    for x in xs:
        arrow(ax, x, hub_y, x, 47.5)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def draw_pptx() -> None:
    from lxml import etree
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Mm, Pt

    prs = Presentation()
    prs.slide_width = Mm(W_MM)
    prs.slide_height = Mm(H_MM)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    navy = RGBColor(0x2F, 0x4A, 0x6D)
    white = RGBColor(0xFF, 0xFF, 0xFF)
    text_c = RGBColor(0x1A, 0x1A, 0x1A)
    fill_rail = RGBColor(0xE8, 0xEE, 0xF4)
    fill_layer = RGBColor(0xF4, 0xF7, 0xFA)
    fill_box = RGBColor(0xFF, 0xFF, 0xFF)
    fill_head = RGBColor(0x2F, 0x4A, 0x6D)

    def set_run(run, text, size=8, bold=False, color=text_c, cn="宋体"):
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Times New Roman"
        rPr = run._r.get_or_add_rPr()
        ea = rPr.find(qn("a:ea"))
        if ea is None:
            ea = etree.SubElement(rPr, qn("a:ea"))
        ea.set("typeface", "黑体" if bold else cn)

    def add_rect(x, y, w, h, fill, text="", *, size=8, bold=False, color=text_c, lw=0.5):
        shape = slide.shapes.add_shape(1, Mm(x), Mm(y), Mm(w), Mm(h))  # 1 = rectangle
        shape.adjustments  # keep shape
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = navy
        shape.line.width = Pt(lw)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        tf.anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.clear()
        if text:
            run = p.add_run()
            set_run(run, text, size=size, bold=bold, color=color)
        return shape

    def add_line(x1, y1, x2, y2):
        conn = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Mm(x1), Mm(y1), Mm(x2), Mm(y2)
        )
        conn.line.color.rgb = navy
        conn.line.width = Pt(0.5)
        return conn

    add_rect(4, 4, 152, 94, white, lw=0.75)

    rails = [(4, 4, 16, "交互层"), (4, 20, 25, "调度层"), (4, 45, 38, "子系统层"), (4, 83, 15, "数据层")]
    for x, y, h, name in rails:
        add_rect(x, y, 13, h, fill_rail, name, size=8, bold=True)
        add_rect(17, y, 139, h, fill_layer)

    add_rect(22, 7, 62, 10, fill_box, "教师", bold=True)
    add_rect(90, 7, 62, 10, fill_box, "学生", bold=True)

    add_rect(22, 23, 130, 20, fill_box)
    # title + subtitle as two overlapping boxes is messy; use one box and a second text box
    title = slide.shapes.add_textbox(Mm(22), Mm(23.4), Mm(130), Mm(6.2))
    tf = title.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "统一调度内核", size=8, bold=True)
    sub = slide.shapes.add_textbox(Mm(22), Mm(28.8), Mm(130), Mm(5.0))
    tf = sub.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "工具调用 · 推理—行动循环", size=7.5)

    cell_w, gap = 29.5, 2.0
    for i, name in enumerate(("问答", "解题", "出题", "研究")):
        add_rect(24.5 + i * (cell_w + gap), 34.2, cell_w, 7.2, fill_rail, name)

    cols = [
        (22.0, "① 索引子系统", ("标题层级结构化切分", "拼接上下文摘要", "知识图谱后端", "向量索引后端")),
        (66.0, "② 上下文治理", ("提示词按稳定性分层", "有效上下文窗口", "轮前预算规划", "轮内优先级驱逐")),
        (110.0, "③ 记忆子系统", ("短期记忆", "中期记忆", "长期记忆", "按学生个性化读写")),
    ]
    for x, head, rows in cols:
        add_rect(x, 47.5, 42.0, 33.0, fill_box)
        add_rect(x, 47.5, 42.0, 6.4, fill_head, head, bold=True, color=white)
        row_h = (33.0 - 6.4) / 4
        for i, row in enumerate(rows):
            add_rect(x, 47.5 + 6.4 + i * row_h, 42.0, row_h, fill_box, row, size=7.5)

    add_rect(22, 86.0, 130, 9.5, fill_box, "PostgreSQL　Redis　外部模型接口（本地无 GPU）")

    add_line(53, 17, 53, 23)
    add_line(121, 17, 121, 23)
    add_line(43, 43, 131, 43)
    add_line(43, 43, 43, 47.5)
    add_line(87, 43, 87, 47.5)
    add_line(131, 43, 131, 47.5)

    prs.save(STEM.with_suffix(".pptx"))


if __name__ == "__main__":
    draw_matplotlib()
    draw_pptx()
    print(f"wrote {STEM}.pdf / .png / .svg / .pptx")
