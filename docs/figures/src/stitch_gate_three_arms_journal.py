"""Journal / conference-style figure: three-arm stitch-gate comparison (终稿 §5.2 / 图 4).

Same query and student state; three policies differ only in what learning-status
is attached. Navy palette matches fig1 / memory_three_layers; token chips and
pass/block marks replace the previous chat-UI mockup.

Outputs:
  docs/figures/stitch_gate_three_arms.png
  docs/figures/stitch_gate_three_arms.pdf
  docs/figures/stitch_gate_three_arms.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STEM = OUT / "stitch_gate_three_arms"

W_MM, H_MM = 160.0, 112.0
NAVY = "#2F4A6D"
EDGE = "#2F4A6D"
FILL_RAIL = "#E8EEF4"
FILL_BOX = "#FFFFFF"
FILL_HEAD = "#2F4A6D"
TEXT = "#1A1A1A"
TEXT_W = "#FFFFFF"
MUTED = "#5C6570"
TEAL = "#2C6E5E"
TEAL_FILL = "#E7F1ED"
CORAL = "#B54A3C"
CORAL_FILL = "#F6E8E6"
AMBER = "#8A5A22"
AMBER_FILL = "#F3E6D0"
DASH = "#9AA3AE"
GRAY_FILL = "#F4F5F7"

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


def rbox(ax, x, y, w, h, facecolor, edgecolor=EDGE, lw=0.5, rad=0.9, z=3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.1,rounding_size={rad}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
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
        zorder=8,
    )


def arrow(ax, x1, y1, x2, y2, *, color=NAVY, lw=0.55, dashed=False, scale=8, z=4):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            linestyle=(0, (2.2, 1.5)) if dashed else "solid",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def chip(ax, x, y, w, h, text, *, fill, edge, tcolor=TEXT, size=6.2, bold=False):
    rbox(ax, x, y, w, h, fill, edgecolor=edge, lw=0.45, rad=0.85, z=5)
    label(ax, x + w / 2, y + h / 2, text, size=size, bold=bold, color=tcolor)


def diamond(ax, cx, cy, rw, rh, fill, edge=EDGE, lw=0.55):
    ax.add_patch(
        Polygon(
            [(cx, cy - rh), (cx + rw, cy), (cx, cy + rh), (cx - rw, cy)],
            closed=True,
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=5,
        )
    )


def mark_ok(ax, x, y):
    ax.plot(
        [x - 1.05, x - 0.2, x + 1.25],
        [y + 0.05, y + 1.15, y - 1.25],
        color=TEAL,
        lw=1.05,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=9,
        clip_on=False,
    )


def mark_no(ax, x, y, color=CORAL):
    ax.plot([x - 1.05, x + 1.05], [y - 1.05, y + 1.05], color=color, lw=0.95, solid_capstyle="round", zorder=9)
    ax.plot([x - 1.05, x + 1.05], [y + 1.05, y - 1.05], color=color, lw=0.95, solid_capstyle="round", zorder=9)


def draw_matplotlib() -> None:
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), dpi=300, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W_MM)
    ax.set_ylim(H_MM, 0)
    ax.set_axis_off()
    ax.set_facecolor("white")

    mx = 5.0
    g_x, g_y, g_w, g_h = mx, 3.4, 52.5, 27.6
    q_x = g_x + g_w + 2.2
    q_w = W_MM - mx - q_x
    q_y, q_h = g_y, g_h

    # ---- left: prereq edge ----
    box(ax, g_x, g_y, g_w, g_h, FILL_BOX, lw=0.55)
    box(ax, g_x, g_y, g_w, 5.0, FILL_HEAD, edgecolor=NAVY, lw=0.55)
    label(ax, g_x + g_w / 2, g_y + 2.5, "课程先修坐标", size=7.3, bold=True, color=TEXT_W)

    ohm_w, kcl_w, nh = 17.6, 14.8, 6.2
    ohm_x = g_x + 4.6
    ohm_y = g_y + 8.6
    kcl_x = g_x + g_w - 4.6 - kcl_w
    chip(ax, ohm_x, ohm_y, ohm_w, nh, "欧姆定律", fill=AMBER_FILL, edge=AMBER, tcolor=AMBER, size=6.8, bold=True)
    chip(ax, kcl_x, ohm_y, kcl_w, nh, "KCL", fill=FILL_RAIL, edge=NAVY, tcolor=NAVY, size=6.8, bold=True)
    arrow(ax, ohm_x + ohm_w, ohm_y + nh / 2, kcl_x, ohm_y + nh / 2, scale=7)
    label(ax, (ohm_x + ohm_w + kcl_x) / 2, ohm_y - 1.55, "先修", size=6.3, color=MUTED)
    label(ax, ohm_x + ohm_w / 2, ohm_y + nh + 2.05, "薄弱 · 可提醒", size=6.1, color=AMBER)
    label(ax, kcl_x + kcl_w / 2, ohm_y + nh + 2.05, "本轮所问", size=6.1, color=NAVY)

    noise_y = g_y + g_h - 6.6
    label(ax, g_x + 3.0, noise_y + 1.5, "无先修边", size=6.1, color=MUTED, ha="left")
    nx0 = g_x + 16.6
    for i, name in enumerate(("电容", "诺顿", "戴维南")):
        chip(ax, nx0 + i * 11.4, noise_y - 0.8, 10.6, 4.6, name, fill=GRAY_FILL, edge=DASH, tcolor=MUTED, size=5.7)

    # ---- right: shared query ----
    box(ax, q_x, q_y, q_w, q_h, FILL_BOX, lw=0.55)
    box(ax, q_x, q_y, q_w, 5.0, FILL_HEAD, edgecolor=NAVY, lw=0.55)
    label(ax, q_x + q_w / 2, q_y + 2.5, "同一问句 · 同一学情（三臂仅策略不同）", size=7.3, bold=True, color=TEXT_W)

    label(ax, q_x + 3.2, q_y + 9.4, "问句", size=6.3, bold=True, color=MUTED, ha="left")
    rbox(ax, q_x + 11.6, q_y + 6.8, q_w - 14.8, 5.2, FILL_RAIL, edgecolor=NAVY, lw=0.4, rad=0.7)
    label(ax, q_x + q_w / 2 + 2.0, q_y + 9.4, "节点电流方程怎么列？", size=7.3, bold=True)

    label(ax, q_x + 3.2, q_y + 16.8, "候选", size=6.3, bold=True, color=MUTED, ha="left")
    cands = [
        (11.6, 24.8, "欧姆定律 · 先修薄弱", AMBER_FILL, AMBER, AMBER, True),
        (37.6, 14.2, "电容充放电", GRAY_FILL, DASH, MUTED, False),
        (52.8, 13.4, "诺顿定理", GRAY_FILL, DASH, MUTED, False),
        (67.2, 15.2, "戴维南定理", GRAY_FILL, DASH, MUTED, False),
        (82.6, 12.2, "叠加原理", GRAY_FILL, DASH, MUTED, False),
    ]
    cy = q_y + 14.2
    for dx, w, txt, fill, edge, tc, bold in cands:
        chip(ax, q_x + dx, cy, w, 5.2, txt, fill=fill, edge=edge, tcolor=tc, size=5.9, bold=bold)

    label(ax, q_x + 3.2, q_y + 23.6, "橙：本题先修薄弱，可入闸", size=6.0, color=AMBER, ha="left")
    label(ax, q_x + 48.0, q_y + 23.6, "灰：无关薄弱点，本轮不入闸", size=6.0, color=MUTED, ha="left")

    # ---- three columns ----
    col_y = 33.6
    col_h = 74.6
    col_w = 48.8
    gap = 1.8
    xs = [mx, mx + col_w + gap, mx + 2 * (col_w + gap)]
    titles = ("不提醒", "无条件全提醒", "按需求概率提醒")
    subs = ("基线 · 关闭注入", "基线 · 照单全收", "本文 · 先修图 + 需求概率")
    head_h = 10.2

    rail_y = (q_y + q_h + col_y) / 2
    ax.plot([xs[0] + col_w / 2, xs[2] + col_w / 2], [rail_y, rail_y], color=NAVY, lw=0.5, zorder=3)
    for x in xs:
        cx = x + col_w / 2
        ax.plot([cx, cx], [q_y + q_h, col_y], color=NAVY, lw=0.5, zorder=3)

    names = ("欧姆", "电容", "诺顿", "戴维南", "叠加")
    tw, th, tg = 7.6, 4.5, 0.85

    for i, x in enumerate(xs):
        is_ours = i == 2
        box(ax, x, col_y, col_w, col_h, TEAL_FILL if is_ours else FILL_BOX, lw=0.75 if is_ours else 0.5)
        box(
            ax,
            x,
            col_y,
            col_w,
            head_h,
            FILL_HEAD if is_ours else FILL_RAIL,
            edgecolor=NAVY,
            lw=0.75 if is_ours else 0.5,
        )
        hc = TEXT_W if is_ours else NAVY
        label(ax, x + 5.2, col_y + 3.5, f"({chr(97 + i)})", size=8, bold=True, color=hc, ha="left")
        label(ax, x + col_w / 2 + 2.4, col_y + 3.5, titles[i], size=8, bold=True, color=hc)
        label(ax, x + col_w / 2, col_y + 7.8, subs[i], size=6.2, color="#D5DEE8" if is_ours else MUTED)

        mech_y = col_y + head_h + 1.8
        mech_h = 25.6
        box(ax, x + 1.5, mech_y, col_w - 3.0, mech_h, FILL_BOX, lw=0.4, edgecolor="#C5CDD6")
        label(ax, x + 2.8, mech_y + 2.3, "候选过闸", size=6.1, bold=True, color=MUTED, ha="left")

        row_w = 5 * tw + 4 * tg
        tx0 = x + (col_w - row_w) / 2
        ty = mech_y + 5.2
        for k, nm in enumerate(names):
            fill, edge, tc = (GRAY_FILL, DASH, MUTED)
            if k == 0:
                fill, edge, tc = (AMBER_FILL, AMBER, AMBER)
            chip(ax, tx0 + k * (tw + tg), ty, tw, th, nm, fill=fill, edge=edge, tcolor=tc, size=5.4, bold=(k == 0))
            mxk = tx0 + k * (tw + tg) + tw - 0.15
            myk = ty - 0.15
            if i == 0:
                mark_no(ax, mxk, myk, color=DASH)
            elif i == 1:
                ax.plot(
                    [mxk - 1.05, mxk - 0.2, mxk + 1.25],
                    [myk + 0.05, myk + 1.15, myk - 1.25],
                    color=CORAL,
                    lw=1.05,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=9,
                    clip_on=False,
                )
            else:
                if k == 0:
                    mark_ok(ax, mxk, myk)
                else:
                    mark_no(ax, mxk, myk)

        policy_y = mech_y + 12.4
        if i == 0:
            label(ax, x + col_w / 2, policy_y, "闸门关闭 · 全部拦截", size=6.4, bold=True, color=NAVY)
        elif i == 1:
            label(ax, x + col_w / 2, policy_y, "闸门全开 · 无关项一并注入", size=6.4, bold=True, color=CORAL)
        else:
            diamond(ax, x + 8.6, policy_y, 3.3, 2.15, TEAL_FILL, edge=TEAL, lw=0.55)
            label(ax, x + 8.6, policy_y, "门", size=5.4, bold=True, color=TEAL)
            label(ax, x + 26.8, policy_y, "p_need 高，仅欧姆定律通过", size=6.3, bold=True, color=TEAL)

        out_ty = mech_y + 16.4
        if i == 0:
            rbox(ax, x + 9.5, out_ty, col_w - 19.0, 6.8, GRAY_FILL, edgecolor=DASH, lw=0.45, rad=0.7)
            label(ax, x + col_w / 2, out_ty + 3.4, "无学情通过", size=6.5, color=MUTED)
        elif i == 1:
            otw, otg = 7.4, 0.7
            ox0 = x + (col_w - (5 * otw + 4 * otg)) / 2
            for k, nm in enumerate(names):
                if k == 0:
                    chip(ax, ox0 + k * (otw + otg), out_ty + 1.0, otw, 4.8, nm, fill=AMBER_FILL, edge=AMBER, tcolor=AMBER, size=5.3, bold=True)
                else:
                    chip(ax, ox0 + k * (otw + otg), out_ty + 1.0, otw, 4.8, nm, fill=CORAL_FILL, edge=CORAL, tcolor=CORAL, size=5.3)
        else:
            chip(ax, x + col_w / 2 - 10.4, out_ty + 1.0, 20.8, 5.0, "欧姆定律", fill=TEAL_FILL, edge=TEAL, tcolor=TEAL, size=6.3, bold=True)

        # output
        out_y = mech_y + mech_h + 1.6
        out_h = 22.0
        box(ax, x + 1.5, out_y, col_w - 3.0, out_h, FILL_BOX, lw=0.4, edgecolor="#C5CDD6")
        label(ax, x + 2.8, out_y + 2.4, "本轮输出", size=6.1, bold=True, color=MUTED, ha="left")
        rbox(ax, x + 2.8, out_y + 4.4, col_w - 5.6, 4.8, FILL_RAIL, edgecolor=NAVY, lw=0.35, rad=0.55)
        label(ax, x + col_w / 2, out_y + 6.8, "KCL 列写步骤（三臂相同）", size=6.2, color=NAVY)

        if i == 0:
            rbox(ax, x + 2.8, out_y + 10.6, col_w - 5.6, 9.2, GRAY_FILL, edgecolor=DASH, lw=0.4, rad=0.55)
            label(ax, x + col_w / 2, out_y + 13.4, "不附带学情", size=7.0, bold=True, color=MUTED)
            label(ax, x + col_w / 2, out_y + 17.4, "先修缺口被漏掉", size=6.2, color=MUTED)
        elif i == 1:
            rbox(ax, x + 2.8, out_y + 10.6, col_w - 5.6, 9.2, CORAL_FILL, edgecolor=CORAL, lw=0.45, rad=0.55)
            label(ax, x + col_w / 2, out_y + 13.4, "附带五条无关薄弱点", size=6.5, bold=True, color=CORAL)
            label(ax, x + col_w / 2, out_y + 17.4, "电容 / 诺顿 / 戴维南 / 叠加 …", size=6.0, color=CORAL)
        else:
            rbox(ax, x + 2.8, out_y + 10.6, col_w - 5.6, 9.2, TEAL_FILL, edgecolor=TEAL, lw=0.5, rad=0.55)
            label(ax, x + col_w / 2, out_y + 13.3, "一句轻提醒（仅欧姆定律）", size=6.4, bold=True, color=TEAL)
            label(ax, x + col_w / 2, out_y + 17.4, "列写 KCL 需用欧姆定律做支路转换", size=5.8, color=TEAL)

        v_y = out_y + out_h + 1.5
        v_h = col_y + col_h - v_y - 1.5
        if i == 0:
            vf, ve, vt = GRAY_FILL, NAVY, NAVY
            v1, v2 = "漏报（FN）", "均衡准确率  0.500"
        elif i == 1:
            vf, ve, vt = CORAL_FILL, CORAL, CORAL
            v1, v2 = "误报（FP）", "均衡准确率  0.500"
        else:
            vf, ve, vt = TEAL_FILL, TEAL, TEAL
            v1, v2 = "正确（TP + TN）", "均衡准确率  1.000"
        box(ax, x + 1.5, v_y, col_w - 3.0, v_h, vf, edgecolor=ve, lw=0.55)
        label(ax, x + col_w / 2, v_y + v_h * 0.38, v1, size=7.4, bold=True, color=vt)
        label(ax, x + col_w / 2, v_y + v_h * 0.72, v2, size=6.4, color=vt)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_matplotlib()
    print(f"wrote {STEM}.pdf / .png / .svg")
