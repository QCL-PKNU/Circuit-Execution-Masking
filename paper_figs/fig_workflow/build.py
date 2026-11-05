"""Q-MaskCut workflow figure, rebuilt from the author's fig_workflow.pptx (same colours) as one editable slide
at the printed size of the paper (IOP text width 153 mm = 6.02 in), so a font size here is the printed size.
Every gate, wire and label is a native shape (no pictures)."""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

OUT = sys.argv[1]
W, H = 6.02, 4.20
ORANGE = RGBColor(0xDF, 0x6A, 0x10)
PURPLE = RGBColor(0x70, 0x30, 0xA0); PURPLE_BG = RGBColor(0xEC, 0xEA, 0xFB); BLUE = RGBColor(0x34, 0x28, 0xC6)
TEAL = RGBColor(0x00, 0x8A, 0x7D); TEAL_BG = RGBColor(0xE6, 0xF5, 0xF3)
RED = RGBColor(0xC0, 0x00, 0x00); RED_BG = RGBColor(0xFD, 0xE4, 0xE4)
GREEN = RGBColor(0x2E, 0x7D, 0x32); GREEN_BG = RGBColor(0xEB, 0xF4, 0xEB)
GREY = RGBColor(0x9E, 0x9E, 0x9E); DARK = RGBColor(0x1A, 0x1A, 0x1A); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NAVY = RGBColor(0x01, 0x2A, 0x51)
FONT = "Arial"
FS = 7.5      # body text, pt (printed)
FT = 8.5      # panel titles

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6])
S = sl.shapes


def _line_fmt(shape, color, width, dash=None):
    ln = shape.line
    if color is None:
        ln.fill.background(); return
    ln.color.rgb = color; ln.width = Pt(width)
    if dash:
        ln._get_or_add_ln().append(etree.fromstring(f'<a:prstDash xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{dash}"/>'))


def text(x, y, w, h, s, size=FS, bold=False, color=DARK, align="c", anchor="m", italic=False):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = {"m": MSO_ANCHOR.MIDDLE, "t": MSO_ANCHOR.TOP, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for i, part in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[align]
        r = p.add_run(); r.text = part
        f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.name = FONT; f.color.rgb = color
    return tb


def box(x, y, w, h, fill=WHITE, line=ORANGE, lw=0.9, dash=None, s=None, size=FS, bold=False, color=DARK,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, align="c"):
    b = S.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = min(0.5, 0.06 / min(w, h))
    if fill is None:
        b.fill.background()
    else:
        b.fill.solid(); b.fill.fore_color.rgb = fill
    _line_fmt(b, line, lw, dash)
    b.shadow.inherit = False
    tf = b.text_frame
    tf.margin_left = tf.margin_right = Inches(0.03); tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
    if s is not None:
        for i, part in enumerate(s.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT}[align]
            r = p.add_run(); r.text = part
            f = r.font; f.size = Pt(size); f.bold = bold; f.name = FONT; f.color.rgb = color
    return b


def line(x1, y1, x2, y2, color=DARK, lw=0.75, arrow=False, dash=None):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(etree.fromstring(f'<a:prstDash xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" val="{dash}"/>'))
    if arrow:
        ln.append(etree.fromstring('<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'))
    return c


def panel(x, y, w, h, title, sub=None):
    box(x, y, w, h, fill=None, line=ORANGE, lw=1.0, dash="dash", shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(x + 0.06, y + 0.03, w - 0.12, 0.17, title, size=FT, bold=True, align="l")
    if sub:
        text(x + 0.06, y + 0.03, w - 0.12, 0.17, sub, size=FS - 0.5, color=NAVY, align="r", italic=True)


# ---------- circuit primitives ----------
G = 0.165   # gate box size (in)


def gate(cx, cy, label, fill=WHITE, edge=ORANGE, color=DARK, w=G, size=7):
    box(cx - w / 2, cy - G / 2, w, G, fill=fill, line=edge, lw=0.8, s=label, size=size, bold=True, color=color,
        shape=MSO_SHAPE.RECTANGLE)


def dot(cx, cy, color=DARK, r=0.03):
    d = S.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    d.fill.solid(); d.fill.fore_color.rgb = color; d.line.fill.background(); d.shadow.inherit = False


def target(cx, cy, color=DARK, r=0.06):
    o = S.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    o.fill.solid(); o.fill.fore_color.rgb = WHITE; _line_fmt(o, color, 0.8); o.shadow.inherit = False
    line(cx - r, cy, cx + r, cy, color, 0.8); line(cx, cy - r, cx, cy + r, color, 0.8)


def cx_gate(x, yc, yt, color=DARK):
    line(x, yc, x, yt + (0.06 if yt > yc else -0.06), color, 0.8)
    dot(x, yc, color); target(x, yt, color)


def cz_gate(x, ya, yb, color=PURPLE):
    line(x, ya, x, yb, color, 0.8); dot(x, ya, color); dot(x, yb, color)


def wire_label(x, y, s, color=DARK, circle=True):
    if circle:
        r = 0.075
        o = S.add_shape(MSO_SHAPE.OVAL, Inches(x - r), Inches(y - r), Inches(2 * r), Inches(2 * r))
        o.fill.solid(); o.fill.fore_color.rgb = WHITE; _line_fmt(o, color, 0.8); o.shadow.inherit = False
        text(x - 0.1, y - 0.08, 0.2, 0.16, s, size=6.5, color=color)
    else:
        text(x - 0.16, y - 0.08, 0.3, 0.16, s, size=FS, color=color, bold=True)


def arrow_label(x, y, w, s, color=NAVY):
    text(x, y, w, 0.16, s, size=FS - 0.5, color=color, italic=True)


# =========================== (a) original circuit ===========================
ax, ay, aw, ah = 0.04, 0.04, 1.50, 1.98
panel(ax, ay, aw, ah, "(a) Original circuit")
ys = [ay + 0.50 + i * 0.27 for i in range(4)]
x0, x1 = ax + 0.15, ax + aw - 0.08
cols = [ax + 0.38 + i * 0.235 for i in range(5)]
for i, y in enumerate(ys):
    line(x0 + 0.075, y, x1, y, DARK, 0.75)
    wire_label(x0, y, str(i))
gate(cols[0], ys[0], "H")
cx_gate(cols[1], ys[0], ys[1]); gate(cols[1], ys[2], "T")
gate(cols[2], ys[1], "X"); cx_gate(cols[2], ys[2], ys[3])
for y in ys:
    gate(cols[4] - 0.05, y, "M")
text(ax + 0.06, ay + ah - 0.44, aw - 0.12, 0.40, "correct output\n0001 or 0010 (bits q3q2q1q0)", size=FS - 0.5,
     color=NAVY)

# =========================== (b) protection + cutting ===========================
bx, by, bw, bh = 1.76, 0.04, 4.22, 1.98
panel(bx, by, bw, bh, "(b) Protection and cutting", "client, before compilation")
names = ["q0", "q1", "d0", "q2", "d1", "q3"]
decoy = {"d0", "d1"}
yb = {n: by + 0.47 + i * 0.20 for i, n in enumerate(names)}
lx = bx + 0.17
c = [bx + 0.46 + i * 0.262 for i in range(10)]
xend = c[9] + 0.12
for n in names:
    col = PURPLE if n in decoy else DARK
    line(lx + 0.13, yb[n], xend, yb[n], col, 0.75, dash="sysDash" if n in decoy else None)
    wire_label(lx, yb[n], n, col, circle=False)
# decoy preparation |1> and original gates
gate(c[0], yb["d0"], "X", PURPLE_BG, PURPLE, PURPLE); gate(c[0], yb["d1"], "X", PURPLE_BG, PURPLE, PURPLE)
gate(c[0], yb["q0"], "H")
cx_gate(c[1], yb["q0"], yb["q1"]); gate(c[1], yb["q2"], "T")
gate(c[2], yb["q1"], "X"); cx_gate(c[2], yb["q2"], yb["q3"])
# rotation pair across the cut on q1
gate(c[3] + 0.03, yb["q1"], "RX(+t)", TEAL_BG, TEAL, TEAL, w=0.36, size=6)
xc = c[4] + 0.02
line(xc, yb["q0"] - 0.10, xc, yb["q1"] + 0.13, RED, 1.25, dash="dash")
text(xc - 0.2, yb["q0"] - 0.24, 0.4, 0.14, "cut", size=FS - 0.5, color=RED, bold=True)
gate(c[5] + 0.01, yb["q1"], "RX(−t)", TEAL_BG, TEAL, TEAL, w=0.36, size=6)
# padlocks (decoy in |1>, so CZ acts as Z on the data wire)
cz_gate(c[6], yb["q1"], yb["d0"]); cz_gate(c[6] + 0.08, yb["q2"], yb["d1"])
# key X on q0, q1, q3 (k = 1011) and H on each decoy
for n in ("q0", "q1", "q3"):
    gate(c[7] + 0.04, yb[n], "X", RED_BG, RED, RED)
for n in decoy:
    gate(c[7] + 0.04, yb[n], "H", PURPLE_BG, PURPLE, PURPLE)
for n in names:
    gate(c[8] + 0.06, yb[n], "M", WHITE, PURPLE if n in decoy else ORANGE)
# fragments
fx = c[9] + 0.22
fw = bx + bw - fx - 0.08
text(fx, by + 0.26, fw, 0.18, "CutQC: 1 cut", size=FS, bold=True)
box(fx, by + 0.46, fw, 0.42, fill=TEAL_BG, line=TEAL, s="Fragment A\nq0, q1 to the cut", size=FS - 0.5)
box(fx, by + 0.94, fw, 0.50, fill=TEAL_BG, line=TEAL, s="Fragment B\nq1 after the cut,\nd0, q2, d1, q3", size=FS - 0.5)
text(fx - 0.05, by + 1.46, fw + 0.1, 0.18, "→ 7 fragment variants", size=FS - 0.5, color=NAVY)
line(xend + 0.01, by + 0.95, fx - 0.02, by + 0.95, DARK, 0.75, arrow=True)
# legend
ly = by + bh - 0.17
items = [(RED_BG, RED, "X", "key k = 1011"), (PURPLE_BG, PURPLE, "X", "decoy set to |1⟩, H at end"),
         (None, PURPLE, None, "padlock CZ"), (TEAL_BG, TEAL, "RX", "rotation pair")]
lxs = [bx + 0.12, bx + 0.98, bx + 2.42, bx + 3.22]
for (fill, edge, lab, desc), x in zip(items, lxs):
    if lab:
        gate(x + 0.08, ly, lab, fill, edge, edge, w=0.20 if lab == "RX" else G, size=6)
    else:
        cz_gate(x + 0.08, ly - 0.06, ly + 0.06)
    text(x + 0.20, ly - 0.08, 1.4, 0.16, desc, size=FS - 0.5, align="l")
line(ax + aw + 0.01, ay + 1.0, bx - 0.01, ay + 1.0, DARK, 1.0, arrow=True)

# =========================== (c) compile + mask ===========================
row2 = 2.14
cx_, cw, ch = 0.04, 1.92, 2.02
panel(cx_, row2, cw, ch, "(c) Compilation and masking")
text(cx_ + 0.06, row2 + 0.19, cw - 0.12, 0.15, "client, every fragment variant", size=FS - 0.5, color=NAVY, italic=True, align="l")
box(cx_ + 0.10, row2 + 0.40, cw - 0.20, 0.30, s="Compile: optimisation level 3,\nbest of 8 layouts, good qubits", size=FS - 0.5)
line(cx_ + cw / 2, row2 + 0.70, cx_ + cw / 2, row2 + 0.79, DARK, 0.75, arrow=True)
# a compiled fragment with its masking: twirl Paulis around the CZ, DD pulses (product = identity) in an idle gap
y1, y2 = row2 + 0.94, row2 + 1.22
for y in (y1, y2):
    line(cx_ + 0.10, y, cx_ + cw - 0.08, y, DARK, 0.75)
gate(cx_ + 0.27, y1, "SX", w=0.19, size=6); gate(cx_ + 0.27, y2, "RZ", w=0.19, size=6)
for x, s in ((cx_ + 0.52, "P"), (cx_ + 0.98, "P\u2032")):
    for y in (y1, y2):
        gate(x, y, s, fill=PURPLE_BG, edge=PURPLE, color=PURPLE, w=0.19, size=6)
cz_gate(cx_ + 0.75, y1, y2, DARK)
for x, s in ((cx_ + 1.24, "RZ"), (cx_ + 1.46, "SX"), (cx_ + 1.68, "RZ")):
    gate(x, y1, s, w=0.19, size=6)
for x, s in ((cx_ + 1.24, "Y"), (cx_ + 1.46, "X"), (cx_ + 1.68, "Z")):
    gate(x, y2, s, fill=WHITE, edge=BLUE, color=BLUE, w=0.19, size=6)
text(cx_ + 0.06, row2 + 1.36, cw - 0.12, 0.14, "P, P\u2032: Pauli twirl around each CZ", size=FS - 1, color=PURPLE, align="l")
text(cx_ + 0.06, row2 + 1.51, cw - 0.12, 0.14, "Y X Z: no-pattern DD in an idle gap (= I)", size=FS - 1, color=BLUE, align="l")
text(cx_ + 0.06, row2 + 1.68, cw - 0.12, 0.30, "added after the user's compilation;\na provider that recompiles removes it",
     size=FS - 0.5, color=BLUE, italic=True)
# (b) -> (c)
line(bx + 0.25, by + bh + 0.005, cx_ + cw - 0.25, row2 - 0.005, DARK, 1.0, arrow=True)

# =========================== provider ===========================
px, pw = 2.10, 1.06
box(px, row2, pw, ch, fill=RGBColor(0xF4, 0xF4, 0xF4), line=GREY, lw=1.0, dash="dash")
text(px + 0.03, row2 + 0.05, pw - 0.06, 0.32, "Cloud provider\n(untrusted)", size=FT, bold=True)
box(px + 0.07, row2 + 0.62, pw - 0.14, 0.48, fill=WHITE, line=ORANGE, s="runs the masked\nfragment variants\n(ibm_marrakesh)", size=FS - 0.5)
text(px + 0.03, row2 + 1.22, pw - 0.06, 0.70, "sees only\nmasked fragments\nand key-flipped\ncounts", size=FS - 0.5,
     color=NAVY, italic=True, anchor="t")
line(cx_ + cw + 0.01, row2 + 0.86, px - 0.01, row2 + 0.86, DARK, 1.0, arrow=True)

# =========================== (d) reconstruction ===========================
dx = px + pw + 0.14
dw = W - dx - 0.04
panel(dx, row2, dw, ch, "(d) Reconstruction and decoding", None)
text(dx + 0.06, row2 + 0.19, dw - 0.12, 0.15, "client", size=FS - 0.5, color=NAVY, italic=True, align="l")
line(px + pw + 0.01, row2 + 0.86, dx - 0.01, row2 + 0.86, DARK, 1.0, arrow=True)
stw = 1.08
sx = dx + 0.08
dsteps = [("CutQC reconstruction", WHITE, ORANGE, DARK), ("drop decoy bits d0, d1", WHITE, ORANGE, DARK),
          ("XOR with key k = 1011", PURPLE_BG, BLUE, BLUE)]
sy = row2 + 0.42
for i, (s, f, e, col) in enumerate(dsteps):
    box(sx, sy, stw, 0.30, fill=f, line=e, s=s, size=FS - 0.5, color=col)
    if i < 2:
        line(sx + stw / 2, sy + 0.30, sx + stw / 2, sy + 0.40, DARK, 0.75, arrow=True)
    sy += 0.40
text(sx - 0.02, sy - 0.02, stw + 0.04, 0.40, "exact: decoded output\n= original output", size=FS - 0.5, color=GREEN, italic=True)

# histograms over the 16 outcomes q3q2q1q0
import random
random.seed(3)
hx = sx + stw + 0.14
hw = dx + dw - hx - 0.08


def hist(y0, hh, title, peaks, peak_col):
    text(hx - 0.04, y0, hw + 0.08, 0.16, title, size=FS - 0.5, bold=True)
    base = y0 + hh
    line(hx, base, hx + hw, base, DARK, 0.6)
    bwid = hw / 16
    noise = [random.uniform(0.01, 0.03) for _ in range(16)]
    tot = sum(noise[i] for i in range(16) if i not in peaks)
    top = None
    for i in range(16):
        v = (1 - tot) / len(peaks) if i in peaks else noise[i]
        bh_ = v * 2 * (hh - 0.36)
        b = S.add_shape(MSO_SHAPE.RECTANGLE, Inches(hx + i * bwid + 0.006), Inches(base - bh_),
                        Inches(bwid - 0.012), Inches(bh_))
        b.fill.solid(); b.fill.fore_color.rgb = peak_col if i in peaks else GREY
        b.line.fill.background(); b.shadow.inherit = False
        if i in peaks:
            top = base - bh_
    xm = hx + (min(peaks) + max(peaks) + 1) / 2 * bwid
    xl = min(max(xm - 0.45, hx), hx + hw - 0.9)
    text(xl, top - 0.15, 0.9, 0.14, " and ".join(format(i, "04b") for i in peaks), size=6.5,
         color=peak_col, bold=True)


hist(row2 + 0.22, 0.82, "without key (provider view)", [0b1001, 0b1010], ORANGE)
hist(row2 + 1.08, 0.80, "with key (owner)", [0b0001, 0b0010], GREEN)
text(hx - 0.04, row2 + ch - 0.13, hw + 0.08, 0.12, "16 outcomes, q3q2q1q0", size=6, color=NAVY)

prs.save(OUT)
print("saved", OUT)
