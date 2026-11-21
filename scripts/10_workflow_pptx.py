"""İş akışı şeması (kullanıcının şablon sunumunun düzeninde): numaralı, kesik çizgili paneller."""
import sys, json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from PIL import Image

T = "/home/claude/work/thumbs/"; F = "/home/claude/figs3/"; OUT = sys.argv[1]
S = json.load(open(sys.argv[2]))  # sayılar
RED = RGBColor(0xC0, 0x00, 0x00); BLK = RGBColor(0, 0, 0); GRY = RGBColor(0x40, 0x40, 0x40)
FONT = "Times New Roman"

prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(10.0)
sl = prs.slides.add_slide(prs.slide_layouts[6])
sh = sl.shapes


def panel(x, y, w, h):
    r = sh.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.background(); r.line.color.rgb = RED; r.line.width = Pt(2.25); r.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    r.shadow.inherit = False
    return r


def text(x, y, w, h, s, size=12, bold=False, align=PP_ALIGN.LEFT, box=False, bullets=False, anchor=MSO_ANCHOR.TOP, color=BLK):
    if box:
        t = sh.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        t.fill.solid(); t.fill.fore_color.rgb = RGBColor(255, 255, 255)
        t.line.color.rgb = BLK; t.line.width = Pt(1); t.shadow.inherit = False
    else:
        t = sh.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = t.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(0.04))
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        sub = False
        if isinstance(ln, tuple):
            ln, sub = ln
        r = p.add_run(); r.text = ("•  " if bullets and not sub else ("      ▪  " if sub else "")) + ln
        r.font.size = Pt((size if not sub else size - 1) * (1.0 if size >= 13 else 1.25)); r.font.bold = bold; r.font.name = FONT; r.font.color.rgb = color
    return t


def num(x, y, n):
    text(x, y, 0.4, 0.45, str(n), size=24, bold=True, color=GRY)


def img(path, x, y, w=None, h=None, border=True):
    im = Image.open(path); ar = im.height / im.width
    if w is not None and h is not None:
        if w * ar > h: w = h / ar
        else: h = w * ar
    elif w is not None: h = w * ar
    else: w = h / ar
    p = sh.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    if border:
        p.line.color.rgb = RGBColor(0x80, 0x80, 0x80); p.line.width = Pt(0.75)
    return p, w, h


def arrow(x1, y1, x2, y2, elbow=False):
    c = sh.add_connector(MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor(0x40, 0x40, 0x40); c.line.width = Pt(2.25)
    ln = c.line._get_or_add_ln()
    from lxml import etree
    te = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
    te.set("type", "triangle"); te.set("w", "med"); te.set("len", "med")
    return c


# 1 GDF verisi
panel(0.15, 0.15, 3.45, 3.25); num(0.22, 0.2, 1)
text(0.6, 0.22, 2.9, 0.75, ["Hazelnut Stands from GDF", "Forest Management Plans"], 13, True, PP_ALIGN.CENTER)
img(T + "t_parcels.png", 0.3, 1.05, w=3.15)
text(0.25, 2.35, 3.3, 1.0, [f"{S['parsel']:,} pure stands", f"{S['ha']:,} ha, {S['bolge']} regional directorates"], 10.5, bullets=True)
# 2 GEE
panel(3.75, 0.15, 3.3, 3.25); num(3.82, 0.2, 2)
text(4.2, 0.22, 2.8, 0.75, ["Data Extraction in", "Google Earth Engine"], 13, True, PP_ALIGN.CENTER)
for k, lab in enumerate(["Sentinel-2 L2A, Cloud Score+", "ERA5-Land monthly climate", "SoilGrids, HWSD v2.0, terrain", "Spring reference composites"]):
    text(4.05, 1.0 + 0.42 * k, 2.7, 0.34, lab, 10, False, PP_ALIGN.CENTER, box=True)
arrow(5.4, 2.7, 5.4, 2.92)
text(4.05, 2.95, 2.7, 0.36, "Zonal statistics per parcel", 10.5, True, PP_ALIGN.CENTER, box=True)
# 3 Sentinel-2
panel(7.2, 0.15, 5.98, 3.25); num(7.27, 0.2, 3)
text(7.7, 0.22, 5.4, 0.45, "Sentinel-2 L2A Images, 2019–2026", 13, True, PP_ALIGN.CENTER)
text(7.4, 0.68, 1.75, 0.32, "NDMI, Apr–May", 10.5, True, PP_ALIGN.CENTER, box=True)
img(T + "t_ndmi_am.png", 7.4, 1.05, w=1.75)
text(9.35, 0.68, 1.75, 0.32, "NDMI, Jul–Aug", 10.5, True, PP_ALIGN.CENTER, box=True)
img(T + "t_ndmi_ja.png", 9.35, 1.05, w=1.75)
text(11.3, 0.68, 1.75, 0.32, "ΔNDMI", 10.5, True, PP_ALIGN.CENTER, box=True)
img(T + "t_dndmi.png", 11.3, 1.05, w=1.75)
arrow(9.17, 1.38, 9.33, 1.38); arrow(11.12, 1.38, 11.28, 1.38)
text(7.35, 1.85, 5.75, 1.55, ["Cloud Score+ masking, seasonal median composites",
                              "NDMI = (B8 − B11) / (B8 + B11), NDVI, NDRE",
                              "ΔNDMI = NDMI(Jul–Aug) − NDMI(Apr–May)",
                              f"SDI = within-parcel z-score, {S['py']:,} parcel-years"], 10.5, bullets=True)
# 4 Değişkenler
panel(0.15, 3.55, 6.9, 3.15); num(0.22, 3.6, 4)
text(0.6, 3.62, 6.3, 0.4, "Predictors, Correlation and VIF Screening", 13, True, PP_ALIGN.CENTER)
text(0.25, 4.05, 3.3, 2.65, ["41 candidates (climate, terrain, soil, stand)",
                             "Correlation, 18 pairs |r| > 0.80",
                             "Stepwise VIF ≤ 10 and pair filter",
                             (f"{S['nx']} predictors retained", True),
                             "Moran's I correlogram"], 10, bullets=True)
for i, (f, lab) in enumerate([("t_elev", "Elevation"), ("t_soildepth", "Soil depth"), ("t_precip", "Precipitation"), ("t_clay", "Clay")]):
    cx = 3.6 + (i % 2) * 1.72; cy = 4.05 + (i // 2) * 1.32
    text(cx, cy, 1.65, 0.28, lab, 10, True, PP_ALIGN.CENTER, box=True)
    img(T + f + ".png", cx, cy + 0.32, w=1.65)
# 5 Modeller
panel(7.2, 3.55, 5.98, 3.15); num(7.27, 3.6, 5)
text(7.7, 3.62, 5.4, 0.4, "Ten Machine Learning Algorithms", 13, True, PP_ALIGN.CENTER)
text(7.45, 4.1, 2.6, 0.32, "Classical and kernel", 10.5, True, PP_ALIGN.CENTER, box=True)
text(7.45, 4.48, 2.6, 1.3, ["Multiple linear regression", "Support vector regression", "k-nearest neighbours", "Multilayer perceptron"], 10.5, bullets=True)
text(10.4, 4.1, 2.6, 0.32, "Tree ensembles", 10.5, True, PP_ALIGN.CENTER, box=True)
text(10.4, 4.48, 2.7, 1.5, ["Random forest", "Extra trees", "Histogram gradient boosting", "XGBoost, LightGBM", "CatBoost"], 10.5, bullets=True)
text(7.4, 5.95, 5.7, 0.7, ["Default and nested CV tuned settings",
                           "Year × RD mean baseline"], 10.5, bullets=True)
# 6 Doğrulama
panel(8.88, 6.85, 4.3, 3.0); num(8.95, 6.9, 6)
text(9.33, 6.92, 3.7, 0.4, "Validation", 13, True, PP_ALIGN.CENTER)
img(F + "Fig_model_r2.png", 9.08, 7.35, w=2.3, h=2.4)
text(11.43, 7.4, 1.72, 2.4, ["Spatial block CV, 10–50 km", "Leave-one-year-out", "Block bootstrap CIs"], 10, bullets=True)
# 7 SHAP
panel(4.88, 6.85, 3.85, 3.0); num(4.95, 6.9, 7)
text(5.33, 6.92, 3.3, 0.4, "Model Interpretation", 13, True, PP_ALIGN.CENTER)
img(F + "Fig_shap_summary.png", 5.08, 7.35, w=2.2, h=2.4)
text(7.33, 7.4, 1.38, 2.4, ["TreeSHAP", "Group shares", "Dependence"], 10, bullets=True)
# 8 Haritalar
panel(0.15, 6.85, 4.58, 3.0); num(0.22, 6.9, 8)
text(0.6, 6.92, 4.05, 0.4, "Model Maps and Additional Checks", 13, True, PP_ALIGN.CENTER)
text(0.3, 7.35, 2.1, 0.3, "Predicted ΔNDMI, 2025", 9.5, True, PP_ALIGN.CENTER, box=True)
img(T + "t_pred_rf.png", 0.3, 7.7, w=2.1)
text(2.5, 7.35, 2.1, 0.3, "Observed SDI, 2025", 9.5, True, PP_ALIGN.CENTER, box=True)
img(T + "t_dsi.png", 2.5, 7.7, w=2.1)
text(0.25, 8.65, 4.45, 1.2, ["Prediction maps of all ten models", "Spring reference window test",
                            "Province yield comparison"], 10, bullets=True)
# oklar
arrow(3.6, 1.6, 3.75, 1.6); arrow(7.05, 1.6, 7.2, 1.6)
arrow(10.2, 3.4, 10.2, 3.55); arrow(1.9, 3.4, 1.9, 3.55)
arrow(7.05, 5.1, 7.2, 5.1)
arrow(11.0, 6.7, 11.0, 6.85); arrow(8.88, 8.35, 8.73, 8.35); arrow(4.88, 8.35, 4.73, 8.35)
prs.save(OUT)
print("ok")
