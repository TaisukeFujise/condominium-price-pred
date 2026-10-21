#!/usr/bin/env python3
"""配布された template.pptx の空欄を中身で埋め、最終発表のスライドを作る。

ページの構成・見出し・デザインはテンプレートのまま使う。中身の多い 2・3 枚目は同じページを複製して 2 枚にし、
最後の「演習を通して学んだこと、反省点」はテンプレートの指示どおりメンバーの人数分に複製する。
書体はテンプレートのテーマに任せ、文字に直接書かない。

uv run --with python-pptx python slides/final/fill_template.py slides/final/template.pptx slides/final/final.pptx
"""

import copy
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (
    XL_CHART_TYPE,
    XL_LABEL_POSITION,
    XL_LEGEND_POSITION,
    XL_MARKER_STYLE,
)
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

EMU = 914400
MEMBERS = ["南條 佑太", "大久保 翔", "藤瀬 太翼", "小林 麟太郎"]
BULLETS = {
    0: ("●", 228600, -228600),
    1: ("○", 457200, -228600),
    2: ("–", 685800, -228600),
}


def sub(parent, tag, **attrs):
    el = etree.SubElement(parent, qn(tag))
    for k, v in attrs.items():
        el.set(k, str(v))
    return el


# 本文の枠はテンプレートのマスターが Noto Sans JP を持つので書体を書かない。
# 新しく足す図形（表・図の箱・グラフ・一覧）はテーマの既定（Arial）になるので、本文にそろえて書く
FACE = "Noto Sans JP"


def run(p, text, sz, b=False, face=None):
    r = sub(p, "a:r")
    rPr = sub(r, "a:rPr", lang="ja-JP", sz=int(sz * 100), b=1 if b else 0)
    if face:
        for tag in ("a:latin", "a:ea", "a:cs"):
            sub(rPr, tag, typeface=face)
    sub(r, "a:t").text = text


def fill_body(shape, items, sizes=(12, 11, 10.5), line=None):
    """本文の枠を (階層, 文, 太字) の並びで埋める。記号はテンプレートと同じ ● ○ を箇条書きの書式で付ける。"""
    txBody = shape._element.find(qn("p:txBody"))
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)
    for lvl, text, b in items:
        p = sub(txBody, "a:p")
        ch, marL, ind = BULLETS[lvl]
        pPr = sub(p, "a:pPr", lvl=lvl, marL=marL, indent=ind, algn="l")
        if line:
            sub(sub(pPr, "a:lnSpc"), "a:spcPct", val=int(line * 1000))
        sub(sub(pPr, "a:spcBef"), "a:spcPts", val=600 if lvl == 0 else 100)
        sub(sub(pPr, "a:spcAft"), "a:spcPts", val=0)
        sub(sub(pPr, "a:buClr"), "a:srgbClr", val="000000" if lvl == 0 else "737373")
        sub(pPr, "a:buSzPct", val=80000 if lvl == 0 else 90000)
        sub(pPr, "a:buFont", typeface="Arial")
        sub(pPr, "a:buChar", char=ch)
        run(p, text, sizes[lvl], b)


def set_cell(tc, text, sz, b=False):
    txBody = tc.find(qn("a:txBody"))
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)
    for line in text.split("\n"):
        p = sub(txBody, "a:p")
        pPr = sub(p, "a:pPr", marL=0, indent=0, algn="l")
        sub(pPr, "a:buNone")
        run(p, line, sz, b, FACE)
    tcPr = tc.find(qn("a:tcPr"))
    for k in ("marT", "marB"):
        tcPr.set(k, str(int(0.03 * EMU)))
    tcPr.set("anchor", "ctr")


def fill_table(frame, widths, rows, y, sz, head_sz, row_h, head_h=0.3):
    """テンプレートの表の列幅を決め、行を足して埋める。見出し行の文言と罫線はテンプレートのまま。"""
    frame.left, frame.top = Inches(0.34), Inches(y)
    tbl = frame._element.find(".//" + qn("a:tbl"))
    for gc, w in zip(
        tbl.find(qn("a:tblGrid")).findall(qn("a:gridCol")), widths, strict=True
    ):
        gc.set("w", str(int(w * EMU)))
    trs = tbl.findall(qn("a:tr"))
    head, blank = trs[0], copy.deepcopy(trs[1])
    for tr in trs[1:]:
        tbl.remove(tr)
    head.set("h", str(int(head_h * EMU)))
    for tc in head.findall(qn("a:tc")):
        txt = "".join(t.text or "" for t in tc.iter(qn("a:t")))
        # 見出しの文言はそのまま、折り返す位置だけ意味の切れ目にそろえる
        txt = txt.replace("試したこと（", "試したこと\n（").replace(
            "、ずれ", "、\nずれ"
        )
        set_cell(tc, txt, head_sz, True)
    for cells in rows:
        tr = copy.deepcopy(blank)
        tr.set("h", str(int(row_h * EMU)))
        for tc, text in zip(tr.findall(qn("a:tc")), cells, strict=True):
            set_cell(tc, text, sz)
        tbl.append(tr)
    frame.width = Inches(sum(widths))
    frame.height = Inches(head_h + row_h * len(rows))


def placeholder(slide, idx):
    return next(s for s in slide.placeholders if s.placeholder_format.idx == idx)


def duplicate(prs, src):
    """src と同じレイアウト・同じ図形のページを作り、src の直後に置く。"""
    pos = list(prs.slides).index(src)
    new = prs.slides.add_slide(src.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in src.shapes:
        new.shapes._spTree.append(copy.deepcopy(sh._element))
    ids = prs.slides._sldIdLst
    moved = ids[-1]
    ids.remove(moved)
    ids.insert(pos + 1, moved)
    return new


def set_title(slide, text):
    tf = placeholder(slide, 0).text_frame
    p = tf.paragraphs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    p.runs[0].text = text


prs = Presentation(sys.argv[1])
s_cover, s_model, s_score, s_ai, s_good, s_bad, s_self = prs.slides
s_story = duplicate(prs, s_model)
s_team = duplicate(prs, s_story)
s_feat = duplicate(prs, s_model)
s_score2 = duplicate(prs, s_score)
s_probe = duplicate(prs, s_score2)
selves = [s_self]
for _ in MEMBERS[1:]:
    selves.append(duplicate(prs, selves[-1]))

# ---- 最終モデルの中身：左に前提の箇条、右に予測ができるまでの流れの図 ---------
INK = RGBColor(0x26, 0x26, 0x26)
LINE = RGBColor(0x59, 0x59, 0x59)


def text_box(slide, x, y, w, h, lines, sz, bold_first=False, align="l"):
    """塗りなし・枠なしの文字の箱を置き、1 行ずつ段落にする。"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    txBody = box._element.find(qn("p:txBody"))
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)
    for i, line in enumerate(lines):
        p = sub(txBody, "a:p")
        sub(sub(p, "a:pPr", marL=0, indent=0, algn=align), "a:buNone")
        run(p, line, sz, bold_first and i == 0, FACE)
    return box


def flow_box(slide, x, y, w, h, lines, sz=10.5):
    """流れの図の 1 段。塗りなしの細い枠（規約：塗りのない箱だけ枠線を付ける）。"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.background()
    shape.line.color.rgb = LINE
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    txBody = shape._element.find(qn("p:txBody"))
    for p in txBody.findall(qn("a:p")):
        txBody.remove(p)
    for i, line in enumerate(lines):
        p = sub(txBody, "a:p")
        sub(sub(p, "a:pPr", marL=0, indent=0, algn="ctr"), "a:buNone")
        run(p, line, sz, i == 0, FACE)
        p.find(qn("a:r")).find(qn("a:rPr")).insert(
            0,
            etree.fromstring(
                f'<a:solidFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:srgbClr val="{INK}"/></a:solidFill>'
            ),
        )


def down_triangle(slide, cx, y):
    """段と段の間の下向きの三角（読む向きと同じ向き）。"""
    w, h = 0.22, 0.12
    tri = slide.shapes.add_shape(
        MSO_SHAPE.FLOWCHART_MERGE, Inches(cx - w / 2), Inches(y), Inches(w), Inches(h)
    )
    tri.fill.solid()
    tri.fill.fore_color.rgb = LINE
    tri.line.fill.background()
    tri.shadow.inherit = False


body = placeholder(s_model, 1)
body.left, body.top, body.width, body.height = (
    Inches(0.34),
    Inches(1.1),
    Inches(4.75),
    Inches(3.95),
)
fill_body(
    body,
    [
        (0, "目的変数：log(㎡単価) − 相場の指数（log）", False),
        (1, "test は 2025年7月〜2026年3月の取引", False),
        (1, "test の時期の指数は、直近 4 四半期の傾きで延ばした", False),
        (0, "前処理と特徴量：次のページにまとめた", False),
        (
            0,
            "行の重み付け：成約価格情報の行を 1.5 倍に重くした",
            False,
        ),
        (1, "成約価格情報は train の 45%、test の 67%", False),
        (
            0,
            "検証：時系列で 3 fold に分けた",
            False,
        ),
        (1, "各 fold は、それより前で学習し直後の 3 四半期を当てる", False),
        (1, "※ 10/19 から CV では 2 段目を前の fold だけで学習", False),
    ],
    sizes=(12.5, 11.5, 11.5),
    line=100,
)
for el in body._element.iter(qn("a:spcPts")):
    if el.get("val") == "600":
        el.set("val", "500")

text_box(
    s_model,
    0.34,
    4.3,
    4.75,
    0.85,
    [
        "注：stacking は、複数モデルの予測を別のモデルで組み合わせること",
        "注：TE（target encoding）の中身は次のページ",
        "データ：国土交通省 不動産情報ライブラリの取引価格・",
        "　 成約価格情報（東京都の中古マンション等）を講義用に加工",
    ],
    10,
)

X0, W = 5.3, 4.36
text_box(s_model, X0, 1.1, W, 0.3, ["予測ができるまでの流れ"], 11.5, bold_first=True)
steps = [
    (1.72, 0.36, ["1 段目：13 モデルを学習し、それぞれ予測する"]),
    (None, 0.96, None),  # 2 つの箱を横に並べる段
    (
        3.06,
        0.62,
        [
            "2 段目：LightGBM が土台からのずれを予測する",
            "入力は各モデルの予測と土台の差（13 個）、面積など",
            "土台の水準を変えないよう、ずれの平均は 0 にする",
        ],
    ),
    (3.74, 0.34, ["seed 3 つ（42・1・2）の予測を平均する"]),
    (
        4.26,
        0.5,
        [
            "7〜9 月の予測が 0.33% 高いと分かり、test 全期間を下げる",
            "LB probing：Public LB の変化からずれを測った（8 ページ）",
        ],
    ),
    (
        4.94,
        0.62,
        [
            "最終の予測：Public LB 0.18571、CV 0.18604 ※",
            "CV は seed 3 つの平均",
            "Private LB 0.19786（Public より 0.012 悪い）",
        ],
    ),
]
y = 1.42
for i, (_top, h, lines) in enumerate(steps):
    if lines is None:
        half = (W - 0.16) / 2
        flow_box(
            s_model,
            X0,
            y,
            half,
            h,
            [
                "土台：4 モデルの予測の平均",
                "LightGBM 2 つ・CatBoost",
                "線形回帰＋LightGBM",
            ],
            sz=10,
        )
        flow_box(
            s_model,
            X0 + half + 0.16,
            y,
            half,
            h,
            [
                "土台に入れない 9 モデル",
                "一部の行の LightGBM 5 つ",
                "（築 30 年超・23 区外など）",
                "ヘドニック回帰 1 つ",
                "TE 付き LightGBM 3 つ",
            ],
            sz=10,
        )
    else:
        flow_box(s_model, X0, y, W, h, lines, sz=10.5 if len(lines) == 1 else 10)
    y += h
    if i < len(steps) - 1:
        down_triangle(s_model, X0 + W / 2, y + 0.01)
        y += 0.09

# ---- 前処理と特徴量：種類ごとに、中身と理由を表にする -------------------------
feat_body = placeholder(s_feat, 1)
feat_body._element.getparent().remove(feat_body._element)
frame_el = copy.deepcopy(next(sh for sh in s_score.shapes if sh.has_table)._element)
s_feat.shapes._spTree.append(frame_el)
feat_frame = next(sh for sh in s_feat.shapes if sh.has_table)
feat_tbl = frame_el.find(".//" + qn("a:tbl"))
# テンプレートの表は 5 列なので、3 列に減らす
for gc in feat_tbl.find(qn("a:tblGrid")).findall(qn("a:gridCol"))[3:]:
    gc.getparent().remove(gc)
for tr in feat_tbl.findall(qn("a:tr")):
    for tc in tr.findall(qn("a:tc"))[3:]:
        tr.remove(tc)
fill_table(
    feat_frame,
    [1.35, 4.85, 3.12],
    [
        [
            "前処理",
            "面積「2,000㎡以上」→ 2000、建築年「戦前」→ 1945、\n駅距離「30分～60分」などの範囲 → 中央値",
            "文字列のままでは大小を比べられない",
        ],
        [
            "前処理",
            "取引時期 → 2019 年 1〜3 月からの経過四半期。\n欠損は NaN のまま",
            "LightGBM は欠損をそのまま扱える",
        ],
        [
            "元の列",
            "築年数・面積・駅距離・建ぺい率・容積率と、\n市区町村・地区・駅・間取り・構造などのカテゴリ",
            "カテゴリは train と test で同じ番号",
        ],
        [
            "位置",
            "地区の緯度・経度と、主要 7 駅（東京・新宿・渋谷・\n池袋・品川・六本木・虎ノ門）までの距離",
            "外部データ（下の注）",
        ],
        [
            "地区・駅の集計",
            "件数と、平均の築年数・面積・駅距離との差",
            "価格を使わない",
        ],
        [
            "建物の集計",
            "同じ建物とみなした行の件数・面積の幅・間取りの種類数",
            "価格を使わない",
        ],
        ["間取り", "部屋数に換算した、一部屋あたりの面積", "価格を使わない"],
        [
            "TE",
            "地区・駅・建物ごとの価格の平均。\n1 つ上の単位の平均へ、件数に応じて寄せる",
            "価格を使うので、fold ごとに\n学習データだけで作る",
        ],
    ],
    y=1.1,
    sz=11.5,
    head_sz=13.5,
    row_h=0.3,
    head_h=0.35,
)
for tc, text in zip(
    feat_tbl.findall(qn("a:tr"))[0].findall(qn("a:tc")),
    ["種類", "中身", "理由・注意"],
    strict=True,
):
    set_cell(tc, text, 13.5, True)
text_box(
    s_feat,
    0.34,
    4.95,
    9.32,
    0.45,
    [
        "注：外部データは、国土交通省の位置参照情報（2024 年版、基準日 2024-07-26）と国土数値情報の駅別乗降客数（2024 年度版）。",
        "　 どちらも test の期間（2025 年 7 月〜）より前の時点のデータ",
    ],
    10,
)

# ---- そのモデルに至った経緯：分かれ目を上から順に並べた流れの図 ---------------
story_body = placeholder(s_story, 1)
story_body._element.getparent().remove(story_body._element)
text_box(
    s_story,
    0.34,
    1.05,
    9.32,
    0.25,
    ["番号は提出した順（6 ページのグラフと同じ）"],
    10.5,
)
STORY = [
    (
        "LightGBM のベースラインで、カテゴリの扱いと学習率を調整し、損失を Huber 損失に変えた",
        "提出 1 → 3：Public LB 0.20542 → 0.19914",
    ),
    (
        "目的変数を総額から㎡単価に変え、相場の指数（log）を引いた：決定木は学習期間より先の値上がりを表せないため",
        "提出 3 → 5：Public LB 0.19914 → 0.19709",
    ),
    (
        "主要駅までの距離と、地区・駅の平均（築年数・面積・駅距離）との差を足し、4 モデルを平均した",
        "提出 5 → 8：Public LB 0.19709 → 0.19251",
    ),
    (
        "土台の 4 モデルを含む 6 モデルで stacking を組み、建物ごとの集計を足した",
        "提出 8 → 9：Public LB 0.19251 → 0.18770",
    ),
    (
        "TE 付きのモデル・2 段目の入力・一部屋あたりの面積・seed 平均を足した：1 つ足すごとの CV の改善は 0.0001〜0.0009",
        "提出 9 → 10：Public LB 0.18770 → 0.18595",
    ),
    (
        "成約価格情報の重みと TE の平滑化を足し、LB probing で予測のずれを補正した",
        "提出 10 → 12：Public LB 0.18595 → 0.18571（提出 11 は補正量を測るための提出）",
    ),
]
y = 1.38
for i, (head, evidence) in enumerate(STORY):
    flow_box(s_story, 0.34, y, 9.32, 0.45, [head, evidence], sz=11)
    y += 0.45
    if i < len(STORY) - 1:
        down_triangle(s_story, 0.34 + 9.32 / 2, y + 0.01)
        y += 0.13
text_box(
    s_story,
    0.34,
    y + 0.08,
    9.32,
    0.5,
    [
        "試してやめたもの：isotonic 回帰による予測の補正（提出 4：CV は 0.20004 → 0.19727 に良くなったが、Public LB は 0.20059 に悪化）",
        "　 あとで分かった Private LB では 0.21876 → 0.21251 と良くなっていた（Public は 7〜9 月、Private は 10〜3 月の行で採点）",
    ],
    10.5,
)

# ---- 各自の貢献・再現性 -------------------------------------------------------
fill_body(
    placeholder(s_team, 1),
    [
        (0, "各自の取り組みと、最終モデルへの貢献", True),
        (
            1,
            "南條：複数モデルの単純平均と「線形回帰＋LightGBM」を作り、誤差の分析を進めた",
            False,
        ),
        (
            1,
            "大久保：相場の指数を引く目的変数、成約価格情報の重み、TE の階層的な平滑化を入れた",
            False,
        ),
        (
            1,
            "藤瀬：実験の仕組み、時系列 CV、地区・駅の集計、stacking、LB probing を AI と作った",
            False,
        ),
        (
            1,
            "小林：㎡単価の目的変数を入れ、間取りの分解・地価公示・誤差の分析を試した",
            False,
        ),
        (0, "再現性", True),
        (
            1,
            "seed は 42・1・2 に固定した",
            False,
        ),
        (
            1,
            "scripts/ の 2 本（seed 平均・補正）で、提出ファイルまで再現できる",
            False,
        ),
        (
            1,
            "提出したファイルは probe/ に置いた",
            False,
        ),
        (
            1,
            "同じ PC では提出ファイルが完全に一致し、別の PC では CV が 0.0001 ほど変わった",
            False,
        ),
    ],
    sizes=(15, 14, 14),
    line=135,
)

# ---- スコアの推移（提出したもの）：折れ線グラフ ------------------------------
# 提出の中身・手元の CV・Public LB。CV が None の提出は、CV を測っていないもの
# 提出ごとの Private LB（10/22 公開）。SUBMISSIONS と同じ順
PRIVATE = [
    0.22596,
    0.22335,
    0.21876,
    0.21251,
    0.20925,
    0.20844,
    0.20670,
    0.20491,
    0.19959,
    0.19828,
    0.19828,
    0.19786,
]
SUBMISSIONS = [
    ("LightGBM のベースライン", 0.20674, 0.20542),
    ("カテゴリの扱いと学習率の調整", 0.20360, 0.20222),
    ("損失関数を Huber 損失に変更", 0.20004, 0.19914),
    ("isotonic 回帰で予測を補正（やめた）", 0.19727, 0.20059),
    ("目的変数を㎡単価にし、相場の指数を引く", 0.19664, 0.19709),
    ("主要駅までの距離、地区の緯度経度", 0.19553, 0.19561),
    ("地区・駅の平均（築年数など）との差", 0.19358, 0.19321),
    ("4 モデルの単純平均", 0.19174, 0.19251),
    ("6 モデルの stacking、建物ごとの集計など", 0.18765, 0.18770),
    ("13 モデル、TE、seed 平均", 0.18531, 0.18595),
    ("LB probing：10 の 7〜9 月だけ 1% 下げた", None, 0.18604),
    ("最終：10 に 2 つの改良と補正 ※", 0.18604, 0.18571),
]
fill_body(
    placeholder(s_score, 1),
    [
        (
            0,
            "4（isotonic）は Public だけ向きが逆、12 は測り方が違うので除くと、CV と Public LB は同じ向きに動いた",
            True,
        ),
        (
            0,
            "※ 12 の CV（白抜き）は 10/19 から 2 段目を前の fold だけで学習した値で、0.0006〜0.0008 大きく出る",
            False,
        ),
        (
            0,
            "10 を ※ の測り方で測ると 0.18614（seed 42）",
            False,
        ),
        (
            0,
            "Private LB（破線）は一度も上がらず、12 が 0.19786：5 以降は Public より 0.012 ほど大きい",
            False,
        ),
    ],
    sizes=(11, 10.5, 10),
)
desc = placeholder(s_score, 1)
desc.top, desc.height = Inches(0.85), Inches(1.15)
for el in desc._element.iter(qn("a:spcPts")):
    el.set("val", "0")
table = next(sh for sh in s_score.shapes if sh.has_table)
table._element.getparent().remove(table._element)

data = CategoryChartData()
data.categories = [str(i) for i in range(1, len(SUBMISSIONS) + 1)]
data.add_series("CV", [cv for _, cv, _ in SUBMISSIONS], number_format="0.000")
data.add_series("Public LB", [lb for _, _, lb in SUBMISSIONS], number_format="0.000")
# Private LB は最終の提出（12）の値だけ分かっている
data.add_series(
    "Private LB",
    PRIVATE,
    number_format="0.000",
)
chart = s_score.shapes.add_chart(
    XL_CHART_TYPE.LINE_MARKERS,
    Inches(0.34),
    Inches(2.05),
    Inches(6.0),
    Inches(3.35),
    data,
).chart
chart.font.size = Pt(10)
chart.font.name = FACE
chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.TOP
chart.legend.include_in_layout = False
value_axis = chart.value_axis
value_axis.minimum_scale, value_axis.maximum_scale = 0.184, 0.228
value_axis.major_unit = 0.008
value_axis.tick_labels.number_format = "0.000"
value_axis.tick_labels.number_format_is_linked = False
value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)
value_axis.has_title = True
value_axis.axis_title.text_frame.text = "RMSLE（小さいほど良い）"
value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
chart.category_axis.has_title = True
chart.category_axis.axis_title.text_frame.text = "提出の順（右の番号）"
chart.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
chart.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
# 2 本の線は、主役の Public LB を濃い色、手元の CV をグレーにして見分ける
for series, color in zip(
    list(chart.plots[0].series)[:2],
    (RGBColor(0x99, 0x99, 0x99), RGBColor(0x1F, 0x1F, 0x1F)),
    strict=True,
):
    series.smooth = False
    series.format.line.color.rgb = color
    series.format.line.width = Pt(2)
    series.marker.style = XL_MARKER_STYLE.CIRCLE
    series.marker.size = 6
    series.marker.format.fill.solid()
    series.marker.format.fill.fore_color.rgb = color
    series.marker.format.line.color.rgb = color

# 12 の CV は測り方が違い、前の点と比べられないので白抜きにする
last_cv = chart.plots[0].series[0].points[len(SUBMISSIONS) - 1].marker
last_cv.style = XL_MARKER_STYLE.CIRCLE
last_cv.size = 7
last_cv.format.fill.solid()
last_cv.format.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
last_cv.format.line.color.rgb = RGBColor(0x99, 0x99, 0x99)

# 9〜12 は点が近くて読み分けにくいので、Public LB の 10 と 12 だけ値を添える
for i in (9, 11):
    label = chart.plots[0].series[1].points[i].data_label
    label.show_value = True
    dLbl = label._get_or_add_dLbl()
    dLbl.find(qn("c:idx")).addnext(
        etree.fromstring(
            '<c:numFmt xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" formatCode="0.00000" sourceLinked="0"/>'
        )
    )
    label.position = XL_LABEL_POSITION.ABOVE
    label.font.size = Pt(9)
last_cv.format.line.width = Pt(1.5)

private = chart.plots[0].series[2]
private.smooth = False
private.format.line.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
private.format.line.width = Pt(1.5)
private.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
private.marker.style = XL_MARKER_STYLE.SQUARE
private.marker.size = 6
private.marker.format.fill.solid()
private.marker.format.fill.fore_color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
private.marker.format.line.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
for i in (9, 11):
    label = private.points[i].data_label
    label.show_value = True
    label._get_or_add_dLbl().find(qn("c:idx")).addnext(
        etree.fromstring(
            '<c:numFmt xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" formatCode="0.00000" sourceLinked="0"/>'
        )
    )
    label.position = XL_LABEL_POSITION.ABOVE
    label.font.size = Pt(9)

box = s_score.shapes.add_textbox(Inches(6.5), Inches(2.05), Inches(3.2), Inches(3.35))
txBody = box._element.find(qn("p:txBody"))
for p in txBody.findall(qn("a:p")):
    txBody.remove(p)
p = sub(txBody, "a:p")
sub(sub(p, "a:pPr", marL=0, indent=0), "a:buNone")
run(p, "提出の中身", 11.5, True, FACE)
for name, _, _ in SUBMISSIONS:
    p = sub(txBody, "a:p")
    pPr = sub(p, "a:pPr", marL=274320, indent=-274320)
    sub(sub(pPr, "a:spcBef"), "a:spcPts", val=100)
    sub(pPr, "a:buFont", typeface="Arial")
    sub(pPr, "a:buAutoNum", type="arabicPeriod")
    run(p, name, 10.5, False, FACE)

# ---- スコアの推移（CV だけで判断したもの）----------------------------------
fill_body(
    placeholder(s_score2, 1),
    [
        (
            0,
            "単独では提出していない実験（Public・Private の値はない）：スコアは「変える前の CV → 変えた後の CV」",
            False,
        ),
        (
            0,
            "※ は 10/19 から 2 段目を前の fold だけで学習した CV：前の測り方より 0.0006〜0.0008 大きく出る",
            False,
        ),
        (
            0,
            "両方入れた最終：seed 42 どうし（藤瀬の PC）では 0.18614 → 0.18619 ※ で、seed による揺れ（0.0003 ほど）の範囲",
            False,
        ),
        (
            0,
            "採用した理由：土台だけ（2 段目なし）の CV が seed 3 つすべてで良くなった（平均 0.18799 → 0.18783）",
            False,
        ),
    ],
    sizes=(11, 10.5, 10),
)
desc = placeholder(s_score2, 1)
desc.top, desc.height = Inches(0.82), Inches(1.15)
for el in desc._element.iter(qn("a:spcPts")):
    el.set("val", "0")
fill_table(
    next(sh for sh in s_score2.shapes if sh.has_table),
    [2.85, 2.05, 0.82, 0.9, 2.7],
    [
        ["㎡単価を目的変数にする", "0.20004 → 0.19845", "—", "—", "採用"],
        [
            "間取りを部屋数などに分解する",
            "0.19727 → 0.19736",
            "—",
            "—",
            "不採用、isotonic 込みで測定",
        ],
        ["建物ごとの集計", "0.18944 → 0.18872", "—", "—", "採用"],
        [
            "2 段目の入力に 5 モデルを足す",
            "0.18765 → 0.18680",
            "—",
            "—",
            "採用、TE 付き 1 つを含む",
        ],
        ["一部屋あたりの面積", "0.18599 → 0.18548", "—", "—", "採用"],
        [
            "TE 付きのモデルを 2 つ足す",
            "0.18628 → 0.18614 ※",
            "—",
            "—",
            "採用、計 13 モデル（提出 10）",
        ],
        [
            "成約価格情報の重みを 1.5 倍",
            "0.18622 → 0.18606 ※",
            "—",
            "—",
            "採用、大久保の PC で比較",
        ],
        [
            "TE を階層的に平滑化",
            "0.18622 → 0.18611 ※",
            "—",
            "—",
            "採用、大久保の PC で比較",
        ],
        [
            "相場から外れた取引を外す",
            "0.0002〜0.0018 悪化",
            "—",
            "—",
            "不採用、CV が悪化",
        ],
        [
            "近くの地価公示の価格",
            "0.0003〜0.0010 悪化",
            "—",
            "—",
            "不採用、LightGBM だけで比較",
        ],
        [
            "2 段目の列 9 通り・設定 10 通り",
            "差 0.0002 以内",
            "—",
            "—",
            "不採用、どれも効果なし",
        ],
    ],
    y=2.12,
    sz=11.5,
    head_sz=13.5,
    row_h=0.25,
    head_h=0.5,
)

# ---- LB probing：Public LB の変化から予測のずれを測る -------------------------
table = next(sh for sh in s_probe.shapes if sh.has_table)
table._element.getparent().remove(table._element)
PROBE = [
    ["提出 10（10/20）", "Public LB 0.18595（L₀）"],
    [
        "提出 11：10 の 7〜9 月の予測だけを 1% 下げて提出",
        "Public LB 0.18604（L₁）：少し悪くなった",
    ],
    [
        "2 つの LB から、7〜9 月の予測のずれ m を逆算",
        "L₁² ＝ L₀² − 2cm ＋ c²（c ＝ 1%）から m ＝ ＋0.33%",
    ],
    [
        "提出 12：test 全期間の予測を 0.33% 下げて提出",
        "Public LB 0.18571（成約の重み・TE の平滑化も追加）",
    ],
]
y = 1.2
for i, lines in enumerate(PROBE):
    flow_box(s_probe, 0.34, y, 4.6, 0.62, lines, sz=11)
    y += 0.62
    if i < len(PROBE) - 1:
        down_triangle(s_probe, 0.34 + 4.6 / 2, y + 0.03)
        y += 0.2
probe_body = placeholder(s_probe, 1)
probe_body.left, probe_body.top, probe_body.width, probe_body.height = (
    Inches(5.2),
    Inches(1.15),
    Inches(4.46),
    Inches(3.85),
)
fill_body(
    probe_body,
    [
        (0, "置いた仮定", True),
        (1, "Public LB は test の 7〜9 月の行だけで採点される", False),
        (2, "Public は test の約 37%、7〜9 月の行は 37.2%", False),
        (1, "10〜3 月の予測も、7〜9 月と同じだけ高い", False),
        (1, "結果：11 と 10 の Private LB が同じ 0.19828", False),
        (2, "仮定どおり、Private に 7〜9 月はない", False),
        (0, "式の読み方", True),
        (1, "下げた量 c が 2m より大きいと LB は悪くなる", False),
        (2, "提出 11 の悪化の幅から m を求められる", False),
        (0, "注意", True),
        (1, "Private LB は 0.19786（Public より 0.012 悪い）", False),
        (2, "10 の 0.19828 より 0.0004 良い", False),
        (2, "補正で LB が動くのは最大 0.0036", False),
        (2, "差の大半は補正以外による", False),
    ],
    sizes=(13, 12, 11.5),
    line=110,
)

# ---- AI の使い方 -------------------------------------------------------------
fill_body(
    placeholder(s_ai, 1),
    [
        (0, "使った AI ツールと任せたこと", True),
        (
            1,
            "Claude Code：コード・PR の本文・レビューを任せ、夜間は複数のエージェントに実験を順に回させた",
            False,
        ),
        (
            1,
            "Fable（Claude のモデル。藤瀬が使用）：残った誤差の分析と、LB probing のスクリプト作りを任せた",
            False,
        ),
        (1, "Gemini（大久保が使用）：AI が夜間にまとめた PR を検証した", False),
        (0, "AI が誤っていたこと", True),
        (
            1,
            "AI は、安い価格帯かどうかを当てる分類器を足すと CV が良くなると報告した",
            False,
        ),
        (2, "2 段目の入力に正解の情報が漏れていて、入力を直すと改善が消えた", False),
        (
            1,
            "AI が、別のセッションで書き換えられた PR に気づかず、古い手順の提出を案内した",
            False,
        ),
        (2, "人が PR の本文を読み直して気づいた", False),
        (0, "AI と作り、人が CV を見て採否を決めたもの", True),
        (1, "採用：stacking、TE 付きのモデル、seed 平均、LB probing", False),
        (
            1,
            "不採用：相場から外れた取引を外す案（3 つのしきい値すべてで CV が悪化）",
            False,
        ),
        (0, "AI への指示ファイルに書いたルール", True),
        (
            1,
            "テスト期間の取引データを見ない、価格を使う特徴量は fold ごとに作る、重い計算は同時に 1 本だけ実行する",
            False,
        ),
    ],
    sizes=(14, 12.5, 12),
    line=105,
)

# ---- 良かった点・悪かった点（全員で振り返って書き直す）----------------------
fill_body(
    placeholder(s_good, 1),
    [
        (0, "話し合いの材料：全員で振り返ってから書き直す", True),
        (
            1,
            "PR と Issue で実験を記録し、CV と Public LB の対応を最後まで残せた",
            False,
        ),
        (
            1,
            "4 と 12 を除き CV と Public LB は向きがそろい、多くの採否を CV で決められた",
            False,
        ),
        (1, "CV は Private LB とも、1〜10 のすべてで同じ向きに動いた", False),
        (1, "効かなかった実験も、不採用として理由と一緒に記録した", False),
        (1, "AI に夜間の実験を任せ、朝に人が結果を見て採否を決めた", False),
    ],
    sizes=(14, 13.5, 13),
    line=115,
)
fill_body(
    placeholder(s_bad, 1),
    [
        (0, "話し合いの材料：全員で振り返ってから書き直す", True),
        (1, "重い計算が 1 台の PC に集まり、試せる量が限られた", False),
        (
            1,
            "後半の改善を支えた stacking に移ったのが、締め切りの 3 日前だった",
            False,
        ),
        (1, "PC によって CV が 0.0001 ほど違い、小さな改善を比べにくかった", False),
        (
            1,
            "PR を前の PR の上に重ねて作ったため、どのブランチから始めるかが分かりにくかった",
            False,
        ),
        (
            1,
            "Public LB が test のどの部分で採点されるかを、終盤まで確かめなかった",
            False,
        ),
        (
            2,
            "isotonic で CV と Public が食い違ったとき、Public は 1 四半期先だけを見ていると割り引けなかった",
            False,
        ),
    ],
    sizes=(14, 13.5, 13),
    line=115,
)

# ---- タイトル：結論を 1 文にし、テンプレートの見出しは左上に「項目：」として残す ----
TITLES = [
    (s_model, "13 モデルの stacking で㎡単価を予測する最終モデル"),
    (s_feat, "前処理と、価格を使わない集計を中心にした特徴量"),
    (s_story, "目的変数の見直しと stacking による改善"),
    (s_team, "4 人の分担と、提出ファイルを再現できる仕組み"),
    (s_score, "ほぼ同じ向きに動いた CV と Public LB"),
    (s_score2, "後半ほど小さくなった、提出しなかった実験の改善幅"),
    (
        s_probe,
        "LB probing による 7〜9 月の予測のずれの測定と補正",
    ),
    (s_ai, "実装と夜間の実験は AI、採否の判断は人という分担"),
]
for slide, title in TITLES:
    heading = placeholder(slide, 0).text_frame.text
    set_title(slide, title)
    label = text_box(slide, 0.34, 0.1, 9.32, 0.28, [f"項目：{heading}"], 9.5)
    for r in label._element.iter(qn("a:rPr")):
        r.insert(
            0,
            etree.fromstring(
                '<a:solidFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:srgbClr val="7F7F7F"/></a:solidFill>'
            ),
        )

# ---- 各自の学んだこと・反省点（本人が書く）-------------------------------------
for slide, name in zip(selves, MEMBERS, strict=True):
    set_title(slide, f"{name} 演習を通して学んだこと、反省点")
    fill_body(
        placeholder(slide, 1),
        [
            (0, "自分の担当と、最終モデルへの貢献", True),
            (1, "（本人が書く）", False),
            (0, "演習を通して学んだこと", True),
            (1, "（本人が書く）", False),
            (0, "反省点", True),
            (1, "（本人が書く）", False),
        ],
        sizes=(14, 13.5, 13),
        line=115,
    )

prs.save(sys.argv[2])
