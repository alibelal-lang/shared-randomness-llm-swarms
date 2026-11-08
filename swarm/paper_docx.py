"""Fill the Apart submission template (.docx) from the paper's Claude Doc.

  python -m swarm.paper_docx <template.docx> <doc.xml> <out.docx>

doc.xml is the Claude Docs prose view of the paper (report/paper_doc.xml). The template's title block, author
table, abstract box and footnote are kept; its guidance text and info box are removed; the body is rebuilt in the
template's own fonts and heading formats.
"""

import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile
from xml.sax.saxutils import escape

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
FONT = '<w:rFonts w:ascii="Old Standard TT" w:cs="Old Standard TT" w:eastAsia="Old Standard TT" w:hAnsi="Old Standard TT"/>'
MONO = '<w:rFonts w:ascii="Courier New" w:cs="Courier New" w:eastAsia="Courier New" w:hAnsi="Courier New"/>'
FIGS = {"blob/b3de9e25-2bfa": ("report/figures/fig1_size_and_coupling.png", 5.8),
        "blob/20d553df-098f": ("report/figures/fig2_mechanism.png", 3.7)}
TITLE = "Shared randomness makes large LLM swarms coordinate like small ones"


class Body:
    def __init__(self):
        self.out, self.rels, self.media = [], [], []
        self.next_rid, self.next_pic = 100, 1
        self.ordered_lists = 0

    def rid(self):
        self.next_rid += 1
        return f"rId{self.next_rid}"

    # ---- runs
    def run(self, text, b=False, i=False, sz=None, mono=False, color=None, u=False):
        rpr = (MONO if mono else FONT) + ("<w:b/><w:bCs/>" if b else "") + ("<w:i/><w:iCs/>" if i else "")
        rpr += f'<w:color w:val="{color}"/>' if color else ""
        rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else ""
        rpr += '<w:u w:val="single"/>' if u else ""  # schema order: color, sz, u
        return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

    def runs(self, el, b=False, i=False, link=None, sz=None, force_i=None):
        """Walk a Claude Docs <text>/<bold>/<italic>/<link> subtree into Word runs."""
        xml = []
        ii = i if force_i is None else force_i

        def emit(t):
            if not t:
                return
            r = self.run(t, b=b, i=ii, sz=sz, color="1155CC" if link else None, u=bool(link))
            if link:
                rid = self.rid()
                self.rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/'
                                 f'2006/relationships/hyperlink" Target="{escape(link)}" TargetMode="External"/>')
                r = f'<w:hyperlink r:id="{rid}" w:history="1">{r}</w:hyperlink>'
            xml.append(r)
        emit(el.text)
        for c in el:
            nb, ni, nl = b, i, link
            if c.tag == "bold":
                nb = True
            elif c.tag == "italic":
                ni = True
            elif c.tag == "link":
                nl = c.get("href")
            if c.tag in ("text", "bold", "italic", "link"):
                xml.append(self.runs(c, nb, ni, nl, sz, force_i))
            emit(c.tail)
        return "".join(xml)

    def para_runs(self, p, **kw):
        return "".join(self.runs(t, **kw) for t in p if t.tag == "text")

    # ---- blocks
    def p(self, inner, ppr=""):
        self.out.append(f"<w:p><w:pPr>{ppr}</w:pPr>{inner}</w:p>")

    def heading(self, text, level):
        if level == 2:
            ppr = ('<w:pStyle w:val="Heading2"/><w:keepNext/><w:spacing w:before="280" w:after="80" w:line="276" '
                   'w:lineRule="auto"/>')
            self.p(self.run(text, b=True, sz=28, color="434343"), ppr)
        else:
            ppr = ('<w:pStyle w:val="Heading3"/><w:keepNext/><w:spacing w:before="160" w:after="40" w:line="276" '
                   'w:lineRule="auto"/>')
            self.p(self.run(text, b=True, sz=24, color="666666"), ppr)

    def body_para(self, p):
        self.p(self.para_runs(p), '<w:spacing w:after="80" w:line="276" w:lineRule="auto"/>')

    def caption(self, p):
        self.p(self.para_runs(p, sz=19, force_i=True),
               '<w:spacing w:before="40" w:after="200" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/>')

    def caption_above(self, p):  # table captions sit above their table
        self.p(self.para_runs(p, sz=19, force_i=True),
               '<w:keepNext/><w:spacing w:before="40" w:after="100" w:line="240" w:lineRule="auto"/>'
               '<w:jc w:val="both"/>')

    def listing(self, lst):
        if lst.get("kind") == "ordered":
            self.ordered_lists += 1
            num = 5 if self.ordered_lists == 1 else 1
        else:
            num = 3
        for li in lst.findall("listItem"):
            for p in li.findall("paragraph"):
                self.p(self.para_runs(p), f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num}"/></w:numPr>'
                       '<w:spacing w:after="60" w:line="276" w:lineRule="auto"/><w:ind w:left="720" w:hanging="360"/>')

    def image(self, media):
        path, width_in = FIGS[media.get("blob")]
        data = open(path, "rb").read()
        wpx, hpx = struct.unpack(">II", data[16:24])
        cx = int(width_in * 914400)
        cy = int(cx * hpx / wpx)
        rid, n = self.rid(), self.next_pic
        self.next_pic += 1
        name = f"image{n}.png"
        self.media.append((name, path))
        self.rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                         f'relationships/image" Target="media/{name}"/>')
        alt = escape(media.get("alt") or "", {'"': "&quot;"})
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{n}" name="Figure {n}" descr="{alt}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks '
            f'noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats'
            f'.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{n}" name="{name}"/><pic:cNvPicPr/>'
            f'</pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect">'
            f'<a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        self.p(drawing, '<w:keepNext/><w:spacing w:before="120" w:after="0" w:line="240" w:lineRule="auto"/>'
               '<w:jc w:val="center"/>')

    def code(self, cb):
        src = "".join(cb.itertext())
        if cb.get("language") == "latex" and r"\frac{1}{4N}" in src:
            def r(t, plain=False):
                sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
                return f'<m:r>{sty}<w:rPr>{FONT}</w:rPr><m:t xml:space="preserve">{escape(t)}</m:t></m:r>'
            p_i = f"<m:sSub><m:e>{r('p')}</m:e><m:sub>{r('i')}</m:sub></m:sSub>"
            omml = (f'<m:oMathPara><m:oMath>{r("Var", True)}{r("(")}{r("x")}{r(")")}{r(" = ")}'
                    f'<m:f><m:num>{r("1")}</m:num><m:den>{r("4N")}</m:den></m:f>'
                    f'<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr><m:sub>{r("i=1")}</m:sub>'
                    f'<m:sup>{r("N")}</m:sup><m:e>{p_i}{r("(1 − ")}{p_i}{r(")")}</m:e></m:nary>'
                    f'</m:oMath></m:oMathPara>')
            self.p(omml, '<w:spacing w:before="60" w:after="120"/><w:jc w:val="center"/>')
            return
        lines = src.split("\n")
        for k, line in enumerate(lines):
            after = "120" if k == len(lines) - 1 else "0"
            self.p(self.run(line, mono=True, sz=16),
                   f'<w:shd w:val="clear" w:color="auto" w:fill="F3F3F3"/><w:spacing w:before="0" w:after="{after}" '
                   'w:line="240" w:lineRule="auto"/><w:ind w:left="144" w:right="144"/>')

    def table(self, tb):
        rows = tb.findall("row")
        ncol = len(rows[0].findall("cell"))
        widths = [1500, 3000, 900, 1860, 1000, 1100] if ncol == 6 else [9360 // ncol] * (ncol - 1) + [
            9360 - (9360 // ncol) * (ncol - 1)]
        grid = "".join(f'<w:gridCol w:w="{x}"/>' for x in widths)
        border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        borders = "".join(border.format(s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        trs = []
        for ri, row in enumerate(rows):
            tcs = []
            for ci, cell in enumerate(row.findall("cell")):
                hdr = cell.get("header") == "true"
                shd = '<w:shd w:val="clear" w:color="auto" w:fill="F0EFEC"/>' if hdr else ""
                ps = "".join(
                    f'<w:p><w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/></w:pPr>'
                    f'{self.para_runs(p, b=hdr, sz=17)}</w:p>' for p in cell.findall("paragraph"))
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shd}</w:tcPr>{ps}</w:tc>')
            trpr = "<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>" if ri == 0 else "<w:trPr><w:cantSplit/></w:trPr>"
            trs.append(f"<w:tr>{trpr}{''.join(tcs)}</w:tr>")
        self.out.append(
            f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders>'
            f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="72" w:type="dxa"/><w:right w:w="72" '
            f'w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
        self.p("", '<w:spacing w:after="120"/>')


def build_body(doc):
    """Return (abstract paragraph element, Body) from the Claude Docs prose root."""
    b = Body()
    kids = list(doc)
    abstract, skip_next, i = None, False, 0
    while i < len(kids):
        el = kids[i]
        i += 1
        if el.tag == "paragraph":
            h = el.get("heading")
            text = "".join(el.itertext()).strip()
            if h == "1" or el.find("date") is not None:
                continue
            if h == "2" and text == "Abstract":
                abstract = kids[i]
                i += 1
                continue
            if h in ("2", "3"):
                b.heading(text, int(h))
                continue
            media = el.find("media")
            if media is not None:
                b.image(media)
                continue
            if not text:
                continue
            all_italic = all((t.find("italic") is not None and not (t.text or "").strip()) for t in el.findall("text"))
            if all_italic and text.startswith("Figure "):
                b.caption(el)
            elif all_italic and text.startswith("Table "):
                b.caption_above(el)
            else:
                b.body_para(el)
        elif el.tag == "list":
            b.listing(el)
        elif el.tag == "codeBlock":
            b.code(el)
        elif el.tag == "table":
            b.table(el)
    return abstract, b


def main(template, doc_xml, out):
    doc = etree.fromstring(open(doc_xml, "rb").read())
    abstract, body = build_body(doc)
    work = tempfile.mkdtemp()
    with zipfile.ZipFile(template) as z:
        z.extractall(work)
    dpath = os.path.join(work, "word", "document.xml")
    tree = etree.parse(dpath)
    root = tree.getroot()
    wbody = root.find("w:body", NS)

    def ptext(e):
        return "".join(e.itertext())

    # title
    for t in root.iter(f"{{{W}}}t"):
        if t.text == "PROJECT TITLE":
            t.text = TITLE
    # authors: the template's 3 x 2 grid of narrow cells becomes one centred placeholder line
    t1 = next(t for t in root.iter(f"{{{W}}}t") if t.text == "Author name 1")
    grid = next(t1.iterancestors(f"{{{W}}}tbl"))
    author = etree.fromstring(
        f'<w:p xmlns:w="{W}"><w:pPr><w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/>'
        f'<w:jc w:val="center"/></w:pPr>{body.run("[Author name], [Affiliation]")}</w:p>')
    grid.addprevious(author)
    grid.getparent().remove(grid)
    # abstract text
    for p in root.iter(f"{{{W}}}p"):
        if ptext(p).startswith("Summarize your project"):
            for r in p.findall("w:r", NS):
                p.remove(r)
            frag = body.para_runs(abstract)
            wrap = etree.fromstring(f'<x xmlns:w="{W}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/'
                                    f'2006/relationships">{frag}</x>')
            for c in wrap:
                p.append(c)
            break
    # drop the info box and everything from "1. Introduction" on (keep sectPr)
    kids = list(wbody)
    intro = next(k for k in kids if k.tag == f"{{{W}}}p" and ptext(k).strip() == "1. Introduction")
    for k in kids:
        if k.tag == f"{{{W}}}tbl" and "How to use this template" in ptext(k):
            wbody.remove(k)
    kids = list(wbody)
    start = kids.index(intro)
    # remove empty spacer paragraphs right before the intro, keep one
    j = start - 1
    while j > 0 and kids[j].tag == f"{{{W}}}p" and not ptext(kids[j]).strip() and kids[j - 1].tag == f"{{{W}}}p" \
            and not ptext(kids[j - 1]).strip():
        wbody.remove(kids[j])
        j -= 1
    sect = wbody.find("w:sectPr", NS)
    pgmar = sect.find("w:pgMar", NS)
    if pgmar is not None and pgmar.get(f"{{{W}}}gutter") is None:
        pgmar.set(f"{{{W}}}gutter", "0")
    for k in kids[start:]:
        if k is not sect and k.getparent() is wbody:
            wbody.remove(k)
    nsdecl = " ".join(f'xmlns:{p}="{u}"' for p, u in root.nsmap.items() if p)
    frag = etree.fromstring(f"<x {nsdecl}>{''.join(body.out)}</x>")
    for c in frag:
        sect.addprevious(c)
    tree.write(dpath, xml_declaration=True, encoding="UTF-8", standalone=True)

    # relationships, media, content types
    rpath = os.path.join(work, "word", "_rels", "document.xml.rels")
    rels = open(rpath, encoding="utf-8").read().replace("</Relationships>", "".join(body.rels) + "</Relationships>")
    open(rpath, "w", encoding="utf-8").write(rels)
    os.makedirs(os.path.join(work, "word", "media"), exist_ok=True)
    for name, src in body.media:
        shutil.copy(src, os.path.join(work, "word", "media", name))
    cpath = os.path.join(work, "[Content_Types].xml")
    ct = open(cpath, encoding="utf-8").read()
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Default ", '<Default ContentType="image/png" Extension="png"/><Default ', 1)
    open(cpath, "w", encoding="utf-8").write(ct)

    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, files in os.walk(work):
            for f in files:
                full = os.path.join(dirpath, f)
                arc = os.path.relpath(full, work)
                z.write(full, arc)
    shutil.rmtree(work)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
