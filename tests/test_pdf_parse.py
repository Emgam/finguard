"""PDF parsing on a generated regulatory-style PDF (skipped when reportlab is not installed)."""
import tempfile
from pathlib import Path
from unittest import SkipTest

from finguard.rag.chunk import chunk_blocks
from finguard.rag.parse import parse_pdf

TOPICS = ["scope and definitions", "reporting frequency", "data breakdowns", "card payments", "governance"]


def _body(topic: str) -> str:
    return (f"This part concerns {topic}. Payment service providers should report fraudulent payment transactions to the "
            f"competent authority, with details on {topic}. The report covers card-not-present transactions, credit transfers "
            "and direct debits. ") * 4


def _make_pdf(path: Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.platypus import Table, TableStyle
    except ImportError as exc:
        raise SkipTest("reportlab not installed") from exc
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    pages = 5
    for p in range(1, pages + 1):
        c.setFont("Helvetica", 8)
        c.drawString(50, h - 30, "GUIDELINES ON TEST FRAUD REPORTING")              # running header
        c.drawString(w - 120, 30, f"Page {p} of {pages}")                        # running footer
        y = h - 80
        if p == 1:
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, y, "Guidelines on test fraud reporting")
            y -= 40
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, f"{p}. Section number {p}")
        y -= 25
        if p == 2:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, "Article 3 Reporting frequency")
            y -= 18
        c.setFont("Helvetica", 10)
        words = _body(TOPICS[p - 1]).split()
        line = []
        for word in words:
            if c.stringWidth(" ".join(line + [word]), "Helvetica", 10) > 480:
                c.drawString(50, y, " ".join(line))
                y -= 13
                line = []
            line.append(word)
        if p == 3:
            c.drawString(50, y, "This paragraph discusses the regula-")
            y -= 13
            c.drawString(50, y, "tion of reporting obligations.")
            y -= 13
        c.drawString(50, y, " ".join(line))
        if p == 4:
            t = Table([["Category", "Count"], ["Card-not-present", "120"], ["Lost or stolen", "45"]], colWidths=[150, 80])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
            t.wrapOn(c, w, h)
            t.drawOn(c, 50, 200)
        c.showPage()
    c.save()


def test_pdf_headers_footers_headings_tables_and_hyphenation():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gl.pdf"
        _make_pdf(path)
        blocks = parse_pdf(path)
    text = "\n".join(b.text for b in blocks)
    assert "GUIDELINES ON TEST FRAUD REPORTING" not in text and "Page 2 of 5" not in text      # running lines removed
    heads = [(b.text, b.level, b.page) for b in blocks if b.kind == "heading"]
    assert ("Guidelines on test fraud reporting", 1, 1) in heads
    assert any(t == "2. Section number 2" and lvl == 2 for t, lvl, _ in heads)
    assert any(t.startswith("Article 3") and pg == 2 for t, _, pg in heads)
    tables = [b for b in blocks if b.kind == "table"]
    assert tables and "Card-not-present | 120" in tables[0].text and tables[0].page == 4
    chunks = chunk_blocks(blocks, "gl")
    assert any("regulation of reporting obligations" in c.text for c in chunks)
    assert all(c.page_start is not None for c in chunks)
    assert any("Article 3 Reporting frequency." in c.text for c in chunks)                # short sections share chunks
    assert any("Card-not-present | 120" in c.text for c in chunks)                         # small table kept in the text


def test_article_headings_split_from_inline_text_but_not_from_citations():
    from finguard.rag.parse import _structural_split
    assert _structural_split("Art. 5 (nouveau) – Les établissements assujettis doivent") == ("Art. 5 (nouveau)", "Les établissements assujettis doivent")
    assert _structural_split("Article Premier – La présente circulaire s'applique") == ("Article Premier", "La présente circulaire s'applique")
    assert _structural_split("Recommendation 10 Customer due diligence") == ("Recommendation 10 Customer due diligence", "")
    assert _structural_split("Article 18 of the RTS on SCA and CSC. The EBA clarifies") is None


def test_consolidated_header_guideline_titles_footnote_markers():
    """Small-type running header that changes part-way, a large 'Guideline N: title' line, a
    sentence citing 'Guideline 1.1', and a raised footnote marker."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise SkipTest("reportlab not installed") from exc
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "consolidated.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        w, h = A4
        for p in range(1, 21):
            c.setFont("Helvetica", 8)
            c.drawString(50, h - 30, "FINAL REPORT ON TEST GUIDELINES" if p <= 16 else "CONSOLIDATED TEST GUIDELINES REPORT")
            y = h - 80
            if p == 17:
                c.setFont("Helvetica", 16)
                c.drawString(50, y, "Guideline 3: Frequency, reporting timelines and reporting period")
                y -= 30
            c.setFont("Helvetica", 10)
            body = [f"Page {p} body: payment service providers report fraud data on {TOPICS[p % 5]} every six months.",
                    "The data described in Guideline 1.1, regardless of the amount, is reported in full.",
                    "The competent authority aggregates the transactions"]
            for line in body:
                c.drawString(50, y, line)
                y -= 13
            if p == 18:
                width = c.stringWidth(body[-1], "Helvetica", 10)
                c.setFont("Helvetica", 6)
                c.drawString(50 + width + 0.5, y + 13 + 4, "9")                 # raised footnote marker
                c.setFont("Helvetica", 10)
                c.drawString(50 + width + 6, y + 13, " received.")
            c.showPage()
        c.save()
        blocks = parse_pdf(path)
    text = "\n".join(b.text for b in blocks)
    assert "CONSOLIDATED TEST GUIDELINES REPORT" not in text and "FINAL REPORT ON TEST GUIDELINES" not in text
    heads = [b.text for b in blocks if b.kind == "heading"]
    assert "Guideline 3: Frequency, reporting timelines and reporting period" in heads
    assert not any(t.startswith("Guideline 1") for t in heads)
    assert "transactions9" not in text and "transactions received." in text.replace("  ", " ")
