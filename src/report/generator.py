"""Report generator — produces Markdown, PDF, and DOCX reports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.models import AnalysisReport, RequirementVerdict

import config as cfg

logger = logging.getLogger(__name__)

VERDICT_LABELS = {
    "match": "Соответствует",
    "partial": "Частично соответствует",
    "mismatch": "Не соответствует",
    "needs_clarification": "Требует уточнения",
}

VERDICT_ICONS = {
    "match": "✅",
    "partial": "🟡",
    "mismatch": "❌",
    "needs_clarification": "❓",
}

CATEGORY_LABELS = {
    "technical": "Техническое",
    "sla": "SLA",
    "legal": "Юридическое",
    "commercial": "Коммерческое",
    "security": "Информационная безопасность",
    "other": "Прочее",
}


def _section_label(v: RequirementVerdict) -> str:
    """Format the section/point label with fallback."""
    section = v.section.strip() if v.section else ""
    if section:
        return f"Пункт {section}"
    return f"Требование №{v.requirement_id}"


def _req_text(v: RequirementVerdict, max_len: int = 200) -> str:
    """Get requirement text with fallback to reasoning."""
    text = v.requirement_text.strip() if v.requirement_text else ""
    if text:
        return text[:max_len]
    # Fallback: use reasoning as context
    if v.reasoning:
        return f"[текст не извлечён] {v.reasoning[:max_len]}"
    return "[текст требования не извлечён]"


def generate_markdown(report: AnalysisReport) -> str:
    """Generate a full Markdown report."""
    lines = []
    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    lines.append(f"# Отчёт по анализу ТЗ: {report.document_name}")
    lines.append(f"*Дата анализа: {now}*\n")

    # Prominent compliance percentage at the very top
    pct = report.compliance_percentage
    lines.append(f"## Общий процент соответствия: {pct}%")
    lines.append(f"**{report.score} из {report.max_score} баллов**\n")

    # Methodology
    lines.append("### Методика оценки\n")
    lines.append("| Вердикт | Баллы |")
    lines.append("|---|---|")
    lines.append("| Полное соответствие | 2 |")
    lines.append("| Частичное соответствие | 1 |")
    lines.append("| Несоответствие / Требует уточнения | 0 |")
    lines.append("")
    lines.append(f"Максимальный балл = {report.total} пунктов × 2 = {report.max_score}. "
                 f"Набрано {report.score} баллов. "
                 f"Итоговый процент = {report.score} / {report.max_score} × 100 = **{pct}%**")
    lines.append("")

    # Summary table
    lines.append("## Сводка\n")
    lines.append(f"| Показатель | Значение |")
    lines.append(f"|---|---|")
    lines.append(f"| Всего требований | {report.total} |")
    lines.append(f"| {VERDICT_ICONS['match']} Соответствует | {report.match_count} |")
    lines.append(f"| {VERDICT_ICONS['partial']} Частично | {report.partial_count} |")
    lines.append(f"| {VERDICT_ICONS['mismatch']} Не соответствует | {report.mismatch_count} |")
    lines.append(f"| {VERDICT_ICONS['needs_clarification']} Требует уточнения | {report.clarification_count} |")
    lines.append(f"| **Баллы** | **{report.score} / {report.max_score}** |")
    lines.append(f"| **Общее соответствие** | **{pct}%** |")
    lines.append("")

    if report.summary:
        lines.append("### Резюме\n")
        lines.append(report.summary)
        lines.append("")

    # Mismatches detail
    mismatches = [v for v in report.verdicts if v.verdict == "mismatch"]
    if mismatches:
        lines.append("## Несоответствия\n")
        for v in mismatches:
            cat = CATEGORY_LABELS.get(v.category, v.category)
            lines.append(f"### {_section_label(v)} ({cat})")
            lines.append(f"> {_req_text(v)}")
            lines.append(f"\n**Причина:** {v.reasoning}")
            lines.append(f"\n**Рекомендация:** {v.recommendation}")
            if v.source_urls:
                links = ", ".join(f"[{_url_short_name(u)}]({u})" for u in v.source_urls[:5])
                lines.append(f"\n**Документация:** {links}")
            lines.append("")

    # Partial matches
    partials = [v for v in report.verdicts if v.verdict == "partial"]
    if partials:
        lines.append("## Частичное соответствие\n")
        for v in partials:
            cat = CATEGORY_LABELS.get(v.category, v.category)
            lines.append(f"### {_section_label(v)} ({cat})")
            lines.append(f"> {_req_text(v)}")
            lines.append(f"\n**Обоснование:** {v.reasoning}")
            lines.append(f"\n**Рекомендация:** {v.recommendation}")
            if v.source_urls:
                links = ", ".join(f"[{_url_short_name(u)}]({u})" for u in v.source_urls[:5])
                lines.append(f"\n**Документация:** {links}")
            lines.append("")

    # Needs clarification
    clarifications = [v for v in report.verdicts if v.verdict == "needs_clarification"]
    if clarifications:
        lines.append("## Требуют уточнения\n")
        lines.append("| Пункт ТЗ | Категория | Требование | Комментарий |")
        lines.append("|---|---|---|---|")
        for v in clarifications:
            text_short = _req_text(v, 80).replace("|", "\\|").replace("\n", " ")
            reasoning = v.reasoning.replace("|", "\\|").replace("\n", " ")
            cat = CATEGORY_LABELS.get(v.category, v.category)
            section = v.section or f"#{v.requirement_id}"
            lines.append(f"| {section} | {cat} | {text_short} | {reasoning} |")
        lines.append("")

    # Full detail by category
    lines.append("## Детализация по всем требованиям\n")
    categories_order = ["technical", "sla", "security", "legal", "commercial", "other"]
    for cat_key in categories_order:
        cat_verdicts = [v for v in report.verdicts if v.category == cat_key]
        if not cat_verdicts:
            continue
        cat_label = CATEGORY_LABELS.get(cat_key, cat_key)
        lines.append(f"### {cat_label}\n")
        for v in cat_verdicts:
            icon = VERDICT_ICONS.get(v.verdict, "")
            label = VERDICT_LABELS.get(v.verdict, v.verdict)
            lines.append(f"**{_section_label(v)}** {icon} {label} (уверенность: {v.confidence:.0%})")
            lines.append(f"> {_req_text(v)}")
            if v.reasoning:
                lines.append(f"\n*Обоснование:* {v.reasoning}")
            if v.evidence:
                lines.append(f"\n*Источник:* {v.evidence}")
            if v.source_urls:
                links = ", ".join(f"[{_url_short_name(u)}]({u})" for u in v.source_urls[:5])
                lines.append(f"\n*Документация:* {links}")
            if v.recommendation:
                lines.append(f"\n*Рекомендация:* {v.recommendation}")
            lines.append("")

    return "\n".join(lines)


def _url_short_name(url: str) -> str:
    """Extract a short label from a cloud.ru/docs URL."""
    # https://cloud.ru/docs/s3e/ug/topics/overview.html -> s3e / overview
    path = url.replace("https://cloud.ru/docs/", "").rstrip("/")
    parts = [p for p in path.split("/") if p not in ("ug", "topics", "index", "index.html")]
    if len(parts) > 2:
        return f"{parts[0]} / {parts[-1]}"
    return " / ".join(parts) if parts else url


def save_markdown(report: AnalysisReport, output_dir: Path | None = None) -> Path:
    """Save report as Markdown file."""
    output_dir = output_dir or cfg.REPORTS_DIR
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in report.document_name)
    filename = f"report_{safe_name}_{timestamp}.md"
    path = output_dir / filename
    md = generate_markdown(report)
    path.write_text(md, encoding="utf-8")
    logger.info("Saved Markdown report: %s", path)
    return path


def save_docx(report: AnalysisReport, output_dir: Path | None = None) -> Path:
    """Save report as DOCX file."""
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    output_dir = output_dir or cfg.REPORTS_DIR
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in report.document_name)
    filename = f"report_{safe_name}_{timestamp}.docx"
    path = output_dir / filename

    doc = Document()

    # Title
    doc.add_heading(f"Отчёт по анализу ТЗ: {report.document_name}", level=0)
    doc.add_paragraph(f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    # Prominent compliance percentage
    pct = report.compliance_percentage
    p = doc.add_paragraph()
    run = p.add_run(f"Общий процент соответствия: {pct}%")
    run.bold = True
    run.font.size = Pt(18)
    p = doc.add_paragraph()
    run = p.add_run(f"{report.score} из {report.max_score} баллов")
    run.bold = True
    run.font.size = Pt(14)

    # Methodology
    doc.add_heading("Методика оценки", level=2)
    meth_table = doc.add_table(rows=4, cols=2)
    meth_table.style = "Table Grid"
    meth_data = [
        ("Вердикт", "Баллы"),
        ("Полное соответствие", "2"),
        ("Частичное соответствие", "1"),
        ("Несоответствие / Требует уточнения", "0"),
    ]
    for i, (k, val) in enumerate(meth_data):
        meth_table.rows[i].cells[0].text = k
        meth_table.rows[i].cells[1].text = val
    doc.add_paragraph(
        f"Максимальный балл = {report.total} пунктов × 2 = {report.max_score}. "
        f"Набрано {report.score} баллов. "
        f"Итоговый процент = {report.score} / {report.max_score} × 100 = {pct}%"
    )

    # Summary table
    doc.add_heading("Сводка", level=1)
    table = doc.add_table(rows=8, cols=2)
    table.style = "Table Grid"
    summary_data = [
        ("Показатель", "Значение"),
        ("Всего требований", str(report.total)),
        ("Соответствует", str(report.match_count)),
        ("Частично", str(report.partial_count)),
        ("Не соответствует", str(report.mismatch_count)),
        ("Требует уточнения", str(report.clarification_count)),
        ("Баллы", f"{report.score} / {report.max_score}"),
        ("Общее соответствие", f"{pct}%"),
    ]
    for i, (k, v) in enumerate(summary_data):
        table.rows[i].cells[0].text = k
        table.rows[i].cells[1].text = v

    if report.summary:
        doc.add_heading("Резюме", level=2)
        doc.add_paragraph(report.summary)

    # Mismatches
    mismatches = [v for v in report.verdicts if v.verdict == "mismatch"]
    if mismatches:
        doc.add_heading("Несоответствия", level=1)
        t = doc.add_table(rows=1, cols=5)
        t.style = "Table Grid"
        for j, h in enumerate(["Пункт ТЗ", "Требование", "Причина", "Рекомендация", "Ссылки на документацию"]):
            t.rows[0].cells[j].text = h
        for v in mismatches:
            row = t.add_row()
            row.cells[0].text = v.section or f"#{v.requirement_id}"
            row.cells[1].text = _req_text(v, 150)
            row.cells[2].text = v.reasoning
            row.cells[3].text = v.recommendation
            row.cells[4].text = "\n".join(v.source_urls[:5]) if v.source_urls else ""

    # Partial
    partials = [v for v in report.verdicts if v.verdict == "partial"]
    if partials:
        doc.add_heading("Частичное соответствие", level=1)
        t = doc.add_table(rows=1, cols=5)
        t.style = "Table Grid"
        for j, h in enumerate(["Пункт ТЗ", "Требование", "Обоснование", "Рекомендация", "Ссылки на документацию"]):
            t.rows[0].cells[j].text = h
        for v in partials:
            row = t.add_row()
            row.cells[0].text = v.section or f"#{v.requirement_id}"
            row.cells[1].text = _req_text(v, 150)
            row.cells[2].text = v.reasoning
            row.cells[3].text = v.recommendation
            row.cells[4].text = "\n".join(v.source_urls[:5]) if v.source_urls else ""

    doc.save(str(path))
    logger.info("Saved DOCX report: %s", path)
    return path


def save_excel(report: AnalysisReport, output_dir: Path | None = None) -> Path:
    """Save report as Excel file with a detailed table of all requirements."""
    import pandas as pd

    output_dir = output_dir or cfg.REPORTS_DIR
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in report.document_name)
    filename = f"report_{safe_name}_{timestamp}.xlsx"
    path = output_dir / filename

    # Build rows for the main table
    rows = []
    for v in report.verdicts:
        score_val = 2 if v.verdict == "match" else (1 if v.verdict == "partial" else 0)
        rows.append({
            "Пункт ТЗ": v.section or f"#{v.requirement_id}",
            "Текст требования": _req_text(v, 1000),
            "Категория": CATEGORY_LABELS.get(v.category, v.category),
            "Вердикт": VERDICT_LABELS.get(v.verdict, v.verdict),
            "Баллы": score_val,
            "Уверенность": f"{v.confidence:.0%}",
            "Обоснование": v.reasoning,
            "Источник (цитата)": v.evidence,
            "Рекомендация": v.recommendation,
            "Ссылки на документацию": "\n".join(v.source_urls[:5]) if v.source_urls else "",
        })

    df = pd.DataFrame(rows)

    pct = report.compliance_percentage

    # Summary data
    summary_rows = [
        {"Показатель": "Всего требований", "Значение": report.total},
        {"Показатель": "Соответствует", "Значение": report.match_count},
        {"Показатель": "Частично", "Значение": report.partial_count},
        {"Показатель": "Не соответствует", "Значение": report.mismatch_count},
        {"Показатель": "Требует уточнения", "Значение": report.clarification_count},
        {"Показатель": "Баллы", "Значение": f"{report.score} / {report.max_score}"},
        {"Показатель": "Общее соответствие", "Значение": f"{pct}%"},
        {"Показатель": "", "Значение": ""},
        {"Показатель": "Методика оценки", "Значение": ""},
        {"Показатель": "Полное соответствие", "Значение": "2 балла"},
        {"Показатель": "Частичное соответствие", "Значение": "1 балл"},
        {"Показатель": "Несоответствие / Требует уточнения", "Значение": "0 баллов"},
    ]
    df_summary = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Сводка", index=False)
        df.to_excel(writer, sheet_name="Все требования", index=False)

        # Separate sheets by verdict
        for verdict_key, label in VERDICT_LABELS.items():
            filtered = df[df["Вердикт"] == label]
            if not filtered.empty:
                sheet_name = label[:31]  # Excel sheet name max 31 chars
                filtered.to_excel(writer, sheet_name=sheet_name, index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    try:
                        cell_len = len(str(cell.value or ""))
                        if cell_len > max_len:
                            max_len = cell_len
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    logger.info("Saved Excel report: %s", path)
    return path


def save_pdf(report: AnalysisReport, output_dir: Path | None = None) -> Path:
    """Save report as PDF file using fpdf2."""
    from fpdf import FPDF, XPos, YPos

    output_dir = output_dir or cfg.REPORTS_DIR
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in report.document_name)
    filename = f"report_{safe_name}_{timestamp}.pdf"
    path = output_dir / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Try to add a Unicode font with Cyrillic support
    font_added = False
    font_paths = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                pdf.add_font("UniFont", "", fp)
                font_added = True
                break
            except Exception:
                continue

    if not font_added:
        import glob
        for pattern in ["/Library/Fonts/*.ttf", "/System/Library/Fonts/*.ttf",
                        "/System/Library/Fonts/Supplemental/*.ttf"]:
            for f in glob.glob(pattern):
                try:
                    pdf.add_font("UniFont", "", f)
                    font_added = True
                    break
                except Exception:
                    continue
            if font_added:
                break

    font_name = "UniFont" if font_added else "Helvetica"
    # UniFont doesn't have a bold variant; we'll simulate bold via font size

    def _safe(text: str) -> str:
        """Clean text for PDF output."""
        return text.replace("\r", "").strip()

    pdf.add_page()

    # Title
    # Title
    pdf.set_font(font_name, "", 16)
    pdf.multi_cell(0, 10, _safe(f"Отчет по анализу ТЗ: {report.document_name}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(font_name, "", 10)
    pdf.cell(0, 8, f"Дата анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Compliance percentage — prominent
    pct = report.compliance_percentage
    pdf.set_font(font_name, "", 18)
    pdf.cell(0, 12, _safe(f"Общий процент соответствия: {pct}%"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 14)
    pdf.cell(0, 10, _safe(f"{report.score} из {report.max_score} баллов"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Methodology
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 8, _safe("Методика оценки"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 10)
    for line in [
        "Полное соответствие = 2 балла",
        "Частичное соответствие = 1 балл",
        "Несоответствие / Требует уточнения = 0 баллов",
        f"Максимальный балл = {report.total} x 2 = {report.max_score}",
        f"Набрано {report.score} баллов. Итого: {pct}%",
    ]:
        pdf.cell(0, 6, _safe(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Summary
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 8, _safe("Сводка"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 10)
    summary_lines = [
        f"Всего требований: {report.total}",
        f"Соответствует: {report.match_count}",
        f"Частично: {report.partial_count}",
        f"Не соответствует: {report.mismatch_count}",
        f"Требует уточнения: {report.clarification_count}",
        f"Баллы: {report.score} / {report.max_score}",
        f"Общее соответствие: {pct}%",
    ]
    for line in summary_lines:
        pdf.cell(0, 6, _safe(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    if report.summary:
        pdf.set_font(font_name, "", 11)
        pdf.cell(0, 8, _safe("Резюме"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, _safe(report.summary), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    # Verdicts by section
    for verdict_key, verdict_label in VERDICT_LABELS.items():
        filtered = [v for v in report.verdicts if v.verdict == verdict_key]
        if not filtered:
            continue

        pdf.set_font(font_name, "", 13)
        pdf.cell(0, 8, _safe(verdict_label), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        for v in filtered:
            cat = CATEGORY_LABELS.get(v.category, v.category)
            pdf.set_font(font_name, "", 11)
            pdf.multi_cell(0, 6, _safe(f"{_section_label(v)} ({cat})"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(font_name, "", 9)
            pdf.multi_cell(0, 5, _safe(f"Требование: {_req_text(v, 300)}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if v.reasoning:
                pdf.multi_cell(0, 5, _safe(f"Обоснование: {v.reasoning}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if v.recommendation:
                pdf.multi_cell(0, 5, _safe(f"Рекомендация: {v.recommendation}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if v.source_urls:
                pdf.multi_cell(0, 5, _safe(f"Документация: {', '.join(v.source_urls[:5])}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

    pdf.output(str(path))
    logger.info("Saved PDF report: %s", path)
    return path
