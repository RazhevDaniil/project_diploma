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

KEY_MATCHES_LIMIT = 10
CATEGORY_PRIORITY = {
    "technical": 0,
    "sla": 1,
    "security": 2,
    "legal": 3,
    "commercial": 4,
    "other": 5,
}

PLATFORM_VERDICT_SYMBOLS = {
    "match": "+",
    "partial": "±",
    "mismatch": "-",
    "needs_clarification": "?",
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


def _priority_sort_key(v: RequirementVerdict) -> tuple:
    return (
        CATEGORY_PRIORITY.get(v.category, 9),
        -v.confidence,
        v.section or "",
        v.requirement_id,
    )


def _assessment_ref_key(assessment: dict | object) -> str:
    source_urls = getattr(assessment, "source_urls", []) if not isinstance(assessment, dict) else assessment.get("source_urls", [])
    source_titles = getattr(assessment, "source_titles", []) if not isinstance(assessment, dict) else assessment.get("source_titles", [])
    if source_urls:
        return str(source_urls[0])
    if source_titles:
        return str(source_titles[0])
    return ""


def _collect_reference_map(report: AnalysisReport) -> dict[str, int]:
    refs: dict[str, int] = {}
    for verdict in report.verdicts:
        for assessment in verdict.platform_assessments:
            key = _assessment_ref_key(assessment)
            if key and key not in refs:
                refs[key] = len(refs) + 1
        for url in verdict.source_urls:
            if url and url not in refs:
                refs[url] = len(refs) + 1
    return refs


def _assessment_ref_label(assessment: dict | object, refs: dict[str, int]) -> str:
    key = _assessment_ref_key(assessment)
    if not key or key not in refs:
        return ""
    return f"[{refs[key]}]"


def _assessment_symbol(assessment: dict | object, refs: dict[str, int]) -> str:
    verdict = getattr(assessment, "verdict", "") if not isinstance(assessment, dict) else assessment.get("verdict", "")
    symbol = PLATFORM_VERDICT_SYMBOLS.get(verdict, "?")
    ref = _assessment_ref_label(assessment, refs)
    return f"{symbol} {ref}".strip()


def _platform_names(report: AnalysisReport) -> list[str]:
    names = []
    for verdict in report.verdicts:
        for assessment in verdict.platform_assessments:
            if assessment.platform_name and assessment.platform_name not in names:
                names.append(assessment.platform_name)
    return names


def _platform_matrix_rows(report: AnalysisReport, refs: dict[str, int]) -> list[dict]:
    platform_names = _platform_names(report)
    rows = []
    for verdict in report.verdicts:
        row = {
            "Пункт ТЗ": verdict.section or f"#{verdict.requirement_id}",
            "Требование": _req_text(verdict, 140),
        }
        by_platform = {item.platform_name: item for item in verdict.platform_assessments}
        for platform_name in platform_names:
            assessment = by_platform.get(platform_name)
            row[platform_name] = _assessment_symbol(assessment, refs) if assessment else "-"
        rows.append(row)
    return rows


def _platform_totals(report: AnalysisReport) -> dict[str, tuple[int, int, int, int]]:
    totals = {}
    for platform_name in _platform_names(report):
        match_count = partial_count = mismatch_count = clarification_count = 0
        for verdict in report.verdicts:
            for assessment in verdict.platform_assessments:
                if assessment.platform_name != platform_name:
                    continue
                if assessment.verdict == "match":
                    match_count += 1
                elif assessment.verdict == "partial":
                    partial_count += 1
                elif assessment.verdict == "mismatch":
                    mismatch_count += 1
                else:
                    clarification_count += 1
        totals[platform_name] = (match_count, partial_count, mismatch_count, clarification_count)
    return totals


def _reference_title_from_key(key: str) -> str:
    if key.startswith("http"):
        return _url_short_name(key)
    return key


def _decision_summary(report: AnalysisReport) -> str:
    if report.mismatch_count > 0:
        return "Обнаружены несоответствия. Начните проверку отчёта с блокеров ниже."
    if report.clarification_count > 0:
        return "Критичных блокеров не найдено, но есть пункты, требующие ручного уточнения."
    if report.partial_count > 0:
        return "Явных блокеров не найдено, но есть частичные соответствия, требующие доработки."
    return "Явных блокеров не найдено. Достаточно выборочно перепроверить подтверждённые ключевые требования."


def _top_key_matches(report: AnalysisReport, limit: int = KEY_MATCHES_LIMIT) -> list[RequirementVerdict]:
    matches = [v for v in report.verdicts if v.verdict == "match"]
    matches.sort(key=_priority_sort_key)
    return matches[:limit]


def _format_problem_entry(lines: list[str], verdict: RequirementVerdict, reason_label: str) -> None:
    cat = CATEGORY_LABELS.get(verdict.category, verdict.category)
    lines.append(f"### {_section_label(verdict)} ({cat})")
    lines.append(f"**Пункт / раздел в ТЗ:** {verdict.section or f'#{verdict.requirement_id}'}")
    lines.append(f"**Требование:** {_req_text(verdict)}")
    lines.append(f"**{reason_label}:** {verdict.reasoning or 'Требуется ручная проверка'}")
    if verdict.recommendation:
        lines.append(f"**Рекомендация:** {verdict.recommendation}")
    if verdict.source_urls:
        links = ", ".join(f"[{_url_short_name(u)}]({u})" for u in verdict.source_urls[:5])
        lines.append(f"**Документация:** {links}")
    if verdict.platform_assessments:
        lines.append("**Оценка по платформам/услугам:**")
        for assessment in verdict.platform_assessments:
            label = VERDICT_LABELS.get(assessment.verdict, assessment.verdict)
            source_type = "внешняя услуга" if assessment.source_type == "external_service" else "платформа"
            lines.append(f"- {assessment.platform_name} ({source_type}): {label}. {assessment.reasoning}")
    if verdict.requires_external_service:
        lines.append(f"**Нужна проработка подрядчиков:** {verdict.external_service_notes or 'Да'}")
    lines.append("")


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
    lines.append(f"**Быстрый вывод:** {_decision_summary(report)}\n")

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

    refs = _collect_reference_map(report)
    matrix_rows = _platform_matrix_rows(report, refs)
    platform_names = _platform_names(report)
    if matrix_rows and platform_names:
        lines.append("## Матрица соответствия по платформам\n")
        header = ["Пункт ТЗ", "Требование"] + platform_names
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in matrix_rows:
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ") for column in header) + " |")
        totals = _platform_totals(report)
        lines.append("")
        lines.append("### Итого по платформам\n")
        lines.append("| Платформа / услуга | + | ± | - | ? |")
        lines.append("|---|---:|---:|---:|---:|")
        for platform_name, (match_count, partial_count, mismatch_count, clarification_count) in totals.items():
            lines.append(f"| {platform_name} | {match_count} | {partial_count} | {mismatch_count} | {clarification_count} |")
        lines.append("")

    external_items = [v for v in report.verdicts if v.requires_external_service]
    if external_items:
        lines.append("## Требования для проработки внешних услуг / подрядчиков\n")
        for v in external_items:
            lines.append(f"- **{v.section or f'#{v.requirement_id}'}**: {_req_text(v, 180)}")
            if v.external_service_notes:
                lines.append(f"  - {v.external_service_notes}")
        lines.append("")

    mismatches = sorted([v for v in report.verdicts if v.verdict == "mismatch"], key=_priority_sort_key)
    clarifications = sorted([v for v in report.verdicts if v.verdict == "needs_clarification"], key=_priority_sort_key)
    partials = sorted([v for v in report.verdicts if v.verdict == "partial"], key=_priority_sort_key)
    key_matches = _top_key_matches(report)

    # Priority checks first
    lines.append("## Что проверить в первую очередь\n")
    lines.append(f"- Несоответствия: **{report.mismatch_count}**")
    lines.append(f"- Требуют уточнения: **{report.clarification_count}**")
    lines.append(f"- Частичные соответствия: **{report.partial_count}**")
    lines.append("")

    if mismatches:
        lines.append("## Несоответствия\n")
        for v in mismatches:
            _format_problem_entry(lines, v, "Причина")

    if clarifications:
        lines.append("## Требуют уточнения\n")
        for v in clarifications:
            _format_problem_entry(lines, v, "Комментарий")

    if partials:
        lines.append("## Частичное соответствие\n")
        for v in partials:
            _format_problem_entry(lines, v, "Обоснование")

    if key_matches:
        lines.append("## Подтверждённые важные соответствия\n")
        lines.append(f"Показаны только наиболее значимые подтверждённые пункты: {len(key_matches)} из {report.match_count}.")
        lines.append("")
        for v in key_matches:
            cat = CATEGORY_LABELS.get(v.category, v.category)
            lines.append(f"### {_section_label(v)} ({cat})")
            lines.append(f"**Пункт / раздел в ТЗ:** {v.section or f'#{v.requirement_id}'}")
            lines.append(f"**Требование:** {_req_text(v)}")
            if v.reasoning:
                lines.append(f"**Почему соответствует:** {v.reasoning}")
            if v.source_urls:
                links = ", ".join(f"[{_url_short_name(u)}]({u})" for u in v.source_urls[:5])
                lines.append(f"**Документация:** {links}")
            lines.append("")

    if refs:
        lines.append("## Сноски RAG\n")
        for key, index in sorted(refs.items(), key=lambda item: item[1]):
            if key.startswith("http"):
                lines.append(f"[{index}] [{_reference_title_from_key(key)}]({key})")
            else:
                lines.append(f"[{index}] {key}")
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

    refs = _collect_reference_map(report)
    matrix_rows = _platform_matrix_rows(report, refs)
    platform_names = _platform_names(report)
    if matrix_rows and platform_names:
        doc.add_heading("Матрица соответствия по платформам", level=1)
        header = ["Пункт ТЗ", "Требование"] + platform_names
        matrix_table = doc.add_table(rows=1, cols=len(header))
        matrix_table.style = "Table Grid"
        for j, h in enumerate(header):
            matrix_table.rows[0].cells[j].text = h
        for item in matrix_rows:
            row = matrix_table.add_row()
            for j, column in enumerate(header):
                row.cells[j].text = str(item.get(column, ""))

        doc.add_heading("Итого по платформам", level=2)
        totals = _platform_totals(report)
        totals_table = doc.add_table(rows=1, cols=5)
        totals_table.style = "Table Grid"
        for j, h in enumerate(["Платформа / услуга", "+", "±", "-", "?"]):
            totals_table.rows[0].cells[j].text = h
        for platform_name, (match_count, partial_count, mismatch_count, clarification_count) in totals.items():
            row = totals_table.add_row()
            row.cells[0].text = platform_name
            row.cells[1].text = str(match_count)
            row.cells[2].text = str(partial_count)
            row.cells[3].text = str(mismatch_count)
            row.cells[4].text = str(clarification_count)

    external_items = [v for v in report.verdicts if v.requires_external_service]
    if external_items:
        doc.add_heading("Требования для проработки внешних услуг / подрядчиков", level=1)
        for v in external_items:
            doc.add_paragraph(
                f"{v.section or f'#{v.requirement_id}'}: {_req_text(v, 180)} "
                f"{v.external_service_notes or ''}",
                style=None,
            )

    doc.add_heading("Что проверить в первую очередь", level=1)
    doc.add_paragraph(_decision_summary(report))

    mismatches = sorted([v for v in report.verdicts if v.verdict == "mismatch"], key=_priority_sort_key)
    clarifications = sorted([v for v in report.verdicts if v.verdict == "needs_clarification"], key=_priority_sort_key)
    partials = sorted([v for v in report.verdicts if v.verdict == "partial"], key=_priority_sort_key)
    key_matches = _top_key_matches(report)

    if mismatches:
        doc.add_heading("Несоответствия", level=1)
        t = doc.add_table(rows=1, cols=6)
        t.style = "Table Grid"
        for j, h in enumerate(["Пункт ТЗ", "Категория", "Требование", "Причина", "Рекомендация", "Ссылки на документацию"]):
            t.rows[0].cells[j].text = h
        for v in mismatches:
            row = t.add_row()
            row.cells[0].text = v.section or f"#{v.requirement_id}"
            row.cells[1].text = CATEGORY_LABELS.get(v.category, v.category)
            row.cells[2].text = _req_text(v, 150)
            row.cells[3].text = v.reasoning
            row.cells[4].text = v.recommendation
            row.cells[5].text = "\n".join(v.source_urls[:5]) if v.source_urls else ""

    if clarifications:
        doc.add_heading("Требуют уточнения", level=1)
        t = doc.add_table(rows=1, cols=4)
        t.style = "Table Grid"
        for j, h in enumerate(["Пункт ТЗ", "Категория", "Требование", "Комментарий"]):
            t.rows[0].cells[j].text = h
        for v in clarifications:
            row = t.add_row()
            row.cells[0].text = v.section or f"#{v.requirement_id}"
            row.cells[1].text = CATEGORY_LABELS.get(v.category, v.category)
            row.cells[2].text = _req_text(v, 150)
            row.cells[3].text = v.reasoning

    if partials:
        doc.add_heading("Частичное соответствие", level=1)
        t = doc.add_table(rows=1, cols=6)
        t.style = "Table Grid"
        for j, h in enumerate(["Пункт ТЗ", "Категория", "Требование", "Обоснование", "Рекомендация", "Ссылки на документацию"]):
            t.rows[0].cells[j].text = h
        for v in partials:
            row = t.add_row()
            row.cells[0].text = v.section or f"#{v.requirement_id}"
            row.cells[1].text = CATEGORY_LABELS.get(v.category, v.category)
            row.cells[2].text = _req_text(v, 150)
            row.cells[3].text = v.reasoning
            row.cells[4].text = v.recommendation
            row.cells[5].text = "\n".join(v.source_urls[:5]) if v.source_urls else ""

    if key_matches:
        doc.add_heading("Подтверждённые важные соответствия", level=1)
        t = doc.add_table(rows=1, cols=4)
        t.style = "Table Grid"
        for j, h in enumerate(["Пункт ТЗ", "Категория", "Требование", "Документация"]):
            t.rows[0].cells[j].text = h
        for v in key_matches:
            row = t.add_row()
            row.cells[0].text = v.section or f"#{v.requirement_id}"
            row.cells[1].text = CATEGORY_LABELS.get(v.category, v.category)
            row.cells[2].text = _req_text(v, 150)
            row.cells[3].text = "\n".join(v.source_urls[:5]) if v.source_urls else ""

    if refs:
        doc.add_heading("Сноски RAG", level=1)
        for key, index in sorted(refs.items(), key=lambda item: item[1]):
            doc.add_paragraph(f"[{index}] {key}")

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
        platform_summary = []
        for assessment in v.platform_assessments:
            platform_summary.append(
                f"{assessment.platform_name}: {VERDICT_LABELS.get(assessment.verdict, assessment.verdict)}"
            )
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
            "Оценка по платформам": "\n".join(platform_summary),
            "Нужна проработка подрядчиков": "Да" if v.requires_external_service else "Нет",
            "Внешние услуги / комментарий": v.external_service_notes,
        })

    df = pd.DataFrame(rows)
    refs = _collect_reference_map(report)
    df_matrix = pd.DataFrame(_platform_matrix_rows(report, refs))
    detail_rows = []
    for v in report.verdicts:
        for assessment in v.platform_assessments:
            detail_rows.append(
                {
                    "Пункт ТЗ": v.section or f"#{v.requirement_id}",
                    "Требование": _req_text(v, 500),
                    "Платформа / услуга": assessment.platform_name,
                    "Тип источника": assessment.source_type,
                    "Вердикт": VERDICT_LABELS.get(assessment.verdict, assessment.verdict),
                    "Уверенность": f"{assessment.confidence:.0%}",
                    "Обоснование": assessment.reasoning,
                    "Сноски": ", ".join(assessment.evidence_refs),
                    "Источники": "\n".join(assessment.source_urls or assessment.source_titles),
                    "Рекомендация": assessment.recommendation,
                }
            )
    df_platform_detail = pd.DataFrame(detail_rows)
    df_refs = pd.DataFrame(
        [{"Сноска": f"[{index}]", "Источник": key} for key, index in sorted(refs.items(), key=lambda item: item[1])]
    )

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
        if not df_matrix.empty:
            df_matrix.to_excel(writer, sheet_name="Матрица платформ", index=False)
        if not df_platform_detail.empty:
            df_platform_detail.to_excel(writer, sheet_name="Платформы детально", index=False)
        if not df_refs.empty:
            df_refs.to_excel(writer, sheet_name="Сноски RAG", index=False)
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

    totals = _platform_totals(report)
    if totals:
        pdf.set_font(font_name, "", 12)
        pdf.cell(0, 8, _safe("Итого по платформам / услугам"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 9)
        for platform_name, (match_count, partial_count, mismatch_count, clarification_count) in totals.items():
            pdf.multi_cell(
                0,
                5,
                _safe(f"{platform_name}: + {match_count}, ± {partial_count}, - {mismatch_count}, ? {clarification_count}"),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
        pdf.ln(3)

    external_items = [v for v in report.verdicts if v.requires_external_service]
    if external_items:
        pdf.set_font(font_name, "", 12)
        pdf.cell(0, 8, _safe("Внешние услуги / подрядчики"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 9)
        for v in external_items[:20]:
            pdf.multi_cell(
                0,
                5,
                _safe(f"{v.section or f'#{v.requirement_id}'}: {_req_text(v, 220)} {v.external_service_notes or ''}"),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
        pdf.ln(3)

    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 8, _safe("Что проверить в первую очередь"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 10)
    pdf.multi_cell(0, 6, _safe(_decision_summary(report)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    sections = [
        ("Несоответствия", sorted([v for v in report.verdicts if v.verdict == "mismatch"], key=_priority_sort_key), "Причина"),
        ("Требуют уточнения", sorted([v for v in report.verdicts if v.verdict == "needs_clarification"], key=_priority_sort_key), "Комментарий"),
        ("Частичное соответствие", sorted([v for v in report.verdicts if v.verdict == "partial"], key=_priority_sort_key), "Обоснование"),
        ("Подтверждённые важные соответствия", _top_key_matches(report), "Почему соответствует"),
    ]

    for title, items, reason_label in sections:
        if not items:
            continue
        pdf.set_font(font_name, "", 13)
        pdf.cell(0, 8, _safe(title), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        for v in items:
            cat = CATEGORY_LABELS.get(v.category, v.category)
            pdf.set_font(font_name, "", 11)
            pdf.multi_cell(0, 6, _safe(f"{_section_label(v)} ({cat})"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(font_name, "", 9)
            pdf.multi_cell(0, 5, _safe(f"Пункт / раздел в ТЗ: {v.section or f'#{v.requirement_id}'}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(0, 5, _safe(f"Требование: {_req_text(v, 300)}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if v.reasoning:
                pdf.multi_cell(0, 5, _safe(f"{reason_label}: {v.reasoning}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if v.recommendation and v.verdict != "match":
                pdf.multi_cell(0, 5, _safe(f"Рекомендация: {v.recommendation}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if v.source_urls:
                pdf.multi_cell(0, 5, _safe(f"Документация: {', '.join(v.source_urls[:5])}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)

    pdf.output(str(path))
    logger.info("Saved PDF report: %s", path)
    return path
