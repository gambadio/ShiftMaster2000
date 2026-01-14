"""
PDF Export for Calendar Preview

Exports the schedule calendar view to PDF using reportlab directly for reliable rendering.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from datetime import date, datetime, timedelta
from io import BytesIO
import re

from models import ScheduleEntry, TEAMS_COLOR_NAMES


def _parse_entry_date(date_str: Optional[str]) -> Optional[date]:
    """Parse schedule date strings in ISO or M/D/Y formats."""
    if not date_str:
        return None

    value = date_str.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_color_code(value: Optional[str]) -> str:
    """Return Teams color code (1-13) even when value includes names like '1. Weiß'."""
    if not value:
        return "1"

    candidate = value.strip()

    if candidate in TEAMS_COLOR_NAMES:
        return candidate

    match = re.search(r"\d+", candidate)
    if match:
        code = match.group(0)
        if code in TEAMS_COLOR_NAMES:
            return code

    normalized_candidate = candidate.lower().replace("ß", "ss")
    for code, name in TEAMS_COLOR_NAMES.items():
        normalized_name = name.lower().replace("ß", "ss")
        if normalized_candidate == normalized_name or normalized_name in normalized_candidate:
            return code

    return "1"


# Teams color palette (RGB tuples for reportlab)
TEAMS_COLORS_RGB = {
    "1": {"bg": (1.0, 1.0, 1.0), "text": (0.2, 0.19, 0.19)},
    "2": {"bg": (0.0, 0.47, 0.83), "text": (1.0, 1.0, 1.0)},
    "3": {"bg": (0.06, 0.49, 0.06), "text": (1.0, 1.0, 1.0)},
    "4": {"bg": (0.53, 0.39, 0.72), "text": (1.0, 1.0, 1.0)},
    "5": {"bg": (0.89, 0.0, 0.55), "text": (1.0, 1.0, 1.0)},
    "6": {"bg": (1.0, 0.73, 0.0), "text": (0.2, 0.19, 0.19)},
    "8": {"bg": (0.0, 0.13, 0.31), "text": (1.0, 1.0, 1.0)},
    "9": {"bg": (0.0, 0.29, 0.11), "text": (1.0, 1.0, 1.0)},
    "10": {"bg": (0.36, 0.18, 0.57), "text": (1.0, 1.0, 1.0)},
    "11": {"bg": (0.82, 0.2, 0.22), "text": (1.0, 1.0, 1.0)},
    "12": {"bg": (0.79, 0.31, 0.06), "text": (1.0, 1.0, 1.0)},
    "13": {"bg": (0.63, 0.62, 0.62), "text": (1.0, 1.0, 1.0)},
}


def export_calendar_to_pdf(
    schedule_entries: List[ScheduleEntry],
    start_date: date,
    end_date: date,
    title: Optional[str] = "Schedule Preview"
) -> bytes:
    """
    Export calendar view as PDF bytes using reportlab.

    Args:
        schedule_entries: List of ScheduleEntry objects
        start_date: Start date for export
        end_date: End date for export
        title: Optional title for the PDF

    Returns:
        PDF file as bytes
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF export. "
            "Install it with: pip install reportlab"
        )

    pdf_buffer = BytesIO()

    # A4 landscape with margins
    page_width, page_height = landscape(A4)
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        leftMargin=1*cm,
        rightMargin=1*cm,
        topMargin=1*cm,
        bottomMargin=1*cm
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=6,
        textColor=colors.HexColor('#1b263b')
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        textColor=colors.HexColor('#666666')
    )

    story = []

    # Generate pages (one week per page)
    current_week_start = start_date

    # Align to Monday
    days_since_monday = current_week_start.weekday()
    if days_since_monday > 0:
        current_week_start = current_week_start - timedelta(days=days_since_monday)

    first_page = True
    while current_week_start <= end_date:
        if not first_page:
            story.append(PageBreak())
        first_page = False

        week_end = current_week_start + timedelta(days=6)

        # Clip to actual date range
        view_start = max(current_week_start, start_date)
        view_end = min(week_end, end_date)

        # Add header
        week_num = view_start.isocalendar()[1]
        story.append(Paragraph(title or "Schedule Preview", title_style))
        story.append(Paragraph(
            f"Week {week_num}: {view_start.strftime('%B %d')} - {view_end.strftime('%B %d, %Y')}",
            subtitle_style
        ))

        # Build table for this week
        table = _build_week_table_reportlab(
            schedule_entries,
            view_start,
            view_end,
            page_width - 2*cm  # Available width
        )
        if table:
            story.append(table)

        current_week_start = week_end + timedelta(days=1)

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.read()


def _build_week_table_reportlab(
    schedule_entries: List[ScheduleEntry],
    view_start: date,
    view_end: date,
    available_width: float
):
    """Build the weekly timetable as a reportlab Table."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    # Generate date range for view
    dates = []
    current = view_start
    while current <= view_end:
        dates.append(current)
        current += timedelta(days=1)

    if not dates:
        return None

    # Get unique employees
    employees = sorted(set(entry.employee_name for entry in schedule_entries))

    if not employees:
        return None

    # Build grid data: employee x date
    dates_set = set(dates)
    grid_data: Dict[str, Dict[date, List[ScheduleEntry]]] = {
        emp: {d: [] for d in dates} for emp in employees
    }

    # Track seen entries to prevent duplicates
    seen_entries: set = set()

    def add_entry_to_grid(emp_grid: dict, target_date: date, entry: ScheduleEntry) -> None:
        entry_key = (
            entry.employee_name,
            target_date.isoformat(),
            getattr(entry, 'start_time', None),
            getattr(entry, 'end_time', None),
            entry.entry_type,
            getattr(entry, 'notes', None) or getattr(entry, 'label', None)
        )
        if entry_key not in seen_entries:
            seen_entries.add(entry_key)
            emp_grid[target_date].append(entry)

    for entry in schedule_entries:
        emp_grid = grid_data.get(entry.employee_name)
        if not emp_grid:
            continue

        entry_start = _parse_entry_date(entry.start_date)
        if not entry_start:
            continue

        entry_end = _parse_entry_date(getattr(entry, 'end_date', None))
        end_time = getattr(entry, 'end_time', None)

        end_time_clean = None
        if end_time:
            end_time_clean = end_time.strip()[:5] if len(end_time.strip()) >= 5 else end_time.strip()

        ends_at_midnight = end_time_clean in ("00:00", "0:00", "24:00") if end_time_clean else False

        if not entry_end or entry_end == entry_start:
            if entry_start in dates_set:
                add_entry_to_grid(emp_grid, entry_start, entry)
        elif ends_at_midnight and (entry_end - entry_start).days == 1:
            if entry_start in dates_set:
                add_entry_to_grid(emp_grid, entry_start, entry)
        else:
            actual_end = entry_end
            if ends_at_midnight:
                actual_end = entry_end - timedelta(days=1)

            current_day = entry_start
            while current_day <= actual_end:
                if current_day in dates_set:
                    add_entry_to_grid(emp_grid, current_day, entry)
                current_day += timedelta(days=1)

    # Calculate total hours per employee
    employee_hours: Dict[str, float] = {emp: 0.0 for emp in employees}
    for entry in schedule_entries:
        if entry.entry_type == "shift" and entry.employee_name in employee_hours:
            if entry.start_time and entry.end_time:
                try:
                    start = datetime.strptime(entry.start_time[:5], "%H:%M")
                    end = datetime.strptime(entry.end_time[:5], "%H:%M")
                    if end < start:
                        end += timedelta(days=1)
                    hours = (end - start).total_seconds() / 3600
                    if entry.unpaid_break:
                        hours -= entry.unpaid_break / 60
                    employee_hours[entry.employee_name] += hours
                except:
                    pass

    # Define styles for cells
    header_style = ParagraphStyle(
        'HeaderStyle',
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.white
    )
    emp_style = ParagraphStyle(
        'EmpStyle',
        fontSize=7,
        leading=9,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#1b263b')
    )
    shift_style = ParagraphStyle(
        'ShiftStyle',
        fontSize=6,
        leading=8,
        alignment=TA_LEFT
    )

    # Day names
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Build table data
    table_data = []

    # Header row
    header_row = [Paragraph("<b>Employee</b>", header_style)]
    for d in dates:
        day_name = day_names[d.weekday()]
        header_row.append(Paragraph(f"<b>{d.day}</b><br/>{day_name}", header_style))
    table_data.append(header_row)

    # Data rows
    for emp in employees:
        total_hrs = employee_hours.get(emp, 0)
        row = [Paragraph(f"<b>{emp}</b><br/><font size='6' color='#666666'>{total_hrs:.1f} Hrs</font>", emp_style)]

        for d in dates:
            entries_for_day = grid_data[emp][d]
            cell_content = []

            for entry in entries_for_day:
                color_code = _normalize_color_code(entry.color_code)
                if entry.entry_type == "time_off":
                    color_code = "13"

                rgb = TEAMS_COLORS_RGB.get(color_code, TEAMS_COLORS_RGB["1"])
                bg_hex = '#%02x%02x%02x' % (int(rgb["bg"][0]*255), int(rgb["bg"][1]*255), int(rgb["bg"][2]*255))
                text_hex = '#%02x%02x%02x' % (int(rgb["text"][0]*255), int(rgb["text"][1]*255), int(rgb["text"][2]*255))

                # Format time display
                time_str = ""
                if entry.start_time and entry.end_time:
                    start_t = entry.start_time.split(':')[0:2]
                    end_t = entry.end_time.split(':')[0:2]
                    time_str = f"{':'.join(start_t)}-{':'.join(end_t)}"

                # Determine label
                if entry.entry_type == "time_off":
                    label = entry.reason or "Time Off"
                else:
                    label = entry.notes or entry.label or "Shift"

                # Truncate long labels
                if len(label) > 15:
                    label = label[:14] + "…"

                shift_text = f"<font color='{text_hex}'><b>{label}</b></font>"
                if time_str:
                    shift_text += f"<br/><font size='5' color='{text_hex}'>{time_str}</font>"

                cell_content.append((shift_text, bg_hex))

            if cell_content:
                # Create nested table for multiple shifts
                shift_cells = []
                for text, bg in cell_content:
                    shift_cells.append([Paragraph(text, shift_style)])

                if len(shift_cells) == 1:
                    row.append(Paragraph(cell_content[0][0], shift_style))
                else:
                    # Multiple shifts - join with line breaks
                    combined = "<br/>".join([c[0] for c in cell_content])
                    row.append(Paragraph(combined, shift_style))
            else:
                row.append("")

        table_data.append(row)

    # Calculate column widths
    num_days = len(dates)
    emp_col_width = 85 * mm
    day_col_width = (available_width - emp_col_width) / num_days
    col_widths = [emp_col_width] + [day_col_width] * num_days

    # Create table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Table styling
    style_commands = [
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1b263b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),

        # Employee column
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f5f5f5')),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),

        # All cells
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]

    # Weekend column highlighting
    for i, d in enumerate(dates):
        if d.weekday() >= 5:  # Saturday or Sunday
            col_idx = i + 1  # +1 for employee column
            style_commands.append(('BACKGROUND', (col_idx, 0), (col_idx, 0), colors.HexColor('#2d3748')))
            style_commands.append(('BACKGROUND', (col_idx, 1), (col_idx, -1), colors.HexColor('#fafafa')))

    # Add colored backgrounds for shift cells
    for row_idx, emp in enumerate(employees, start=1):
        for col_idx, d in enumerate(dates, start=1):
            entries_for_day = grid_data[emp][d]
            if entries_for_day:
                # Use the first entry's color for the cell background
                entry = entries_for_day[0]
                color_code = _normalize_color_code(entry.color_code)
                if entry.entry_type == "time_off":
                    color_code = "13"
                rgb = TEAMS_COLORS_RGB.get(color_code, TEAMS_COLORS_RGB["1"])
                bg_color = colors.Color(rgb["bg"][0], rgb["bg"][1], rgb["bg"][2])
                style_commands.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), bg_color))

    table.setStyle(TableStyle(style_commands))

    return table


def get_date_range_from_entries(entries: List[ScheduleEntry]) -> tuple:
    """
    Extract min and max dates from schedule entries.

    Returns:
        Tuple of (min_date, max_date) or (None, None) if no valid dates
    """
    import pandas as pd

    dates = []
    for e in entries:
        try:
            dt = pd.to_datetime(e.start_date).date()
            dates.append(dt)
        except:
            pass

    if not dates:
        return None, None

    return min(dates), max(dates)
