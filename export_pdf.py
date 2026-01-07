"""
PDF Export for Calendar Preview

Exports the schedule calendar view to PDF using xhtml2pdf (pure Python, no GTK required).
"""

from __future__ import annotations
from typing import List, Optional
from datetime import date, timedelta
from io import BytesIO

from models import ScheduleEntry


def export_calendar_to_pdf(
    schedule_entries: List[ScheduleEntry],
    start_date: date,
    end_date: date,
    title: Optional[str] = "Schedule Preview"
) -> bytes:
    """
    Export calendar view as PDF bytes.

    Args:
        schedule_entries: List of ScheduleEntry objects
        start_date: Start date for export
        end_date: End date for export
        title: Optional title for the PDF

    Returns:
        PDF file as bytes
    """
    try:
        from xhtml2pdf import pisa
    except ImportError:
        raise ImportError(
            "xhtml2pdf is required for PDF export. "
            "Install it with: pip install xhtml2pdf"
        )

    from preview import build_calendar_html

    # Generate pages (one week per page)
    pages_html = []
    current_week_start = start_date

    # Align to Monday
    days_since_monday = current_week_start.weekday()
    if days_since_monday > 0:
        current_week_start = current_week_start - timedelta(days=days_since_monday)

    while current_week_start <= end_date:
        week_end = current_week_start + timedelta(days=6)

        # Clip to actual date range
        view_start = max(current_week_start, start_date)
        view_end = min(week_end, end_date)

        # Build HTML for this week
        week_html = build_calendar_html(
            schedule_entries,
            view_start,
            view_end,
            for_pdf=True
        )

        # Add week header
        week_num = view_start.isocalendar()[1]
        week_header = f"""
        <div class="page-header">
            <h2>{title}</h2>
            <p>Week {week_num}: {view_start.strftime('%B %d')} - {view_end.strftime('%B %d, %Y')}</p>
        </div>
        """

        pages_html.append(f"<div class='pdf-page'>{week_header}{week_html}</div>")

        current_week_start = week_end + timedelta(days=1)

    # Combine all pages
    full_html = _wrap_for_pdf(pages_html, title)

    # Convert to PDF using xhtml2pdf
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(full_html, dest=pdf_buffer)
    
    if pisa_status.err:
        raise RuntimeError(f"PDF generation failed with {pisa_status.err} errors")
    
    pdf_buffer.seek(0)
    return pdf_buffer.read()


def _wrap_for_pdf(pages: List[str], title: str) -> str:
    """Wrap HTML pages with PDF-specific styles and structure."""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            @page {{
                size: A4 landscape;
                margin: 1cm;
            }}

            body {{
                font-family: Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: white;
                color: black;
                font-size: 10px;
            }}

            .pdf-page {{
                page-break-after: always;
                padding: 10px;
            }}

            .pdf-page:last-child {{
                page-break-after: auto;
            }}

            .page-header {{
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #333;
            }}

            .page-header h2 {{
                margin: 0 0 5px 0;
                font-size: 16px;
                color: #333;
            }}

            .page-header p {{
                margin: 0;
                font-size: 12px;
                color: #666;
            }}

            /* Override table styles for PDF */
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 9px;
            }}

            th, td {{
                padding: 4px 6px;
                border: 1px solid #ddd;
                text-align: left;
                vertical-align: top;
            }}

            th {{
                background-color: #f5f5f5;
                font-weight: bold;
            }}

            .shift-block {{
                font-size: 8px;
                padding: 2px 4px;
                margin: 1px 0;
                border-radius: 3px;
            }}

            .employee-name {{
                font-size: 10px;
                font-weight: bold;
            }}

            .employee-hours {{
                font-size: 8px;
            }}

            .day-cell {{
                min-width: 80px;
            }}
        </style>
    </head>
    <body>
        {''.join(pages)}
    </body>
    </html>
    """


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
