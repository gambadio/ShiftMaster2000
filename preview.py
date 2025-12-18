"""
Calendar Preview Component for AI Shift Studio

Displays schedule in a Teams Shifts-style calendar view
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from models import ScheduleEntry, ScheduleConflict, TEAMS_COLOR_NAMES


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

    # Direct match on numeric codes
    if candidate in TEAMS_COLOR_NAMES:
        return candidate

    # Extract first digit sequence (handles '1. Weiß', 'Color 2', etc.)
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


def build_calendar_html(
    schedule_entries: List[ScheduleEntry],
    view_start: date,
    view_end: date,
    for_pdf: bool = False
) -> str:
    """
    Build calendar HTML without rendering. Returns HTML string.

    Args:
        schedule_entries: List of ScheduleEntry objects
        view_start: Start date for the view
        view_end: End date for the view
        for_pdf: If True, use print-friendly styles (white background)

    Returns:
        Complete HTML string with styles and table
    """
    from datetime import timedelta

    # Generate date range for view
    dates = []
    current = view_start
    while current <= view_end:
        dates.append(current)
        current += timedelta(days=1)

    if not dates:
        return "<p>No dates to display</p>"

    # Get unique employees
    employees = sorted(set(entry.employee_name for entry in schedule_entries))

    if not employees:
        return "<p>No employees found in schedule</p>"

    # Build grid data: employee x date
    dates_set = set(dates)
    grid_data = {emp: {d: [] for d in dates} for emp in employees}

    # Track seen entries to prevent duplicates
    seen_entries: set = set()

    def add_entry_to_grid(emp_grid: dict, target_date: date, entry) -> None:
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

            current = entry_start
            while current <= actual_end:
                if current in dates_set:
                    add_entry_to_grid(emp_grid, current, entry)
                current += timedelta(days=1)

    # Calculate total hours per employee
    employee_hours = {emp: 0.0 for emp in employees}
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

    # Build HTML
    html_parts = ['<style>']

    # Choose colors based on output mode
    if for_pdf:
        bg_main = '#ffffff'
        bg_header = '#f0f0f0'
        border_color = '#cccccc'
        text_color = '#000000'
        text_secondary = '#666666'
    else:
        bg_main = '#0d1b2a'
        bg_header = '#1b263b'
        border_color = '#2d3748'
        text_color = '#e2e8f0'
        text_secondary = '#cbd5e0'

    html_parts.append(f"""
        .teams-schedule {{
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: {bg_main};
        }}
        .teams-schedule th {{
            background-color: {bg_header};
            border: 1px solid {border_color};
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            color: {text_color};
        }}
        .teams-schedule td {{
            border: 1px solid {border_color};
            padding: 6px;
            vertical-align: top;
            min-height: 60px;
        }}
        .employee-cell {{
            font-weight: 600;
            background-color: {bg_header};
            padding: 12px 8px !important;
            white-space: nowrap;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .employee-name {{
            font-size: 14px;
            color: {text_color};
        }}
        .employee-hours {{
            font-size: 11px;
            color: {text_secondary};
            margin-top: 2px;
        }}
        .day-cell {{
            min-width: 120px;
            background-color: {bg_main};
        }}
        .shift-block {{
            margin: 3px 0;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.3;
        }}
        .shift-label {{
            font-weight: 600;
            display: block;
        }}
        .shift-time {{
            font-size: 11px;
            opacity: 0.9;
            margin-top: 2px;
        }}
        /* Teams color scheme */
        .color-1 {{ background-color: #ffffff; border: 2px solid #d2d0ce; color: #323130; }}
        .color-2 {{ background-color: #0078d4; color: white; }}
        .color-3 {{ background-color: #107c10; color: white; }}
        .color-4 {{ background-color: #8764b8; color: white; }}
        .color-5 {{ background-color: #e3008c; color: white; }}
        .color-6 {{ background-color: #ffb900; color: #323130; }}
        .color-8 {{ background-color: #002050; color: white; }}
        .color-9 {{ background-color: #004b1c; color: white; }}
        .color-10 {{ background-color: #5c2e91; color: white; }}
        .color-11 {{ background-color: #d13438; color: white; }}
        .color-12 {{ background-color: #ca5010; color: white; }}
        .color-13 {{ background-color: #a19f9d; color: white; }}
        .week-header {{
            font-size: 11px;
            color: {text_secondary};
            text-align: left;
            padding: 4px 8px !important;
            background-color: {bg_header} !important;
        }}
        .date-header {{
            font-size: 12px;
            color: {text_color};
            font-weight: 600;
        }}
    </style>
    """)

    html_parts.append('<table class="teams-schedule">')
    html_parts.append('<thead>')

    # Week row
    html_parts.append('<tr><th class="week-header">Week</th>')
    week_groups = []
    current_week = None
    for d in dates:
        week_num = d.isocalendar()[1]
        if week_num != current_week:
            week_groups.append({'week': week_num, 'count': 1})
            current_week = week_num
        else:
            week_groups[-1]['count'] += 1

    for wg in week_groups:
        html_parts.append(f'<th class="week-header" colspan="{wg["count"]}">Week {wg["week"]}</th>')
    html_parts.append('</tr>')

    # Month row
    html_parts.append('<tr><th class="week-header">Month</th>')
    month_groups = []
    current_month = None
    for d in dates:
        month_name = d.strftime("%B %Y")
        month_key = (d.year, d.month)
        if month_key != current_month:
            month_groups.append({'name': month_name, 'count': 1})
            current_month = month_key
        else:
            month_groups[-1]['count'] += 1

    for mg in month_groups:
        html_parts.append(f'<th class="week-header" colspan="{mg["count"]}">{mg["name"]}</th>')
    html_parts.append('</tr>')

    # Date row
    html_parts.append('<tr><th class="employee-cell">Employee</th>')
    for d in dates:
        day_name = d.strftime("%a")
        html_parts.append(f'<th><div class="date-header">{d.day}</div><div style="font-size:10px;color:{text_secondary};">{day_name}</div></th>')
    html_parts.append('</tr>')
    html_parts.append('</thead>')

    # Data rows
    html_parts.append('<tbody>')
    for emp in employees:
        total_hrs = employee_hours.get(emp, 0)
        html_parts.append('<tr>')
        html_parts.append(f'<td class="employee-cell">')
        html_parts.append(f'<div class="employee-name">{emp}</div>')
        html_parts.append(f'<div class="employee-hours">{total_hrs:.1f} Hrs</div>')
        html_parts.append('</td>')

        for d in dates:
            entries_for_day = grid_data[emp][d]
            html_parts.append('<td class="day-cell">')

            for entry in entries_for_day:
                color_code = _normalize_color_code(entry.color_code)
                color_class = f"color-{color_code}"

                time_str = ""
                if entry.start_time and entry.end_time:
                    start_t = entry.start_time.split(':')[0:2]
                    end_t = entry.end_time.split(':')[0:2]
                    time_str = f"{':'.join(start_t)} - {':'.join(end_t)}"

                if entry.entry_type == "time_off":
                    label = entry.reason or "Time Off"
                    color_class = "color-13"
                else:
                    label = entry.notes or entry.label or "Shift"

                html_parts.append(f'<div class="shift-block {color_class}">')
                html_parts.append(f'<span class="shift-label">{label}</span>')
                if time_str:
                    html_parts.append(f'<div class="shift-time">{time_str}</div>')
                html_parts.append('</div>')

            html_parts.append('</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')
    html_parts.append('</table>')

    return ''.join(html_parts)


def render_calendar_preview(
    schedule_entries: List[ScheduleEntry],
    start_date: date,
    end_date: date,
    title: Optional[str] = "Schedule Preview"
):
    """
    Render a calendar-style preview of the schedule with navigation

    Args:
        schedule_entries: List of ScheduleEntry objects
        start_date: Start date of the overall schedule range
        end_date: End date of the overall schedule range
        title: Optional title for the preview section
    """
    key_suffix = title or "schedule_preview"

    if title:
        st.subheader(title)

    if not schedule_entries:
        st.info("No schedule entries to display.")
        return

    # Initialize session state for calendar view offset (weeks from start)
    view_key = f"calendar_view_offset_{key_suffix}"
    jump_date_key = f"calendar_jump_date_{key_suffix}"
    last_jump_key = f"calendar_last_jump_{key_suffix}"
    
    if view_key not in st.session_state:
        st.session_state[view_key] = 0

    # Calculate current view start
    days_per_view = 7
    offset_start = start_date + timedelta(days=st.session_state[view_key] * days_per_view)
    days_since_monday = offset_start.weekday()
    current_view_start = offset_start - timedelta(days=days_since_monday)

    # Callback for date picker to avoid infinite rerun loop
    def on_date_jump():
        selected = st.session_state[jump_date_key]
        # Calculate the Monday of the selected date's week
        days_since_monday = selected.weekday()
        target_week_start = selected - timedelta(days=days_since_monday)
        # Only update if it's a different week
        if target_week_start != current_view_start:
            days_diff = (target_week_start - start_date).days
            st.session_state[view_key] = days_diff // 7
            st.session_state[last_jump_key] = target_week_start

    # Navigation controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("◀ Previous Week", key=f"prev_{key_suffix}"):
            st.session_state[view_key] -= 1
            st.rerun()

    with col2:
        if st.button("Next Week ▶", key=f"next_{key_suffix}"):
            st.session_state[view_key] += 1
            st.rerun()

    with col3:
        if st.button("📅 Go to Today", key=f"today_{key_suffix}"):
            # Navigate to the week containing today
            target_date = date.today()

            # Align target to the Monday of its week
            days_since_monday = target_date.weekday()
            week_start = target_date - timedelta(days=days_since_monday)

            # Calculate weeks from start_date to target week
            days_diff = (week_start - start_date).days
            st.session_state[view_key] = days_diff // 7
            st.rerun()

    with col4:
        # Date picker for jumping to any week - use on_change callback
        st.date_input(
            "Jump to date",
            value=current_view_start,
            key=jump_date_key,
            label_visibility="collapsed",
            on_change=on_date_jump
        )

    # Use already calculated view window
    view_start = current_view_start
    view_end = view_start + timedelta(days=6)  # Always show full week (Mon-Sun)

    # Show current week's date range
    st.caption(f"📅 {view_start.strftime('%b %d')} - {view_end.strftime('%b %d, %Y')}")

    # Generate date range for current view (always 7 days)
    dates = [view_start + timedelta(days=i) for i in range(7)]

    # Get unique employees
    employees = sorted(set(entry.employee_name for entry in schedule_entries))

    if not employees:
        st.info("No employees found in schedule.")
        return

    # Build grid data: employee x date
    # Pre-create date set for O(1) lookup
    dates_set = set(dates)
    grid_data = {emp: {d: [] for d in dates} for emp in employees}

    # Track seen entries to prevent duplicates
    # Key: (employee_name, date, start_time, end_time, entry_type)
    seen_entries: set = set()

    def add_entry_to_grid(emp_grid: dict, target_date: date, entry) -> None:
        """Add entry to grid if not already present (prevents duplicates)"""
        # Create a unique key for this entry on this date
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
            
        # Handle multi-day entries (especially time-off periods like vacations)
        entry_end = _parse_entry_date(getattr(entry, 'end_date', None))
        end_time = getattr(entry, 'end_time', None)

        # Normalize end_time - extract HH:MM
        end_time_clean = None
        if end_time:
            end_time_clean = end_time.strip()[:5] if len(end_time.strip()) >= 5 else end_time.strip()

        # Check if ends at midnight (00:00) - affects how we handle the end date
        ends_at_midnight = end_time_clean in ("00:00", "0:00", "24:00") if end_time_clean else False

        # Determine if this is a single-day entry or needs multi-day handling
        if not entry_end or entry_end == entry_start:
            # Single day entry (start and end date are the same)
            if entry_start in dates_set:
                add_entry_to_grid(emp_grid, entry_start, entry)
        elif ends_at_midnight and (entry_end - entry_start).days == 1:
            # Overnight shift OR 1-day time-off ending at midnight
            # e.g., shift 17:00-00:00 OR holiday from Jan 1 00:00 to Jan 2 00:00
            # Both should only show on the start date
            if entry_start in dates_set:
                add_entry_to_grid(emp_grid, entry_start, entry)
        else:
            # True multi-day entry (like vacations spanning multiple days)
            # When end_time is 00:00, the end_date is exclusive (don't include it)
            # e.g., vacation from Dec 15 to Dec 20 at 00:00 means Dec 15-19 inclusive
            actual_end = entry_end
            if ends_at_midnight:
                actual_end = entry_end - timedelta(days=1)

            current = entry_start
            while current <= actual_end:
                if current in dates_set:
                    add_entry_to_grid(emp_grid, current, entry)
                current += timedelta(days=1)

    # Render as a dataframe with colored cells
    st.caption(f"Showing {len(employees)} employees across {len(dates)} days")

    # Create HTML table with Teams Shifts styling
    html_parts = ['<style>']
    html_parts.append("""
        .teams-schedule {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0d1b2a;
        }
        .teams-schedule th {
            background-color: #1b263b;
            border: 1px solid #2d3748;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
            color: #e2e8f0;
        }
        .teams-schedule td {
            border: 1px solid #2d3748;
            padding: 6px;
            vertical-align: top;
            min-height: 60px;
        }
        .employee-cell {
            font-weight: 600;
            background-color: #1b263b;
            padding: 12px 8px !important;
            white-space: nowrap;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .employee-name {
            font-size: 14px;
            color: #e2e8f0;
        }
        .employee-hours {
            font-size: 11px;
            color: #cbd5e0;
            margin-top: 2px;
        }
        .day-cell {
            min-width: 140px;
            background-color: #0d1b2a;
        }
        .shift-block {
            margin: 3px 0;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.3;
            cursor: pointer;
            transition: transform 0.1s;
            position: relative;
            border: 2px solid transparent;
        }
        .shift-block:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .shift-block.generated {
            border-color: rgba(255, 255, 255, 0.5);
            box-shadow:
                0 0 4px rgba(255, 255, 255, 0.3),
                0 0 8px rgba(255, 255, 255, 0.2),
                inset 0 0 6px rgba(255, 255, 255, 0.08);
            animation: glow-pulse 2s ease-in-out infinite;
        }
        @keyframes glow-pulse {
            0%, 100% {
                box-shadow:
                    0 0 4px rgba(255, 255, 255, 0.3),
                    0 0 8px rgba(255, 255, 255, 0.2),
                    inset 0 0 6px rgba(255, 255, 255, 0.08);
            }
            50% {
                box-shadow:
                    0 0 6px rgba(255, 255, 255, 0.35),
                    0 0 10px rgba(255, 255, 255, 0.25),
                    inset 0 0 8px rgba(255, 255, 255, 0.1);
            }
        }
        .shift-label {
            font-weight: 600;
            display: block;
        }
        .shift-time {
            font-size: 11px;
            opacity: 0.9;
            margin-top: 2px;
        }
        /* Teams color scheme */
        .color-1 { background-color: #ffffff; border: 2px solid #d2d0ce; color: #323130; }
        .color-2 { background-color: #0078d4; color: white; }
        .color-3 { background-color: #107c10; color: white; }
        .color-4 { background-color: #8764b8; color: white; }
        .color-5 { background-color: #e3008c; color: white; }
        .color-6 { background-color: #ffb900; color: #323130; }
        .color-8 { background-color: #002050; color: white; }
        .color-9 { background-color: #004b1c; color: white; }
        .color-10 { background-color: #5c2e91; color: white; }
        .color-11 { background-color: #d13438; color: white; }
        .color-12 { background-color: #ca5010; color: white; }
        .color-13 { background-color: #a19f9d; color: white; }
        .week-header {
            font-size: 11px;
            color: #cbd5e0;
            text-align: left;
            padding: 4px 8px !important;
            background-color: #1b263b !important;
        }
        .date-header {
            font-size: 12px;
            color: #e2e8f0;
            font-weight: 600;
        }
        .hours-total {
            font-size: 11px;
            color: #cbd5e0;
            display: block;
            margin-top: 2px;
        }
    </style>
    """)

    # Calculate total hours per employee in a single pass (O(n) instead of O(n*m))
    employee_hours = {emp: 0.0 for emp in employees}
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

    html_parts.append('<table class="teams-schedule">')

    # Header row with week info
    html_parts.append('<thead>')

    # Week row - calculate colspan for each week
    html_parts.append('<tr><th class="week-header">Week</th>')
    week_groups = []
    current_week = None
    for d in dates:
        week_num = d.isocalendar()[1]
        if week_num != current_week:
            week_groups.append({'week': week_num, 'count': 1})
            current_week = week_num
        else:
            week_groups[-1]['count'] += 1

    for wg in week_groups:
        html_parts.append(f'<th class="week-header" colspan="{wg["count"]}">Week {wg["week"]}</th>')
    html_parts.append('</tr>')

    # Month row - calculate colspan for each month
    html_parts.append('<tr><th class="week-header">Month</th>')
    month_groups = []
    current_month = None
    for d in dates:
        month_name = d.strftime("%B %Y")
        month_key = (d.year, d.month)
        if month_key != current_month:
            month_groups.append({'name': month_name, 'count': 1})
            current_month = month_key
        else:
            month_groups[-1]['count'] += 1

    for mg in month_groups:
        html_parts.append(f'<th class="week-header" colspan="{mg["count"]}">{mg["name"]}</th>')
    html_parts.append('</tr>')

    # Date row
    html_parts.append('<tr><th class="employee-cell">Employee</th>')
    for d in dates:
        day_name = d.strftime("%a")
        date_str = d.strftime("%m/%d")
        html_parts.append(f'<th><div class="date-header">{d.day}</div><div style="font-size:10px;color:#cbd5e0;">{day_name}</div></th>')
    html_parts.append('</tr>')
    html_parts.append('</thead>')

    # Data rows - employee shifts
    html_parts.append('<tbody>')
    for emp in employees:
        total_hrs = employee_hours.get(emp, 0)
        html_parts.append('<tr>')
        html_parts.append(f'<td class="employee-cell">')
        html_parts.append(f'<div class="employee-name">{emp}</div>')
        html_parts.append(f'<div class="employee-hours">{total_hrs:.1f} Hrs</div>')
        html_parts.append('</td>')

        for d in dates:
            entries_for_day = grid_data[emp][d]
            html_parts.append('<td class="day-cell">')

            for entry in entries_for_day:
                color_code = _normalize_color_code(entry.color_code)
                color_class = f"color-{color_code}"
                extra_class = " generated" if getattr(entry, "source", "") == "generated" else ""

                # Format time display
                time_str = ""
                if entry.start_time and entry.end_time:
                    # Remove seconds if present
                    start_t = entry.start_time.split(':')[0:2]
                    end_t = entry.end_time.split(':')[0:2]
                    time_str = f"{':'.join(start_t)} - {':'.join(end_t)}"

                # Determine label and color
                if entry.entry_type == "time_off":
                    label = entry.reason or "Time Off"
                    color_class = "color-13"  # Force grey for time-off
                    extra_class = ""
                else:
                    # Show notes first (e.g., "Contact Team", "Dispatcher"), then label, then default
                    label = entry.notes or entry.label or "Shift"

                html_parts.append(f'<div class="shift-block {color_class}{extra_class}">')
                html_parts.append(f'<span class="shift-label">{label}</span>')
                if time_str:
                    html_parts.append(f'<div class="shift-time">{time_str}</div>')
                html_parts.append('</div>')

            html_parts.append('</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')
    html_parts.append('</table>')

    # Render using components for better HTML support
    full_html = ''.join(html_parts)
    # Dynamic height based on number of employees (roughly 60px per employee + 150px for headers)
    height = min(800, 150 + (len(employees) * 60))
    components.html(full_html, height=height, scrolling=True)


def render_statistics(schedule_entries: List[ScheduleEntry]):
    """Display schedule statistics"""
    if not schedule_entries:
        return

    shifts = [e for e in schedule_entries if e.entry_type == "shift"]
    timeoffs = [e for e in schedule_entries if e.entry_type == "time_off"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Shifts", len(shifts))

    with col2:
        st.metric("Time-Off Entries", len(timeoffs))

    with col3:
        unique_employees = len(set(e.employee_name for e in schedule_entries))
        st.metric("Employees", unique_employees)

    with col4:
        unique_dates = len(set(e.start_date for e in schedule_entries))
        st.metric("Days Covered", unique_dates)


def render_schedule_conflicts(conflicts: List[ScheduleConflict]):
    """
    Render ScheduleConflict objects from the schedule manager

    Args:
        conflicts: List of ScheduleConflict objects to display
    """
    if not conflicts:
        st.success("✅ No conflicts detected")
        return

    # Count by severity
    error_count = sum(1 for c in conflicts if c.severity == "error")
    warning_count = sum(1 for c in conflicts if c.severity == "warning")
    info_count = sum(1 for c in conflicts if c.severity == "info")

    # Show summary
    severity_icon = {
        "error": "🔴",
        "warning": "⚠️",
        "info": "ℹ️"
    }

    summary_parts = []
    if error_count:
        summary_parts.append(f"🔴 {error_count} errors")
    if warning_count:
        summary_parts.append(f"⚠️ {warning_count} warnings")
    if info_count:
        summary_parts.append(f"ℹ️ {info_count} info")

    st.markdown(f"**{len(conflicts)} conflicts detected:** {', '.join(summary_parts)}")

    with st.expander("📋 View All Conflicts", expanded=True):
        for i, conflict in enumerate(conflicts, 1):
            icon = severity_icon.get(conflict.severity, "•")

            # Build conflict display
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown(f"### {icon}")
            with col2:
                st.markdown(f"**{conflict.conflict_type.value.replace('_', ' ').title()}**")
                st.markdown(f"{conflict.message}")

                # Show additional context if available
                context_parts = []
                if conflict.employee_name:
                    context_parts.append(f"👤 {conflict.employee_name}")
                if conflict.date:
                    context_parts.append(f"📅 {conflict.date}")
                if conflict.shift_role:
                    context_parts.append(f"🏷️ {conflict.shift_role}")

                if context_parts:
                    st.caption(" | ".join(context_parts))

            if i < len(conflicts):
                st.divider()


def render_conflicts(schedule_entries: List[ScheduleEntry]):
    """Detect and display potential conflicts from ScheduleEntry objects"""
    conflicts = _detect_conflicts(schedule_entries)

    if conflicts:
        st.warning(f"⚠️ {len(conflicts)} potential conflicts detected")

        with st.expander("View Conflicts"):
            for i, conflict in enumerate(conflicts, 1):
                st.markdown(f"**{i}. {conflict['type']}**")
                st.markdown(f"- {conflict['description']}")
                st.markdown(f"- Severity: {'🔴' * conflict['severity']}")
    else:
        st.success("✅ No conflicts detected")


def _detect_conflicts(schedule_entries: List[ScheduleEntry]) -> List[Dict[str, Any]]:
    """Simple conflict detection"""
    conflicts = []

    # Group by employee and date
    emp_date_map: Dict[str, Dict[str, List[ScheduleEntry]]] = {}

    for entry in schedule_entries:
        if entry.employee_name not in emp_date_map:
            emp_date_map[entry.employee_name] = {}

        if entry.start_date not in emp_date_map[entry.employee_name]:
            emp_date_map[entry.employee_name][entry.start_date] = []

        emp_date_map[entry.employee_name][entry.start_date].append(entry)

    # Check for overlapping shifts on same day
    for emp, date_map in emp_date_map.items():
        for date_str, entries in date_map.items():
            shifts_on_day = [e for e in entries if e.entry_type == "shift"]
            timeoffs_on_day = [e for e in entries if e.entry_type == "time_off"]

            # Conflict: shift scheduled during time-off
            if shifts_on_day and timeoffs_on_day:
                conflicts.append({
                    "type": "Shift during time-off",
                    "description": f"{emp} has shift(s) scheduled on {date_str} but also has time-off",
                    "severity": 3
                })

            # Conflict: multiple shifts on same day (may be valid for concurrent shifts)
            if len(shifts_on_day) > 1:
                conflicts.append({
                    "type": "Multiple shifts",
                    "description": f"{emp} has {len(shifts_on_day)} shifts on {date_str}",
                    "severity": 1
                })

    return conflicts
