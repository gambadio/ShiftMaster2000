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
    if view_key not in st.session_state:
        st.session_state[view_key] = 0

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

    # Calculate view window (7 days, Monday-Sunday)
    # Indefinite navigation - no boundary restrictions
    days_per_view = 7
    offset_start = start_date + timedelta(days=st.session_state[view_key] * days_per_view)

    # Align to week start (Monday)
    days_since_monday = offset_start.weekday()  # 0 = Monday, 6 = Sunday
    view_start = offset_start - timedelta(days=days_since_monday)
    view_end = view_start + timedelta(days=6)  # Always show full week (Mon-Sun)

    with col4:
        # Show current week's date range instead of week number
        # since navigation is now indefinite
        st.caption(f"{view_start.strftime('%b %d')} - {view_end.strftime('%b %d, %Y')}")

    # Generate date range for current view (always 7 days)
    dates = [view_start + timedelta(days=i) for i in range(7)]

    # Get unique employees
    employees = sorted(set(entry.employee_name for entry in schedule_entries))

    if not employees:
        st.info("No employees found in schedule.")
        return

    # Build grid data: employee x date
    grid_data = {}
    for emp in employees:
        grid_data[emp] = {d: [] for d in dates}

    for entry in schedule_entries:
        entry_date = _parse_entry_date(entry.start_date)
        if entry_date and entry_date in grid_data.get(entry.employee_name, {}):
            grid_data[entry.employee_name][entry_date].append(entry)

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

    # Calculate total hours per employee
    employee_hours = {}
    for emp in employees:
        total_hours = 0
        for entry in schedule_entries:
            if entry.employee_name == emp and entry.entry_type == "shift":
                if entry.start_time and entry.end_time:
                    try:
                        start = datetime.strptime(entry.start_time, "%H:%M")
                        end = datetime.strptime(entry.end_time, "%H:%M")
                        if end < start:
                            end += timedelta(days=1)
                        hours = (end - start).total_seconds() / 3600
                        if entry.unpaid_break:
                            hours -= entry.unpaid_break / 60
                        total_hours += hours
                    except:
                        pass
        employee_hours[emp] = total_hours

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
