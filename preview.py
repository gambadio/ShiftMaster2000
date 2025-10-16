"""
Calendar Preview Component for Shift Prompt Studio

Displays schedule in a Teams Shifts-style calendar view
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import streamlit as st
import pandas as pd
from models import ScheduleEntry, TEAMS_COLOR_NAMES


def render_calendar_preview(
    schedule_entries: List[ScheduleEntry],
    start_date: date,
    end_date: date,
    title: str = "Schedule Preview"
):
    """
    Render a calendar-style preview of the schedule

    Args:
        schedule_entries: List of ScheduleEntry objects
        start_date: Start date of the view
        end_date: End date of the view
        title: Title for the preview section
    """
    st.subheader(title)

    if not schedule_entries:
        st.info("No schedule entries to display.")
        return

    # Generate date range
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

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
        entry_date = datetime.fromisoformat(entry.start_date).date()
        if entry_date in grid_data.get(entry.employee_name, {}):
            grid_data[entry.employee_name][entry_date].append(entry)

    # Render as a dataframe with colored cells
    st.caption(f"Showing {len(employees)} employees across {len(dates)} days")

    # Create HTML table for better styling
    html_parts = ['<style>']
    html_parts.append("""
        .schedule-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 12px;
        }
        .schedule-table th, .schedule-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .schedule-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            position: sticky;
            top: 0;
        }
        .shift-block {
            margin: 2px 0;
            padding: 4px;
            border-radius: 3px;
            font-size: 11px;
            color: #000;
        }
        .color-1 { background-color: #ffffff; border: 1px solid #ccc; }
        .color-2 { background-color: #3b82f6; color: white; }
        .color-3 { background-color: #10b981; color: white; }
        .color-4 { background-color: #a855f7; color: white; }
        .color-5 { background-color: #ec4899; color: white; }
        .color-6 { background-color: #eab308; }
        .color-8 { background-color: #1e3a8a; color: white; }
        .color-9 { background-color: #065f46; color: white; }
        .color-10 { background-color: #581c87; color: white; }
        .color-11 { background-color: #9f1239; color: white; }
        .color-12 { background-color: #854d0e; color: white; }
        .color-13 { background-color: #9ca3af; }
    </style>
    """)

    html_parts.append('<table class="schedule-table">')

    # Header row
    html_parts.append('<thead><tr><th>Employee</th>')
    for d in dates:
        day_name = d.strftime("%a")
        date_str = d.strftime("%m/%d")
        html_parts.append(f'<th>{day_name}<br/>{date_str}</th>')
    html_parts.append('</tr></thead>')

    # Data rows
    html_parts.append('<tbody>')
    for emp in employees:
        html_parts.append(f'<tr><td style="font-weight: bold;">{emp}</td>')
        for d in dates:
            entries_for_day = grid_data[emp][d]
            html_parts.append('<td style="min-width: 120px;">')
            for entry in entries_for_day:
                color_class = f"color-{entry.color_code}" if entry.color_code else "color-1"
                time_str = ""
                if entry.start_time and entry.end_time:
                    time_str = f"{entry.start_time}-{entry.end_time}"

                label = entry.label or entry.reason or "Shift"
                html_parts.append(f'<div class="shift-block {color_class}">')
                html_parts.append(f'{label}')
                if time_str:
                    html_parts.append(f'<br/>{time_str}')
                html_parts.append('</div>')
            html_parts.append('</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')
    html_parts.append('</table>')

    st.markdown(''.join(html_parts), unsafe_allow_html=True)


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


def render_conflicts(schedule_entries: List[ScheduleEntry]):
    """Detect and display potential conflicts"""
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
