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

    # Create HTML table with Teams Shifts styling
    html_parts = ['<style>']
    html_parts.append("""
        .teams-schedule {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #fff;
        }
        .teams-schedule th {
            background-color: #f3f2f1;
            border: 1px solid #edebe9;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .teams-schedule td {
            border: 1px solid #edebe9;
            padding: 6px;
            vertical-align: top;
            min-height: 60px;
        }
        .employee-cell {
            font-weight: 600;
            background-color: #faf9f8;
            padding: 12px 8px !important;
            white-space: nowrap;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .employee-name {
            font-size: 14px;
            color: #323130;
        }
        .employee-hours {
            font-size: 11px;
            color: #605e5c;
            margin-top: 2px;
        }
        .day-cell {
            min-width: 140px;
            background-color: #ffffff;
        }
        .shift-block {
            margin: 3px 0;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.3;
            cursor: pointer;
            transition: transform 0.1s;
        }
        .shift-block:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
            color: #605e5c;
            text-align: left;
            padding: 4px 8px !important;
            background-color: #faf9f8 !important;
        }
        .date-header {
            font-size: 12px;
            color: #323130;
            font-weight: 600;
        }
        .hours-total {
            font-size: 11px;
            color: #605e5c;
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

    # Week row
    week_num = dates[0].isocalendar()[1] if dates else 0
    html_parts.append(f'<tr><th class="week-header">Week: {week_num}</th>')
    for d in dates:
        total_hrs = sum(
            len([e for e in grid_data[emp][d] if e.entry_type == "shift"]) * 8  # Rough estimate
            for emp in employees
        )
        html_parts.append(f'<th class="week-header">{total_hrs} Hrs</th>')
    html_parts.append('</tr>')

    # Date row
    html_parts.append('<tr><th class="employee-cell">Employee</th>')
    for d in dates:
        day_name = d.strftime("%a")
        date_str = d.strftime("%m/%d")
        html_parts.append(f'<th><div class="date-header">{d.day}</div><div style="font-size:10px;color:#605e5c;">{day_name}</div></th>')
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
                color_class = f"color-{entry.color_code}" if entry.color_code else "color-1"

                # Format time display
                time_str = ""
                if entry.start_time and entry.end_time:
                    # Remove seconds if present
                    start_t = entry.start_time.split(':')[0:2]
                    end_t = entry.end_time.split(':')[0:2]
                    time_str = f"{':'.join(start_t)} - {':'.join(end_t)}"

                # Determine label
                if entry.entry_type == "time_off":
                    label = entry.reason or "Time Off"
                else:
                    label = entry.label or "Shift"

                html_parts.append(f'<div class="shift-block {color_class}">')
                html_parts.append(f'<span class="shift-label">{label}</span>')
                if time_str:
                    html_parts.append(f'<div class="shift-time">{time_str}</div>')
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
