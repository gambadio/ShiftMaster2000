"""
Microsoft Teams Shifts Export

Exports schedule entries to Teams-compatible Excel files
"""

from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from models import ScheduleEntry, TEAMS_COLOR_NAMES


def export_to_teams_excel(
    schedule_entries: List[ScheduleEntry],
    output_shifts: str,
    output_timeoff: str
) -> None:
    """
    Export schedule entries to two Teams-compatible Excel files

    Args:
        schedule_entries: List of ScheduleEntry objects
        output_shifts: Path to shifts Excel file
        output_timeoff: Path to time-off Excel file
    """
    shifts = []
    timeoffs = []

    for entry in schedule_entries:
        if entry.entry_type == "shift":
            shifts.append(entry)
        elif entry.entry_type == "time_off":
            timeoffs.append(entry)

    # Export shifts
    if shifts:
        _export_shifts_file(shifts, output_shifts)

    # Export time-off
    if timeoffs:
        _export_timeoff_file(timeoffs, output_timeoff)


def _export_shifts_file(shifts: List[ScheduleEntry], output_path: str) -> None:
    """Export shifts to Teams-compatible Excel"""
    rows = []

    for shift in shifts:
        # Format dates and times for Teams (M/D/YYYY and HH:MM)
        start_date_obj = datetime.fromisoformat(shift.start_date)
        end_date_obj = datetime.fromisoformat(shift.end_date)

        start_date_str = start_date_obj.strftime("%-m/%-d/%Y")  # M/D/YYYY
        end_date_str = end_date_obj.strftime("%-m/%-d/%Y")

        # Get color name with number
        color_display = _format_color_code(shift.color_code)

        row = {
            "Mitglied": shift.employee_name,
            "E-Mail (geschäftlich)": shift.employee_email or "",
            "Gruppe": shift.group or "Service Desk",
            "Startdatum": start_date_str,
            "Startzeit": shift.start_time or "00:00",
            "Enddatum": end_date_str,
            "Endzeit": shift.end_time or "00:00",
            "Themenfarbe": color_display,
            "Bezeichnung": shift.label or "",
            "Unbezahlte Pause (Minuten)": shift.unpaid_break or "",
            "Notizen": shift.notes or "",
            "Geteilt": shift.shared or "1. Geteilt",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, engine='openpyxl')


def _export_timeoff_file(timeoffs: List[ScheduleEntry], output_path: str) -> None:
    """Export time-off to Teams-compatible Excel"""
    rows = []

    for entry in timeoffs:
        # Format dates for Teams (M/D/YYYY)
        start_date_obj = datetime.fromisoformat(entry.start_date)
        end_date_obj = datetime.fromisoformat(entry.end_date)

        start_date_str = start_date_obj.strftime("%-m/%-d/%Y")
        end_date_str = end_date_obj.strftime("%-m/%-d/%Y")

        # Time-off typically uses "00:00" for times or blank
        start_time = entry.start_time if entry.start_time else "00:00"
        end_time = entry.end_time if entry.end_time else "00:00"

        # Get color name (usually "13. Grau" for time-off)
        color_display = _format_color_code(entry.color_code or "13")

        row = {
            "Mitglied": entry.employee_name,
            "E-Mail (geschäftlich)": entry.employee_email or "",
            "Startdatum": start_date_str,
            "Startzeit": start_time,
            "Enddatum": end_date_str,
            "Endzeit": end_time,
            "Grund für arbeitsfreie Zeit": entry.reason or "Ferien",
            "Themenfarbe": color_display,
            "Notizen": entry.notes or "",
            "Geteilt": entry.shared or "1. Geteilt",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, engine='openpyxl')


def _format_color_code(color_code: str | None) -> str:
    """Format color code to Teams format (e.g., '1' -> '1. Weiß')"""
    if not color_code or color_code not in TEAMS_COLOR_NAMES:
        return "1. Weiß"  # Default to white

    color_name = TEAMS_COLOR_NAMES[color_code]
    return f"{color_code}. {color_name}"


def schedule_entries_from_llm_output(llm_output: Dict[str, Any]) -> List[ScheduleEntry]:
    """
    Convert LLM-generated schedule JSON to ScheduleEntry objects

    Expected LLM output format:
    {
        "dates": ["2025-10-16", "2025-10-17", ...],
        "assignments": [
            {
                "date": "2025-10-16",
                "employee_name": "John Doe",
                "employee_email": "john.doe@example.com",
                "role": "Contact Team",
                "shift_id": "contact-0700",
                "start_time": "07:00",
                "end_time": "16:00",
                "color_code": "2",
                "notes": "..."
            },
            ...
        ],
        "violations": [...],
        "notes": "..."
    }
    """
    entries = []

    assignments = llm_output.get("assignments", [])

    for assignment in assignments:
        entry = ScheduleEntry(
            employee_name=assignment.get("employee_name", ""),
            employee_email=assignment.get("employee_email"),
            group=assignment.get("group", "Service Desk"),
            start_date=assignment.get("date", ""),
            start_time=assignment.get("start_time"),
            end_date=assignment.get("date", ""),  # Same day for shifts
            end_time=assignment.get("end_time"),
            color_code=assignment.get("color_code"),
            label=assignment.get("role", ""),
            unpaid_break=assignment.get("unpaid_break"),
            notes=assignment.get("notes"),
            shared="1. Geteilt",
            entry_type="shift",
            reason=None,
        )
        entries.append(entry)

    return entries


def create_sample_export():
    """Create a sample export for testing"""
    from datetime import date

    sample_shifts = [
        ScheduleEntry(
            employee_name="Doe, John-MGB",
            employee_email="john.doe@mgb.ch",
            group="Service Desk",
            start_date="2025-10-16",
            start_time="07:00",
            end_date="2025-10-16",
            end_time="16:00",
            color_code="2",
            label="Contact Team",
            unpaid_break=None,
            notes="",
            shared="1. Geteilt",
            entry_type="shift"
        ),
        ScheduleEntry(
            employee_name="Smith, Jane-MGB",
            employee_email="jane.smith@mgb.ch",
            group="Service Desk",
            start_date="2025-10-16",
            start_time="08:00",
            end_date="2025-10-16",
            end_time="17:00",
            color_code="3",
            label="SOB Wove",
            unpaid_break=None,
            notes="",
            shared="1. Geteilt",
            entry_type="shift"
        ),
    ]

    sample_timeoff = [
        ScheduleEntry(
            employee_name="Doe, John-MGB",
            employee_email="john.doe@mgb.ch",
            group="Service Desk",
            start_date="2025-10-20",
            start_time="00:00",
            end_date="2025-10-22",
            end_time="00:00",
            color_code="13",
            label=None,
            unpaid_break=None,
            notes="Vacation",
            shared="1. Geteilt",
            entry_type="time_off",
            reason="Ferien"
        ),
    ]

    export_to_teams_excel(
        sample_shifts + sample_timeoff,
        "sample_shifts.xlsx",
        "sample_timeoff.xlsx"
    )
    print("Sample export created: sample_shifts.xlsx, sample_timeoff.xlsx")


if __name__ == "__main__":
    create_sample_export()
