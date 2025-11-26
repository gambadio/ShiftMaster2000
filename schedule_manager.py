"""
Schedule Manager: Core logic for schedule entry management and conflict detection
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

from models import (
    GeneratedScheduleEntry, ScheduleConflict, ConflictType,
    ScheduleState, Employee, ShiftTemplate, Project
)


class ScheduleManager:
    """Manages schedule entries, conflict detection, and validation"""

    def __init__(self, project: Project):
        self.project = project
        if not self.project.schedule_state:
            self.project.schedule_state = ScheduleState()

    @property
    def state(self) -> ScheduleState:
        """Get schedule state"""
        return self.project.schedule_state

    def add_generated_entries(self, entries: List[GeneratedScheduleEntry]) -> None:
        """Add newly generated entries and detect conflicts"""
        # Assign proper IDs
        for entry in entries:
            if not entry.id or entry.id.startswith("gen_"):
                entry.id = f"gen_{uuid.uuid4().hex[:8]}"
            entry.source = "generated"

        # Add to state
        self.state.generated_entries.extend(entries)
        self.state.last_generated = datetime.now().isoformat()

        # Detect conflicts - TEMPORARILY DISABLED
        # self.detect_all_conflicts()

    def add_uploaded_entries(self, entries: List[GeneratedScheduleEntry], members: List[Dict[str, str]] = None) -> None:
        """Add uploaded entries from Excel file"""
        # Assign proper IDs
        for entry in entries:
            if not entry.id or not entry.id.startswith("upl_"):
                entry.id = f"upl_{uuid.uuid4().hex[:8]}"
            entry.source = "uploaded"

        # Add to state
        self.state.uploaded_entries = entries
        if members:
            self.state.uploaded_members = members

        # Detect conflicts - TEMPORARILY DISABLED
        # self.detect_all_conflicts()

    def get_all_entries(self) -> List[GeneratedScheduleEntry]:
        """Get all entries (uploaded + generated)"""
        return self.state.uploaded_entries + self.state.generated_entries

    def get_entries_by_date_range(self, start_date: str, end_date: str) -> List[GeneratedScheduleEntry]:
        """Get entries within date range"""
        all_entries = self.get_all_entries()
        filtered = []

        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)

        for entry in all_entries:
            entry_start = self._parse_date(entry.start_date)
            entry_end = self._parse_date(entry.end_date)

            # Check if entry overlaps with date range
            if entry_end >= start_dt and entry_start <= end_dt:
                filtered.append(entry)

        return filtered

    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entry"""
        entry = self.get_entry_by_id(entry_id)
        if not entry:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        entry.is_modified = True

        # Re-detect conflicts
        self.detect_all_conflicts()
        return True

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry"""
        # Try generated entries
        for i, entry in enumerate(self.state.generated_entries):
            if entry.id == entry_id:
                del self.state.generated_entries[i]
                self.detect_all_conflicts()
                return True

        # Try uploaded entries
        for i, entry in enumerate(self.state.uploaded_entries):
            if entry.id == entry_id:
                del self.state.uploaded_entries[i]
                self.detect_all_conflicts()
                return True

        return False

    def clear_generated(self) -> None:
        """Clear all generated entries"""
        self.state.generated_entries = []
        self.state.last_generated = None
        self.state.generation_notes = None
        self.detect_all_conflicts()

    def clear_uploaded(self) -> None:
        """Clear all uploaded entries"""
        self.state.uploaded_entries = []
        self.state.uploaded_members = []
        self.detect_all_conflicts()

    def get_entry_by_id(self, entry_id: str) -> Optional[GeneratedScheduleEntry]:
        """Get entry by ID"""
        for entry in self.get_all_entries():
            if entry.id == entry_id:
                return entry
        return None

    def detect_all_conflicts(self) -> List[ScheduleConflict]:
        """Detect all conflicts and update state"""
        conflicts = []

        # Get all entries
        all_entries = self.get_all_entries()

        # Clear existing conflict flags
        for entry in all_entries:
            entry.has_conflict = False
            entry.conflict_ids = []

        # Detect overlaps (same employee, overlapping times)
        conflicts.extend(self._detect_overlaps(all_entries))

        # Detect time-off conflicts (shift during time-off)
        conflicts.extend(self._detect_time_off_conflicts(all_entries))

        # Detect staffing issues
        conflicts.extend(self._detect_staffing_issues(all_entries))

        # Detect constraint violations
        conflicts.extend(self._detect_constraint_violations(all_entries))

        # Update state
        self.state.conflicts = conflicts

        # Mark entries with conflicts
        for conflict in conflicts:
            for entry_id in conflict.entry_ids:
                entry = self.get_entry_by_id(entry_id)
                if entry:
                    entry.has_conflict = True
                    entry.conflict_ids.append(conflict.id)

        return conflicts

    def _detect_overlaps(self, entries: List[GeneratedScheduleEntry]) -> List[ScheduleConflict]:
        """Detect overlapping shifts for same employee"""
        conflicts = []

        # Only check shifts (not time-off)
        shifts = [e for e in entries if e.entry_type == "shift"]

        # Group by employee
        by_employee = defaultdict(list)
        for shift in shifts:
            by_employee[shift.employee_name].append(shift)

        # Check each employee's shifts for overlaps
        for employee_name, emp_shifts in by_employee.items():
            # Sort by start time
            sorted_shifts = sorted(emp_shifts, key=lambda s: (s.start_date, s.start_time))

            for i in range(len(sorted_shifts)):
                for j in range(i + 1, len(sorted_shifts)):
                    shift1 = sorted_shifts[i]
                    shift2 = sorted_shifts[j]

                    if self._times_overlap(
                        shift1.start_date, shift1.start_time, shift1.end_date, shift1.end_time,
                        shift2.start_date, shift2.start_time, shift2.end_date, shift2.end_time
                    ):
                        conflict = ScheduleConflict(
                            id=f"overlap_{uuid.uuid4().hex[:8]}",
                            conflict_type=ConflictType.OVERLAP,
                            severity="error",
                            message=f"{employee_name} has overlapping shifts on {shift1.start_date}",
                            entry_ids=[shift1.id, shift2.id],
                            date=shift1.start_date,
                            employee_name=employee_name
                        )
                        conflicts.append(conflict)

        return conflicts

    def _detect_time_off_conflicts(self, entries: List[GeneratedScheduleEntry]) -> List[ScheduleConflict]:
        """Detect shifts assigned during time-off"""
        conflicts = []

        shifts = [e for e in entries if e.entry_type == "shift"]
        time_offs = [e for e in entries if e.entry_type == "time_off"]

        # Group by employee
        for shift in shifts:
            employee_name = shift.employee_name

            # Check if shift overlaps with any time-off for this employee
            for time_off in time_offs:
                if time_off.employee_name == employee_name:
                    if self._times_overlap(
                        shift.start_date, shift.start_time, shift.end_date, shift.end_time,
                        time_off.start_date, "00:00", time_off.end_date, "23:59"
                    ):
                        conflict = ScheduleConflict(
                            id=f"timeoff_{uuid.uuid4().hex[:8]}",
                            conflict_type=ConflictType.TIME_OFF_CONFLICT,
                            severity="error",
                            message=f"{employee_name} has a shift during time-off ({time_off.reason or 'time-off'})",
                            entry_ids=[shift.id, time_off.id],
                            date=shift.start_date,
                            employee_name=employee_name
                        )
                        conflicts.append(conflict)

        return conflicts

    def _detect_staffing_issues(self, entries: List[GeneratedScheduleEntry]) -> List[ScheduleConflict]:
        """Detect under/over staffing for shift requirements"""
        conflicts = []

        # Get generated shifts only
        generated_shifts = [e for e in entries if e.entry_type == "shift" and e.source == "generated"]

        # Group by date and role/label
        by_date_role = defaultdict(list)
        for shift in generated_shifts:
            key = (shift.start_date, shift.label or "Unknown Role")
            by_date_role[key].append(shift)

        # Check against shift templates
        for shift_template in self.project.shifts:
            for date_str in self._get_planning_dates():
                weekday = self._get_weekday(date_str)
                weekday_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday]

                # Check if this shift template applies to this weekday
                if weekday_short not in shift_template.weekdays:
                    continue

                required_count = shift_template.required_count.get(weekday_short, 0)
                if required_count == 0:
                    continue

                # Count assigned staff
                key = (date_str, shift_template.role)
                assigned = by_date_role.get(key, [])
                assigned_count = len(assigned)

                if assigned_count < required_count:
                    conflict = ScheduleConflict(
                        id=f"understaffed_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.UNDERSTAFFED,
                        severity="warning",
                        message=f"{shift_template.role} on {date_str}: {assigned_count}/{required_count} assigned",
                        entry_ids=[e.id for e in assigned],
                        date=date_str,
                        shift_role=shift_template.role
                    )
                    conflicts.append(conflict)
                elif assigned_count > required_count:
                    conflict = ScheduleConflict(
                        id=f"overstaffed_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.OVERSTAFFED,
                        severity="info",
                        message=f"{shift_template.role} on {date_str}: {assigned_count}/{required_count} assigned (overstaffed)",
                        entry_ids=[e.id for e in assigned],
                        date=date_str,
                        shift_role=shift_template.role
                    )
                    conflicts.append(conflict)

        return conflicts

    def _detect_constraint_violations(self, entries: List[GeneratedScheduleEntry]) -> List[ScheduleConflict]:
        """Detect employee constraint violations"""
        conflicts = []

        shifts = [e for e in entries if e.entry_type == "shift" and e.source == "generated"]

        for shift in shifts:
            # Find employee
            employee = self._find_employee(shift.employee_name, shift.employee_email)
            if not employee:
                continue

            # Check role assignment
            if shift.label and employee.roles:
                if shift.label not in employee.roles:
                    conflict = ScheduleConflict(
                        id=f"role_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.MISSING_ROLE,
                        severity="warning",
                        message=f"{shift.employee_name} assigned to {shift.label} but doesn't have this role",
                        entry_ids=[shift.id],
                        date=shift.start_date,
                        employee_name=shift.employee_name,
                        shift_role=shift.label
                    )
                    conflicts.append(conflict)

            # Check time constraints
            if employee.earliest_start:
                if shift.start_time < employee.earliest_start:
                    conflict = ScheduleConflict(
                        id=f"time_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.CONSTRAINT_VIOLATION,
                        severity="warning",
                        message=f"{shift.employee_name} starts at {shift.start_time}, before earliest start {employee.earliest_start}",
                        entry_ids=[shift.id],
                        date=shift.start_date,
                        employee_name=shift.employee_name
                    )
                    conflicts.append(conflict)

            if employee.latest_end:
                if shift.end_time > employee.latest_end:
                    conflict = ScheduleConflict(
                        id=f"time_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.CONSTRAINT_VIOLATION,
                        severity="warning",
                        message=f"{shift.employee_name} ends at {shift.end_time}, after latest end {employee.latest_end}",
                        entry_ids=[shift.id],
                        date=shift.start_date,
                        employee_name=shift.employee_name
                    )
                    conflicts.append(conflict)

        return conflicts

    def _find_employee(self, name: str, email: Optional[str] = None) -> Optional[Employee]:
        """Find employee by name or email"""
        for emp in self.project.employees:
            if emp.name == name:
                return emp
            if email and emp.email == email:
                return emp
        return None

    def _times_overlap(
        self, start1: str, time1: str, end1: str, time2: str,
        start2: str, time3: str, end2: str, time4: str
    ) -> bool:
        """Check if two time periods overlap"""
        try:
            dt1_start = self._parse_datetime(start1, time1)
            dt1_end = self._parse_datetime(end1, time2)
            dt2_start = self._parse_datetime(start2, time3)
            dt2_end = self._parse_datetime(end2, time4)

            return dt1_start < dt2_end and dt2_start < dt1_end
        except:
            return False

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string (M/D/YYYY or YYYY-MM-DD)"""
        # Try M/D/YYYY format first
        for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")

    def _parse_datetime(self, date_str: str, time_str: str) -> datetime:
        """Parse date and time strings"""
        date_dt = self._parse_date(date_str)
        try:
            time_parts = time_str.split(":")
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            return date_dt.replace(hour=hour, minute=minute)
        except:
            return date_dt

    def _get_weekday(self, date_str: str) -> int:
        """Get weekday (0=Mon, 6=Sun)"""
        dt = self._parse_date(date_str)
        return dt.weekday()

    def _get_planning_dates(self) -> List[str]:
        """Get all dates in planning period"""
        if not self.project.planning_period:
            return []

        dates = []
        current = datetime.combine(self.project.planning_period.start_date, datetime.min.time())
        end = datetime.combine(self.project.planning_period.end_date, datetime.min.time())

        while current <= end:
            dates.append(current.strftime("%m/%d/%Y"))
            current += timedelta(days=1)

        return dates


def parse_llm_schedule_output(llm_output: Dict[str, Any], project: Optional[Any] = None) -> Tuple[List[GeneratedScheduleEntry], str, List[str]]:
    """Parse LLM output into schedule entries.
    
    Handles both full format and compact token-optimized format.
    
    Full format: {"shifts": [...], "notes": "..."}
    Compact format: {"s": [...], "n": "..."}

    Returns (entries, notes, errors) so the UI can surface detailed diagnostics.
    """
    entries: List[GeneratedScheduleEntry] = []
    errors: List[str] = []
    
    # Detect format and normalize to full format
    normalized = _normalize_llm_output(llm_output, project)
    
    notes = normalized.get("notes", "") or normalized.get("generation_notes", "")

    # Extract shifts
    shifts = normalized.get("shifts", [])
    for idx, shift_data in enumerate(shifts):
        try:
            cleaned = _clean_generated_payload(shift_data)
            entry = GeneratedScheduleEntry(**cleaned)
            entry.entry_type = "shift"
            entries.append(entry)
        except Exception as e:
            errors.append(f"Shift {idx + 1}: {e}")

    # Extract time-off if present
    time_offs = normalized.get("time_off", [])
    for idx, time_off_data in enumerate(time_offs):
        try:
            cleaned = _clean_generated_payload(time_off_data)
            entry = GeneratedScheduleEntry(**cleaned)
            entry.entry_type = "time_off"
            entries.append(entry)
        except Exception as e:
            errors.append(f"Time-off {idx + 1}: {e}")

    return entries, notes, errors


def _normalize_llm_output(llm_output: Dict[str, Any], project: Optional[Any] = None) -> Dict[str, Any]:
    """
    Normalize LLM output from compact or full format to standard full format.
    
    Compact format:
    {
        "s": [{"e": "Name", "d": "2025-01-01", "st": "07:00", "et": "16:00", "c": "1. Weiß", "l": "Op Lead", "r": "Role"}],
        "n": "notes"
    }
    
    Full format:
    {
        "shifts": [{"employee_name": "Name", "start_date": "1/1/2025", ...}],
        "notes": "notes"
    }
    """
    # Already in full format?
    if "shifts" in llm_output:
        return llm_output
    
    # Compact format - expand it
    if "s" not in llm_output:
        return llm_output  # Unknown format, return as-is
    
    # Build employee lookup for missing emails/groups
    emp_lookup: Dict[str, Dict[str, Any]] = {}
    if project:
        for emp in project.employees:
            emp_lookup[emp.name.lower()] = {
                "email": emp.email,
                "group": emp.group or "Service Desk"
            }
    
    shifts = []
    for cs in llm_output.get("s", []):
        # Get employee info
        emp_name = cs.get("e", "")
        emp_info = emp_lookup.get(emp_name.lower(), {})
        
        # Parse date - handle YYYY-MM-DD format and convert to M/D/YYYY
        date_str = cs.get("d", "")
        try:
            from datetime import datetime
            # Try ISO format first (YYYY-MM-DD)
            if "-" in date_str and len(date_str) == 10:
                d = datetime.fromisoformat(date_str).date()
                formatted_date = f"{d.month}/{d.day}/{d.year}"
            else:
                formatted_date = date_str
        except:
            formatted_date = date_str
        
        full_entry = {
            "employee_name": emp_name,
            "employee_email": cs.get("m") or emp_info.get("email", ""),
            "group": cs.get("g") or emp_info.get("group", "Service Desk"),
            "start_date": formatted_date,
            "start_time": cs.get("st", ""),
            "end_date": formatted_date,
            "end_time": cs.get("et", ""),
            "color_code": _normalize_color_code(cs.get("c", "1. Weiß")),
            "label": cs.get("l", ""),
            "unpaid_break": cs.get("b"),
            "notes": cs.get("r", ""),
            "shared": "1. Geteilt"
        }
        shifts.append(full_entry)
    
    return {
        "shifts": shifts,
        "notes": llm_output.get("n", "")
    }


def _normalize_color_code(color: str) -> str:
    """Ensure color_code has proper format like '1. Weiß'"""
    if not color:
        return "1. Weiß"
    
    # Already in correct format?
    if ". " in str(color):
        return color
    
    # Just a number - add the German name
    color_names = {
        "1": "1. Weiß", "2": "2. Blau", "3": "3. Grün", "4": "4. Lila",
        "5": "5. Rosa", "6": "6. Gelb", "8": "8. Dunkelblau", "9": "9. Dunkelgrün",
        "10": "10. Dunkelviolett", "11": "11. Dunkelrosa", "12": "12. Dunkelgelb", "13": "13. Grau"
    }
    return color_names.get(str(color).strip(), "1. Weiß")


def _clean_generated_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Light normalization (strip whitespace) for raw LLM payload dicts."""
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            cleaned[key] = value.strip()
        else:
            cleaned[key] = value
    return cleaned
