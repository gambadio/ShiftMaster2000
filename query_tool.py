"""
Query Tool for LLM-based Schedule Generation

Allows the LLM to query employee data and schedule history during generation.
This enables the LLM to ask specific questions like:
- "Get employee data for Hans Mustermann"
- "What shifts did Maria Schmidt work last week?"
- "Show me all employees with the 'Contact Team' role"
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, date, timedelta


# OpenAI Function Calling schema for Query Tool
QUERY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_schedule_data",
        "description": """Query employee data or schedule history.

Use this tool to retrieve specific information about:
- Employee details (roles, constraints, preferences, availability)
- Past shift assignments for an employee
- Time-off/vacation periods for an employee
- All employees with a specific role
- Schedule entries for a specific date range

This helps you make informed scheduling decisions based on historical data and employee constraints.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["employee_info", "employee_schedule", "employees_by_role", "schedule_by_date", "time_off"],
                    "description": """Type of query:
- employee_info: Get all details about a specific employee
- employee_schedule: Get past/future shifts for an employee
- employees_by_role: List all employees who can work a specific role
- schedule_by_date: Get all schedule entries for a date range
- time_off: Get all time-off entries for an employee or date range"""
                },
                "employee_name": {
                    "type": "string",
                    "description": "Name of the employee to query (for employee_info, employee_schedule, time_off)"
                },
                "role": {
                    "type": "string",
                    "description": "Role to filter by (for employees_by_role)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (for schedule_by_date, employee_schedule)"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (for schedule_by_date, employee_schedule)"
                }
            },
            "required": ["query_type"]
        }
    }
}


# Global storage for context data (set by the calling code)
_query_context: Dict[str, Any] = {
    "employees": [],
    "schedule_entries": [],
    "shifts": []
}


def set_query_context(employees: List[Any], schedule_entries: List[Any], shifts: List[Any] = None):
    """
    Set the context data for queries.

    Args:
        employees: List of Employee objects
        schedule_entries: List of ScheduleEntry or GeneratedScheduleEntry objects
        shifts: List of ShiftTemplate objects (optional)
    """
    global _query_context
    _query_context["employees"] = employees or []
    _query_context["schedule_entries"] = schedule_entries or []
    _query_context["shifts"] = shifts or []


def query_schedule_data(
    query_type: str,
    employee_name: Optional[str] = None,
    role: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a query against the schedule data.

    Returns:
        Dictionary with query results
    """
    try:
        if query_type == "employee_info":
            return _query_employee_info(employee_name)
        elif query_type == "employee_schedule":
            return _query_employee_schedule(employee_name, start_date, end_date)
        elif query_type == "employees_by_role":
            return _query_employees_by_role(role)
        elif query_type == "schedule_by_date":
            return _query_schedule_by_date(start_date, end_date)
        elif query_type == "time_off":
            return _query_time_off(employee_name, start_date, end_date)
        else:
            return {
                "success": False,
                "error": f"Unknown query type: {query_type}",
                "data": None
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Query failed: {str(e)}",
            "data": None
        }


def _query_employee_info(employee_name: Optional[str]) -> Dict[str, Any]:
    """Get detailed information about a specific employee."""
    if not employee_name:
        return {
            "success": False,
            "error": "employee_name is required for employee_info query",
            "data": None
        }

    employees = _query_context.get("employees", [])

    # Find employee (case-insensitive partial match)
    name_lower = employee_name.lower()
    matching = []

    for emp in employees:
        emp_name = getattr(emp, 'name', str(emp)) if hasattr(emp, 'name') else str(emp)
        if name_lower in emp_name.lower():
            # Convert to dict if it's a Pydantic model
            if hasattr(emp, 'model_dump'):
                emp_data = emp.model_dump(mode='json', exclude_none=True)
            elif hasattr(emp, '__dict__'):
                emp_data = {k: v for k, v in emp.__dict__.items() if not k.startswith('_') and v is not None}
            else:
                emp_data = {"name": str(emp)}
            matching.append(emp_data)

    if not matching:
        return {
            "success": False,
            "error": f"No employee found matching '{employee_name}'",
            "data": None,
            "available_employees": [getattr(e, 'name', str(e)) for e in employees[:20]]
        }

    return {
        "success": True,
        "data": matching[0] if len(matching) == 1 else matching,
        "match_count": len(matching)
    }


def _query_employee_schedule(
    employee_name: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str]
) -> Dict[str, Any]:
    """Get schedule entries for a specific employee."""
    if not employee_name:
        return {
            "success": False,
            "error": "employee_name is required for employee_schedule query",
            "data": None
        }

    entries = _query_context.get("schedule_entries", [])
    name_lower = employee_name.lower()

    # Parse date range
    start_dt = _parse_date(start_date) if start_date else None
    end_dt = _parse_date(end_date) if end_date else None

    matching_entries = []
    for entry in entries:
        entry_emp = getattr(entry, 'employee_name', None)
        if not entry_emp or name_lower not in entry_emp.lower():
            continue

        # Filter by date range if specified
        entry_date = _parse_date(getattr(entry, 'start_date', None))
        if entry_date:
            if start_dt and entry_date < start_dt:
                continue
            if end_dt and entry_date > end_dt:
                continue

        # Convert to dict
        if hasattr(entry, 'model_dump'):
            entry_data = entry.model_dump(mode='json', exclude_none=True)
        elif hasattr(entry, '__dict__'):
            entry_data = {k: v for k, v in entry.__dict__.items() if not k.startswith('_') and v is not None}
        else:
            entry_data = {"entry": str(entry)}
        matching_entries.append(entry_data)

    # Sort by date
    matching_entries.sort(key=lambda x: x.get('start_date', ''))

    return {
        "success": True,
        "data": matching_entries,
        "count": len(matching_entries),
        "employee": employee_name,
        "date_range": {"start": start_date, "end": end_date}
    }


def _query_employees_by_role(role: Optional[str]) -> Dict[str, Any]:
    """Get all employees who can work a specific role."""
    if not role:
        return {
            "success": False,
            "error": "role is required for employees_by_role query",
            "data": None
        }

    employees = _query_context.get("employees", [])
    role_lower = role.lower()

    matching = []
    for emp in employees:
        emp_roles = getattr(emp, 'roles', [])
        if any(role_lower in r.lower() for r in emp_roles):
            emp_name = getattr(emp, 'name', str(emp))
            emp_data = {
                "name": emp_name,
                "roles": emp_roles,
                "email": getattr(emp, 'email', None),
                "earliest_start": getattr(emp, 'earliest_start', None),
                "latest_end": getattr(emp, 'latest_end', None)
            }
            # Remove None values
            emp_data = {k: v for k, v in emp_data.items() if v is not None}
            matching.append(emp_data)

    return {
        "success": True,
        "data": matching,
        "count": len(matching),
        "role": role
    }


def _query_schedule_by_date(
    start_date: Optional[str],
    end_date: Optional[str]
) -> Dict[str, Any]:
    """Get all schedule entries for a date range."""
    if not start_date:
        return {
            "success": False,
            "error": "start_date is required for schedule_by_date query",
            "data": None
        }

    entries = _query_context.get("schedule_entries", [])

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date) if end_date else start_dt

    if not start_dt:
        return {
            "success": False,
            "error": f"Invalid start_date format: {start_date}",
            "data": None
        }

    matching_entries = []
    for entry in entries:
        entry_date = _parse_date(getattr(entry, 'start_date', None))
        if not entry_date:
            continue

        if entry_date < start_dt or entry_date > end_dt:
            continue

        # Convert to dict
        if hasattr(entry, 'model_dump'):
            entry_data = entry.model_dump(mode='json', exclude_none=True)
        elif hasattr(entry, '__dict__'):
            entry_data = {k: v for k, v in entry.__dict__.items() if not k.startswith('_') and v is not None}
        else:
            entry_data = {"entry": str(entry)}
        matching_entries.append(entry_data)

    # Sort by date and employee
    matching_entries.sort(key=lambda x: (x.get('start_date', ''), x.get('employee_name', '')))

    # Group by date for easier reading
    by_date = {}
    for entry in matching_entries:
        d = entry.get('start_date', 'unknown')
        if d not in by_date:
            by_date[d] = []
        by_date[d].append({
            "employee": entry.get('employee_name'),
            "type": entry.get('entry_type', 'shift'),
            "time": f"{entry.get('start_time', '?')}-{entry.get('end_time', '?')}",
            "label": entry.get('label') or entry.get('notes') or entry.get('reason')
        })

    return {
        "success": True,
        "data": matching_entries,
        "summary_by_date": by_date,
        "count": len(matching_entries),
        "date_range": {"start": start_date, "end": end_date}
    }


def _query_time_off(
    employee_name: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str]
) -> Dict[str, Any]:
    """Get time-off entries for an employee or date range."""
    entries = _query_context.get("schedule_entries", [])

    # Filter for time-off entries only
    time_off_entries = [
        e for e in entries
        if getattr(e, 'entry_type', None) == 'time_off'
    ]

    # Filter by employee if specified
    if employee_name:
        name_lower = employee_name.lower()
        time_off_entries = [
            e for e in time_off_entries
            if name_lower in getattr(e, 'employee_name', '').lower()
        ]

    # Filter by date range if specified
    start_dt = _parse_date(start_date) if start_date else None
    end_dt = _parse_date(end_date) if end_date else None

    if start_dt or end_dt:
        filtered = []
        for entry in time_off_entries:
            entry_start = _parse_date(getattr(entry, 'start_date', None))
            entry_end = _parse_date(getattr(entry, 'end_date', None)) or entry_start

            if not entry_start:
                continue

            # Check if entry overlaps with the date range
            if start_dt and entry_end and entry_end < start_dt:
                continue
            if end_dt and entry_start > end_dt:
                continue

            filtered.append(entry)
        time_off_entries = filtered

    # Convert to dict and format nicely
    results = []
    for entry in time_off_entries:
        if hasattr(entry, 'model_dump'):
            entry_data = entry.model_dump(mode='json', exclude_none=True)
        elif hasattr(entry, '__dict__'):
            entry_data = {k: v for k, v in entry.__dict__.items() if not k.startswith('_') and v is not None}
        else:
            entry_data = {"entry": str(entry)}

        # Add a human-readable summary
        emp = entry_data.get('employee_name', 'Unknown')
        start = entry_data.get('start_date', '?')
        end = entry_data.get('end_date', start)
        reason = entry_data.get('reason', 'Time Off')
        entry_data['_summary'] = f"{emp}: {reason} from {start} to {end}"

        results.append(entry_data)

    # Sort by start date
    results.sort(key=lambda x: x.get('start_date', ''))

    return {
        "success": True,
        "data": results,
        "count": len(results),
        "filters": {
            "employee": employee_name,
            "start_date": start_date,
            "end_date": end_date
        }
    }


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse a date string in various formats."""
    if not date_str:
        return None

    # Handle date objects directly
    if isinstance(date_str, date):
        return date_str

    # Try various formats
    formats = [
        "%Y-%m-%d",      # 2024-01-15
        "%m/%d/%Y",      # 1/15/2024
        "%d.%m.%Y",      # 15.01.2024
        "%d/%m/%Y",      # 15/01/2024
    ]

    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue

    return None


def process_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a tool call from the LLM and return the result.

    Args:
        tool_call: The tool call object from OpenAI API

    Returns:
        Tool result to send back to the LLM
    """
    try:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")

        if function_name != "query_schedule_data":
            return {
                "success": False,
                "error": f"Unknown function: {function_name}"
            }

        # Parse arguments
        args_str = function.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON arguments: {e}"
            }

        # Execute the query
        return query_schedule_data(
            query_type=args.get("query_type", ""),
            employee_name=args.get("employee_name"),
            role=args.get("role"),
            start_date=args.get("start_date"),
            end_date=args.get("end_date")
        )

    except Exception as e:
        return {
            "success": False,
            "error": f"Tool call processing error: {str(e)}"
        }
