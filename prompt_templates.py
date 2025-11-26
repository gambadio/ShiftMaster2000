from __future__ import annotations
from typing import Dict, Any, Optional
import json
from datetime import timedelta
from models import Project, TEAMS_COLOR_NAMES

def json_block(data: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

TEAMS_COLOR_SPEC = """
### Microsoft Teams Shift Colors
Use these color codes when formatting output. Each shift should be assigned an appropriate color:

1. Weiß (White) - Operation Lead
2. Blau (Blue) - Contact Team, Dispatcher (07:00-16:00)
3. Grün (Green) - Contact Team, SOB roles, SOB Wove
4. Lila (Purple) - Late shifts (10:00-19:00)
5. Rosa (Pink) - Special assignments (e.g., Techbar)
6. Gelb (Yellow) - Late shifts (09:00-18:00), Dispatcher
8. Dunkelblau (Dark Blue) - Project work (e.g., M-Industrie Projekt)
9. Dunkelgrün (Dark Green) - WoVe, PCV roles
10. Dunkelviolett (Dark Purple) - Pikett (on-call duty)
11. Dunkelrosa (Dark Pink) - People Developer, Stellvertretung
12. Dunkelgelb (Dark Yellow) - Livechat shifts
13. Grau (Gray) - Time-off, holidays, sick leave

**Note:** Color 7 is not used. When assigning shifts, include the color_code (1-13) in your output.
"""

SYSTEM_TEMPLATE = """\
{preamble}

## YOUR TASK: ONE-SHOT COMPLETE SCHEDULE GENERATION

You are generating a work schedule in a SINGLE, COMPLETE response. This is NOT a conversation.

DO NOT:
- Ask if the user wants more details
- Say "let me know if you want me to continue"
- Provide partial results and wait for follow-up
- Summarize or abbreviate the output
- Be conversational or chatty

DO:
- Generate ALL shifts for ALL employees for ALL days in the planning period
- Output the COMPLETE JSON structure with every single shift assignment
- Include notes about constraints/violations in the top-level "n" field
- Treat this as a one-shot batch job that produces complete output

{teams_color_spec}

Data dictionary:
- employees[]: people and their capabilities, languages, constraints, and email addresses
- shifts[]: templates describing role, start/end times, weekdays, headcount, and Teams color codes
- meta: version, project name
{planning_period_context}

Rules:
{narrative_rules}

## OUTPUT FORMAT (Token-Optimized)

Return a JSON object with this structure:

```json
{{
  "s": [
    {{
      "e": "Lastname, Firstname-MGB",
      "d": "2025-10-01",
      "st": "07:00",
      "et": "16:00",
      "c": "1. Weiß",
      "n": "Operation Lead"
    }}
  ],
  "x": "Schedule notes, violations, explanations"
}}
```

**Field Key:**
- `s`: shifts array (REQUIRED)
- `e`: employee_name (full name from employees list)
- `d`: date (YYYY-MM-DD)
- `st`: start_time (HH:MM)
- `et`: end_time (HH:MM)
- `c`: color_code ("1. Weiß" to "13. Grau")
- `n`: notes (role/function for the shift, e.g., "Operation Lead", "Dispatcher")
- `b`: unpaid_break minutes (optional, omit if 0)
- `x`: top-level schedule notes/explanations

**Color Codes:**
1=Weiß (Op Lead), 2=Blau (Contact/Dispatcher 07:00), 3=Grün (SOB), 4=Lila (Late 10:00-19:00),
5=Rosa (Special), 6=Gelb (Late 09:00-18:00), 8=Dunkelblau (Project), 9=Dunkelgrün (WoVe),
10=Dunkelviolett (Pikett), 11=Dunkelrosa (People Dev), 12=Dunkelgelb (Livechat), 13=Grau (Time-off)

**CRITICAL**: Output ONLY valid JSON. Generate the COMPLETE schedule in ONE response.
"""

SCHEDULE_ADDENDUM_TEMPLATE = """\
## Schedule History (Condensed Format)

Use this data to enforce rotation fairness and respect unavailability. Model date: **{today}** (Europe/Zurich).

**Format Key:**
- `p`: past entries, `f`: future entries (unavailability)
- `e`: employee name
- `s`: shifts (list of {{r: role, t: time_range, d: dates}})
- `o`: time-off (list of {{r: reason, d: dates}})
- Dates: "YYYY-MM-DD" or "start:end" for consecutive ranges

```json
{schedule_json}
```

**Guidance:**
- Past = history for fairness (don't repeat same late/pikett assignments)
- Future = hard unavailability (unless labeled "tentative")
- Date ranges like "2025-01-01:2025-01-05" mean all days inclusive
"""

def build_system_prompt(
    project: Project,
    schedule_payload: Optional[Dict[str, Any]] = None,
    today_iso: Optional[str] = None,
    planning_period: Optional[tuple[str, str]] = None
) -> str:
    from utils import condense_schedule_payload
    
    data = project.as_compact_json()

    # Build planning period context
    planning_context = ""
    if planning_period:
        start_date, end_date = planning_period
        from datetime import datetime
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        num_days = (end - start).days + 1
        weekdays = []
        current = start
        while current <= end:
            weekdays.append(current.strftime("%A"))
            current += timedelta(days=1)

        planning_context = f"""
### Planning Period
Generate schedule for: **{start_date}** to **{end_date}** (inclusive)
Total days: {num_days}
Weekdays included: {', '.join(set(weekdays))}
"""

    sys = SYSTEM_TEMPLATE.format(
        preamble=project.global_rules.preamble.strip(),
        teams_color_spec=TEAMS_COLOR_SPEC,
        narrative_rules=project.global_rules.narrative_rules.strip(),
        ofmt=project.global_rules.output_format_instructions.strip(),
        planning_period_context=planning_context
    )
    compiled = sys + "\n\nData:\n" + json_block(data)

    if schedule_payload:
        # Use condensed format to save tokens
        condensed = condense_schedule_payload(schedule_payload)
        
        addendum = SCHEDULE_ADDENDUM_TEMPLATE.format(
            today=today_iso or "unknown",
            schedule_json=json.dumps(condensed, ensure_ascii=False, separators=(',', ':'))  # No indentation, compact JSON
        )
        compiled += "\n\n" + addendum

    return compiled
