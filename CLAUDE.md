# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Required Reading

This project maintains two critical documentation files that **you MUST consult** before making code changes:

### `projectstructure.md` - Project Structure Reference
**Always consult this file first** when:
- Locating specific functionality or understanding code organization
- Planning changes that affect multiple files or components
- Understanding data flows and integration points
- Finding exact line numbers for functions, classes, or configuration

The structure file provides a comprehensive map of the entire codebase with exact line numbers, making it the fastest way to understand where code lives and how components interact.

### `knowledge.md` - Technical Knowledge Base
**Consult this file** when working on tasks related to:
- External APIs and their integration patterns
- Framework-specific conventions and best practices
- Architectural decisions and design patterns
- Library usage and configuration
- Any technical specifications documented therein

**Workflow:** Check `projectstructure.md` first to find where code lives, then consult `knowledge.md` for implementation guidance.

---

## Project Overview

**AI Shift Studio** is a Streamlit application for AI-powered shift planning with Microsoft Teams integration. It manages employees, shift templates, scheduling rules, and uses LLMs to generate optimized schedules that can be directly imported into Microsoft Teams Shifts.

## Development Setup

### Environment & Dependencies

The project uses a Python virtual environment located at `.venv/`.

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will launch in a browser at `http://localhost:8501`.

## Architecture

### Core Data Models (`models.py`)

The application is built around Pydantic models that enforce structure and validation:

- **`Project`**: Top-level container holding employees, shifts, rules, and metadata
- **`Employee`**: Represents staff with roles, availability, constraints (hard/soft), languages, time windows, and weekday blockers
- **`ShiftTemplate`**: Defines recurring shift patterns with role, time range, weekdays, and per-weekday required headcount
- **`RuleSet`**: Contains system preamble, narrative rules, and output format instructions for LLM prompts

All models use `model_config = ConfigDict(extra="ignore")` where applicable to gracefully handle extra fields in JSON files.

### Application Flow (`app.py`)

The Streamlit app is organized into 6 tabs:

1. **Employees**: CRUD interface for managing employee records with multiselect for roles (inferred from shifts and existing employees)
2. **Shifts & Roles**: Define shift templates with per-weekday headcount requirements
3. **Rules & Preamble**: Configure the system prompt text, narrative rules, and output format
4. **Schedule File (Optional)**: Upload CSV/Excel files containing past schedules and future availability
5. **Compile & Export**: Generate the final system prompt from all configured data
6. **Test Run (LLM)**: Send compiled prompts to OpenAI-compatible endpoints for testing

**Session State Management:**
- `st.session_state.project`: Holds the current `Project` instance throughout the session
- `st.session_state.schedule_payload`: Parsed schedule data (past/future entries + fairness hints)
- `st.session_state.editing_employee_id` / `editing_shift_id`: Track which entity is being edited

### Utilities (`utils.py`)

**Project I/O:**
- `save_project()` / `load_project_dict()`: Serialize/deserialize projects to/from JSON

**Prompt Generation:**
- `compile_prompt()`: Orchestrates prompt assembly by calling `build_system_prompt()` from `prompt_templates.py`

**LLM Integration:**
- `call_llm()`: Makes OpenAI-compatible API calls with optional JSON mode

**Schedule Parsing:**
- `parse_schedule_to_payload()`: Ingests CSV/Excel files, normalizes column names, expands date ranges, and splits entries into past/future based on "today"
- `_compute_fairness_hints()`: Extracts lightweight rotation fairness metrics (late shifts, Pikett duty) from the last 14 days
- Column name matching is flexible and supports multiple languages (e.g., `DATE_COL_CANDIDATES`, `EMP_COL_CANDIDATES`)

### Prompt Templates (`prompt_templates.py`)

- `SYSTEM_TEMPLATE`: Base prompt structure injecting preamble, rules, and output format
- `SCHEDULE_ADDENDUM_TEMPLATE`: Optional section appended when a schedule file is included
- `build_system_prompt()`: Combines templates with project data and schedule payload into a final markdown-formatted prompt

## Key Design Patterns

### Role Inference
Roles are dynamically inferred from:
1. Roles defined in shift templates (`ShiftTemplate.role`)
2. Roles assigned to employees (`Employee.roles`)

This is cached in `st.session_state.role_options` and used to populate multiselect dropdowns.

### Microsoft Teams Shifts Integration
The application now fully supports importing and exporting Microsoft Teams Shifts data in two modes:

**Single-File Import (Multi-Sheet Excel):**
- Upload one complete Teams export Excel file containing all sheets
- `parse_teams_excel_multisheet()` in `utils.py` automatically detects and parses:
  - **Schichten** (Shifts): Actual shift assignments with dates, times, roles, color codes
  - **Arbeitsfreie Zeit** (Time-Off): Vacation, sick leave, compensation days, etc.
  - **Mitglieder** (Members): Employee list with names and email addresses
- Flexible sheet name detection (case-insensitive, supports German/English variants)
- Extracts member data for validation and export reuse
- Single function call processes all relevant data from the multi-sheet file

**Dual-File Import (Separate Files):**
- Supports two separate file uploads: shifts file and time-off file
- `parse_dual_schedule_files()` combines both files into a unified payload
- Useful when shifts and time-off are exported as separate files

**Common Import Features:**
- Parses German column names from Teams exports (Mitglied, Startdatum, Endzeit, Themenfarbe, etc.)
- Automatically categorizes entries as "shift" or "time_off" types
- Includes email addresses, color codes, groups, and all Teams-specific fields
- Splits data into past/future based on "today" (Europe/Zurich timezone)
- Computes fairness hints from the last 14 days of data

**Single-File Export (Multi-Sheet Excel):**
- `export_to_teams_excel_multisheet()` creates one Excel file with three sheets:
  - **Schichten** (Shifts)
  - **Arbeitsfreie Zeit** (Time-Off)
  - **Mitglieder** (Members)
- Reuses member data from import if available, or extracts unique members from schedule entries
- Ready for direct import back into Microsoft Teams Shifts
- Maintains all Teams-required formatting and field names

**Dual-File Export (Separate Files):**
- `export_to_teams_excel()` creates two separate Excel files for shifts and time-off
- Useful for workflows that require separate shift and time-off files

**Common Export Features:**
- LLM output is formatted to match Teams Shifts import requirements
- Includes proper German field names and Teams color coding system
- 13 color codes mapped to specific roles and shift types (see `TEAMS_FORMAT_SPEC`)
- Date format: M/D/YYYY, Time format: HH:MM (24-hour)
- All data formatted for seamless Teams Shifts import

**Planning Period:**
- Date range selector in "Compile & Export" tab specifies the scheduling timeframe
- Defaults to current date + 6 days (one week)
- Planning period is automatically included in LLM prompts and test runs

### Schedule File Date Handling
- Uses `zoneinfo.ZoneInfo("Europe/Zurich")` for consistent timezone handling
- `parse_schedule_to_payload()` splits entries by comparing dates to "today" (Europe/Zurich)
- Supports both single-date and date-range formats (expands ranges via `_expand_ranges()`)
- Enhanced column detection supports German names: startdatum, enddatum, mitglied, themenfarbe, etc.

### JSON Compilation
- `Project.as_compact_json()` produces a minimal JSON representation excluding None values
- This JSON is embedded in the system prompt using triple-backtick code blocks

### Duplication Feature
Both employees and shifts support a "Duplicate" button that:
1. Deep-copies the entity
2. Modifies the ID (appends `-copy`)
3. Updates the name/label
4. Adds to the project and sets as currently editing

## Microsoft Teams Color Codes

The application includes a comprehensive color mapping system for Teams Shifts:

| Color Code | Color Name (German) | Typical Use Cases |
|------------|---------------------|-------------------|
| 1. Weiß | White | Operation Lead |
| 2. Blau | Blue | Contact Team, Dispatcher (07:00-16:00) |
| 3. Grün | Green | Contact Team, SOB roles, SOB Wove |
| 4. Lila | Purple | Late shifts (10:00-19:00) |
| 5. Rosa | Pink | Special assignments (Techbar) |
| 6. Gelb | Yellow | Late shifts (09:00-18:00), Dispatcher |
| 8. Dunkelblau | Dark Blue | Project work (M-Industrie Projekt) |
| 9. Dunkelgrün | Dark Green | WoVe, PCV roles |
| 10. Dunkelviolett | Dark Purple | Pikett (on-call duty) |
| 11. Dunkelrosa | Dark Pink | People Developer, Stellvertretung |
| 12. Dunkelgelb | Dark Yellow | Livechat shifts |
| 13. Grau | Gray | Time-off, holidays, sick leave |

**Note:** Color 7 is not used in the Teams system. When adding new roles, choose appropriate colors based on shift characteristics (time of day, role type, special duties).

## Project Structure Documentation

**When unsure where specific functionality is located, consult `projectstructure.md`** - a comprehensive reference documenting all files, classes, functions, data flows, and integration points with exact line numbers.

**After making significant structural changes** (new files, major refactoring, new classes/functions, architectural changes), run the project-structure-analyzer agent to update `projectstructure.md`:

```
run the project analyzer agent
```

This ensures the documentation stays synchronized with the codebase.

## Version Control Workflow

**After completing all edits in a conversation**, use the commit-creator agent to save changes to git. This should be done at the end of a complete task when all file modifications are finished:

The commit-creator agent will:
- Review all staged and unstaged changes
- Generate an appropriate commit message following the repository's style
- Create the commit with proper attribution

**Important:** Only create commits after a logical chunk of work is complete, not after individual file edits.

## Working with the Codebase

### Adding a New Employee Field
1. Update `Employee` model in `models.py`
2. Add corresponding input widget in the Employees tab (`app.py:94-174`)
3. Update the employee save logic (`app.py:176-203`)
4. Consider whether the field should appear in `as_compact_json()`

### Adding a New Shift Field
1. Update `ShiftTemplate` model in `models.py`
2. Add input widget in Shifts & Roles tab (`app.py:261-272`)
3. Update shift save logic (`app.py:293-322`)

### Modifying Prompt Structure
- Edit `SYSTEM_TEMPLATE` or `SCHEDULE_ADDENDUM_TEMPLATE` in `prompt_templates.py`
- For output format changes, users typically modify `RuleSet.output_format_instructions` in the UI

### Supporting New Schedule File Formats
- Add new column name candidates to the lists in `utils.py:58-67`
- Ensure `_normalize_cols()` handles the normalization correctly

## Data Files

- **`test.json`**: Example project file showing the JSON schema with employees and shifts
- Users save/load project files via the sidebar; these are not version-controlled by default (see `.gitignore`)

## Important Notes

- **No TypeScript/Node.js**: This is a pure Python project despite the repo name suggesting a larger codebase
- **LLM Integration**: Requires an OpenAI-compatible API (OpenAI, OpenRouter, etc.) and API key for test runs
- **Timezone**: Hardcoded to `Europe/Zurich` for schedule parsing
- **No Tests**: No test suite is present; validation relies on Pydantic models and manual testing via the UI
