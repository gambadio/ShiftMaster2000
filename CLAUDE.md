# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Shift Studio** is a Streamlit application for AI-powered shift planning with Microsoft Teams integration. It manages employees, shift templates, scheduling rules, and uses LLMs to generate optimized schedules that can be directly imported into Microsoft Teams Shifts.

## Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
# Launches at http://localhost:8501

# Run tests
python test_save_load.py
```

## Architecture

### Core Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application with 6-tab UI (Employees, Shifts & Roles, Rules & Preamble, Schedule File, Compile & Export, Test Run) |
| `models.py` | Pydantic data models for all entities |
| `utils.py` | Project I/O, schedule parsing, Teams import/export |
| `prompt_templates.py` | LLM prompt templates and assembly |
| `llm_manager.py` | Multi-provider LLM calls with reasoning/extended thinking support |
| `llm_client.py` | Unified LLM client class with retry logic and streaming |
| `schedule_manager.py` | Schedule entry management and conflict detection |
| `minizinc_tool.py` | MiniZinc constraint solving integration for schedule optimization |
| `export_teams.py` | Microsoft Teams Shifts export formatting |
| `export_pdf.py` | PDF schedule export |
| `preview.py` | Schedule preview rendering |
| `translations.py` | Multi-language UI support |
| `mcp_config.py` | Model Context Protocol server configuration |
| `query_tool.py` | Query tool for schedule analysis |

### Data Models (`models.py`)

**Core entities:**
- `Project`: Top-level container with employees, shifts, rules, LLM config, schedule state
- `Employee`: Staff with roles, availability, constraints, email (for Teams)
- `ShiftTemplate`: Recurring shift patterns with role, times, weekdays, per-weekday headcount, Teams color codes
- `RuleSet`: System preamble, narrative rules, output format instructions
- `ScheduleEntry` / `GeneratedScheduleEntry`: Unified shift/time-off representation

**LLM configuration:**
- `LLMConfig`: Complete LLM settings including provider config, generation params, reasoning options
- `LLMProviderConfig`: Provider-specific settings (API keys, endpoints, model selection)
- `ProviderType`: Enum for OPENAI, OPENROUTER, AZURE, CUSTOM providers
- `ChatMessage` / `ChatSession`: Conversation state management

**Schedule management:**
- `ScheduleState`: Holds uploaded/generated entries and conflicts
- `ScheduleConflict` / `ConflictType`: Conflict detection and resolution
- `PlanningPeriod`: Date range for schedule generation

### LLM Provider Support

The application supports multiple LLM providers with provider-specific features:

| Provider | Features |
|----------|----------|
| **OpenAI** | GPT-4o, o1/o3 reasoning models with `reasoning_effort` parameter |
| **OpenRouter** | Access to multiple models, reasoning parameter support |
| **Azure OpenAI** | Enterprise deployments with custom endpoints |
| **Custom** | Any OpenAI-compatible endpoint |

**Reasoning model support** (`llm_manager.py`):
- Extended timeouts (30 min read timeout) for reasoning models
- Streaming callbacks for content and thinking chunks
- Automatic provider detection and parameter adjustment

### MiniZinc Integration (`minizinc_tool.py`)

Experimental constraint satisfaction integration for schedule optimization:
- OpenAI function calling schema for LLM-driven model generation
- Solver detection and selection (gecode, chuffed, highs preferred)
- Handles complex constraints: coverage requirements, fairness, rest periods, role assignments

### Microsoft Teams Integration

**Import** (in `utils.py`):
- `parse_teams_excel_multisheet()`: Single-file import parsing Schichten, Arbeitsfreie Zeit, Mitglieder sheets
- `parse_dual_schedule_files()`: Two-file import for separate shifts/time-off files
- German column name detection, timezone handling (Europe/Zurich)

**Export** (in `utils.py`, `export_teams.py`):
- `export_to_teams_excel_multisheet()`: Three-sheet Excel file ready for Teams import
- 13 color codes mapped to shift types (see TEAMS_COLOR_NAMES in models.py)
- Date format: M/D/YYYY, Time format: HH:MM (24-hour)

### Session State Keys

- `st.session_state.project`: Current `Project` instance
- `st.session_state.schedule_payload`: Parsed schedule data
- `st.session_state.editing_employee_id` / `editing_shift_id`: Entity being edited
- `st.session_state.role_options`: Cached roles inferred from shifts and employees

## Key Design Patterns

**Role inference**: Roles dynamically collected from `ShiftTemplate.role` and `Employee.roles`, cached in session state.

**JSON compilation**: `Project.as_compact_json()` produces minimal JSON (excluding None values) embedded in LLM prompts.

**Entity duplication**: Employees/shifts support duplication with ID suffix `-copy`.

**Extra field handling**: Models use `ConfigDict(extra="ignore")` for graceful JSON compatibility.

## Working with the Codebase

### Adding a New Employee Field
1. Update `Employee` model in `models.py`
2. Add input widget in Employees tab (`app.py`)
3. Update employee save logic in `app.py`

### Adding a New Shift Field
1. Update `ShiftTemplate` model in `models.py`
2. Add input widget in Shifts & Roles tab
3. Update shift save logic

### Adding a New LLM Provider
1. Add enum value to `ProviderType` in `models.py`
2. Update `LLMProviderConfig` with provider-specific fields
3. Add routing case in `llm_manager.py:call_llm_with_reasoning()`
4. Implement provider function following `_call_openai()` pattern

### Supporting New Schedule File Formats
- Add column name candidates to lists in `utils.py` (DATE_COL_CANDIDATES, EMP_COL_CANDIDATES, etc.)
- Update `_normalize_cols()` for normalization

## Important Notes

- **Timezone**: Hardcoded to `Europe/Zurich` for all schedule parsing
- **LLM Integration**: Requires API key for selected provider (stored in `LLMProviderConfig.api_key`)
- **MiniZinc**: Experimental feature requiring MiniZinc installation with gecode/chuffed solver
- **Pure Python**: No TypeScript/Node.js despite the repo name

## Version Control

After completing tasks, use the commit-creator agent to create git commits with appropriate messages following conventional commit style.
