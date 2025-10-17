# Shift Prompt Studio - Project Structure

**Version:** 2.0
**Last Updated:** 2025-10-17
**Purpose:** Microsoft Teams-compatible shift planning with AI and constraint optimization

---

## Overview

Shift Prompt Studio is a sophisticated Streamlit application that combines AI-driven schedule generation with constraint programming for optimal shift planning. The application supports full Microsoft Teams Shifts integration (import/export), multi-language UI (English/German), and optional mathematical optimization through MiniZinc.

### Key Features

- **Teams Integration**: Full import/export compatibility with Microsoft Teams Shifts (multi-sheet Excel)
- **AI Schedule Generation**: LLM-powered schedule creation with tool calling support
- **Constraint Solver**: Optional MiniZinc integration for provably optimal schedules
- **Multi-Language UI**: English and German translations
- **Interactive Chat**: Conversational schedule refinement
- **Calendar Preview**: Teams Shifts-style visual calendar
- **Shift Pattern Detection**: Automatic template creation from historical data

---

## Directory Structure

```
ShiftMaster2000/
├── app.py                              # Main Streamlit application (1677 lines)
├── models.py                           # Pydantic data models (209 lines)
├── utils.py                            # Core utilities (949 lines)
├── prompt_templates.py                 # System prompt construction (162 lines)
├── llm_manager.py                      # LLM integration with tool calling (535 lines) 🆕
├── llm_client.py                       # Multi-provider LLM client (426 lines)
├── translations.py                     # UI translations (456 lines)
├── preview.py                          # Calendar preview component (430 lines)
├── export_teams.py                     # Teams Excel export (382 lines)
├── mcp_config.py                       # MCP server configuration (133 lines)
├── solver_models.py                    # Solver JSON contract models (244 lines) 🆕
├── solver_service.py                   # MiniZinc execution wrapper (357 lines) 🆕
├── solver_utils.py                     # Solver detection and utilities (224 lines) 🆕
├── shift_schedule.mzn                  # MiniZinc constraint model (185 lines) 🆕
├── requirements.txt                    # Python dependencies
├── CLAUDE.md                           # Project instructions for Claude Code
├── knowledge.md                        # Technical knowledge base
├── SOLVER_SETUP.md                     # User guide for solver installation 🆕
├── SOLVER_INTEGRATION_SUMMARY.md       # Technical documentation 🆕
└── SOLVER_INTEGRATION_COMPLETE.md      # Testing guide 🆕
```

---

## Core Components

### `app.py` - Main Application (1677 lines)

**Purpose**: Streamlit UI orchestration and application flow

**Key Sections**:
- Lines 30-63: Custom CSS for Teams Shifts styling
- Lines 68-105: Session state initialization
- Lines 109-196: Sidebar (project management, save/load)
- Lines 200-212: Tab structure (11 tabs)

**Tab Implementations**:

#### Tab 1: Employee Management (lines 217-361)
- CRUD operations for employees with role inference
- Multi-language support, email validation for Teams export
- Session state: `editing_employee_id`, `selected_emp_name`
- Lines 217-234: Employee list display
- Lines 236-275: Employee form (name, email, roles, languages, time windows)
- Lines 277-309: Save logic with validation
- Lines 311-361: Remove functionality

#### Tab 2: Shift Templates (lines 365-481)
- Shift template management with Teams color codes
- Per-weekday headcount configuration
- Concurrent shifts relationships
- Lines 365-384: Shift list display
- Lines 386-434: Shift form (ID, role, times, colors, weekdays)
- Lines 436-459: Save logic
- Lines 461-481: Remove functionality

#### Tab 3: Rules & Preamble (lines 486-498)
- System preamble, narrative rules, output format instructions
- Direct editing of `project.global_rules`

#### Tab 4: Schedule Import (lines 502-723)
- Single-file (multi-sheet) and dual-file import modes
- Auto-detect shift patterns feature (lines 519-620)
- Employee auto-population from Teams members data
- Preview generation with statistics
- Lines 502-516: Import mode selection
- Lines 519-620: Auto-detect shift patterns workflow
- Lines 622-672: Single-file import with parsing
- Lines 674-723: Dual-file import

#### Tab 5: Planning Period & Solver Configuration (lines 727-846) 🆕
**Purpose**: Configure planning date range and optional constraint solver

**Date Range Selection (Lines 734-747)**:
- Start/end date inputs with validation
- Days count display
- Stored in `project.planning_period`

**Solver Configuration (Lines 749-846) - NEW**:

**1. Solver Availability Check (Lines 756-767)**:
```python
solver_info = get_solver_info()
is_available = solver_info["available"]
st.info(solver_info["message"])  # Shows MiniZinc version and solvers
st.checkbox("Enable Constraint Solver Mode", key="enable_solver")
```

**2. Solver Backend and Timeout (Lines 769-788)**:
- Backend selection: chuffed (default), gecode, coin-bc
- Timeout: 5-120 seconds, default 15

**3. Soft Constraint Weights (Lines 791-807)**:
- Fairness Weight: 0.0-10.0, default 3.0
- Coverage Weight: 0.0-10.0, default 5.0
- Late Violation Weight: 0.0-10.0, default 2.0
- Pikett Violation Weight: 0.0-10.0, default 2.0
- Overtime Weight: 0.0-10.0, default 2.5

**4. Constraint Rules (Lines 810-836)**:
- No Consecutive Late Shifts toggle
- Minimum Pikett Gap (days): 0-14, default 7
- French Speakers per Week (Dispatcher): 0-10, default 1

**5. Installation Instructions (Lines 840-846)**:
- Expander with platform-specific instructions

**Session State Keys for Solver**:
- `enable_solver` - Boolean toggle
- `solver_backend` - "chuffed", "gecode", or "coin-bc"
- `solver_timeout` - Integer seconds
- `solver_weights` - Dict of float weights
- `no_consecutive_late` - Boolean
- `pikett_gap_days` - Integer
- `fr_dispatcher_per_week` - Integer

#### Tab 6: Prompt Preview (lines 851-894)
- Compiled system prompt display with statistics
- Character/token counts, download button
- Lines 851-865: Build prompt with planning period
- Lines 867-879: Display statistics
- Lines 881-894: Download button and text area

#### Tab 7: LLM Configuration (lines 898-1214)
- Multi-provider support (OpenAI, OpenRouter, Azure, Custom)
- Model fetching and selection
- Generation parameters (temperature, tokens, penalties)
- Reasoning effort for o1/o3 models
- MCP server configuration
- Lines 898-936: Provider configuration UI
- Lines 938-1014: Model selection and parameters
- Lines 1016-1098: Generation settings
- Lines 1100-1214: MCP server management

#### Tab 8: Chat Interface (lines 1218-1362)
- Interactive conversation with LLM
- Token usage tracking (prompt, completion, reasoning)
- Session history saving
- Lines 1218-1247: Token usage display
- Lines 1249-1321: Message display loop
- Lines 1323-1362: Message input and send button

#### Tab 9: Generate Schedule (lines 1366-1443) 🆕
**Enhanced Generation Workflow**:

**Lines 1378-1406**: Schedule generation with tool calling
```python
if st.button("🚀 Generate Schedule"):
    # Build prompt with planning period
    prompt = build_system_prompt(project, schedule_payload, today, planning_period)

    # Call LLM with tools enabled if solver available
    enable_tools = st.session_state.get("enable_solver", False) and is_solver_available

    result = call_llm_sync(
        prompt=prompt,
        config=project.llm_config,
        user_message="Produce the schedule now.",
        enable_tools=enable_tools  # 🆕 Tool calling flag
    )
```

**Lines 1402-1411**: Tool usage display
- Shows tool calls in expander if solver was invoked
- Displays tool name, status, and input preview

**Lines 1413-1443**: Result parsing and storage
- Parse LLM JSON output
- Convert to ScheduleEntry objects
- Store in `st.session_state.generated_entries`

#### Tab 10: Preview (lines 1447-1580)
- Dual preview mode (imported + generated)
- Calendar view with navigation controls
- Statistics and conflict detection
- Lines 1447-1504: Imported schedule preview
- Lines 1506-1580: Generated schedule preview

#### Tab 11: Export (lines 1584-1677)
- Single-file (multi-sheet) and dual-file export
- Teams-compatible Excel formatting
- Members sheet generation
- Lines 1584-1623: Export format selection
- Lines 1625-1660: Single-file export
- Lines 1662-1677: Dual-file export

**Key Functions**:
- Lines 69-87: `initialize_session_state()` - Sets up all session variables
- Lines 133-156: Project loading with backward compatibility
- Lines 167-187: Complete state saving

**Dependencies**: models, utils, llm_client, llm_manager, export_teams, preview, mcp_config, prompt_templates, translations

---

### `models.py` - Data Models (209 lines)

**Purpose**: Pydantic schemas for type-safe data handling

**Core Classes**:

#### `ShiftTemplate` (lines 39-53)
```python
class ShiftTemplate(BaseModel):
    id: ShiftID
    role: str
    start_time: str  # "HH:MM"
    end_time: str    # "HH:MM"
    weekdays: List[str]
    required_count: Dict[str, int]  # per weekday
    color_code: Optional[str]  # "1" to "13"
    concurrent_shifts: List[str]
    notes: Optional[str]
```

#### `Employee` (lines 55-72)
```python
class Employee(BaseModel):
    id: str
    name: str
    email: Optional[str]  # Required for Teams
    percent: Optional[int]  # FTE percentage
    roles: List[str]
    languages: List[str]
    earliest_start: Optional[str]
    latest_end: Optional[str]
    weekday_blockers: Dict[str, str]
    hard_constraints: List[str]
    soft_preferences: List[str]
    group: Optional[str]  # Teams grouping
```

#### `ScheduleEntry` (lines 73-90)
Unified representation for shifts and time-off:
- employee_name, employee_email
- start_date, end_date, start_time, end_time
- color_code, entry_type ("shift" or "time_off")
- reason (for time-off), label, notes
- group, shared, unpaid_break

#### `PlanningPeriod` (lines 91-94)
Date range for schedule generation

#### `LLMProviderConfig` (lines 110-137)
Multi-provider configuration:
- OpenAI: api_key, model
- OpenRouter: api_key, http_referer, x_title
- Azure: api_key, azure_endpoint, azure_deployment, api_version
- Custom: base_url, model

#### `LLMConfig` (lines 139-166)
Complete LLM configuration:
- provider_config: LLMProviderConfig
- model_family: "claude", "openai", "custom"
- model_name: str
- temperature, top_p, max_tokens
- frequency_penalty, presence_penalty
- seed, stop_sequences
- json_mode, enable_streaming
- budget_tokens, reasoning_effort
- enable_interleaved_thinking
- mcp_servers: List[MCPServerConfig]

#### `ChatMessage` & `ChatSession` (lines 168-181)
Conversation state management with token tracking:
- messages: List[ChatMessage]
- total_prompt_tokens, total_completion_tokens, total_reasoning_tokens

#### `Project` (lines 192-208)
Top-level container:
```python
class Project(BaseModel):
    name: str
    version: str
    employees: List[Employee]
    shifts: List[ShiftTemplate]
    global_rules: RuleSet
    llm_config: Optional[LLMConfig]
    planning_period: Optional[PlanningPeriod]

    def as_compact_json(self) -> Dict[str, Any]:
        # Returns minimal JSON for prompt inclusion
```

**Constants**:
- Lines 24-37: `TEAMS_COLOR_NAMES` - 13 Teams Shifts colors

---

### `utils.py` - Core Utilities (949 lines)

**Purpose**: Data processing, parsing, and transformation utilities

**Major Function Groups**:

#### Project I/O (lines 36-116)
- `save_project(path, project)` (36-41): Save project to JSON
- `save_complete_state(path, project, schedule_payload, generated, conversations, members)` (43-73): Save everything
- `load_project_dict(data)` (75-81): Load project with format detection
- `load_complete_state(data)` (83-116): Restore complete application state

#### Schedule File Parsing (lines 153-334)
**Column Detection** (lines 155-169):
- DATE_COL_CANDIDATES: date, datum, startdatum, start_date, start date
- EMP_COL_CANDIDATES: employee, name, mitarbeiter, mitglied, member
- EMAIL_COL_CANDIDATES: email, e_mail, e-mail, e_mail_geschaftlich
- GROUP_COL_CANDIDATES: group, gruppe, team, shift team
- COLOR_COL_CANDS: color, colour, themenfarbe, shift color, color code

**Helper Functions**:
- `_normalize_cols(df)` (171-182): Column name normalization
- `_pick(df, candidates)` (184-188): Flexible column selection
- `_parse_date(val)` (190-206): Robust date parsing
- `_expand_ranges(df, col_from, col_to)` (222-246): Date range expansion
- `_compute_fairness_hints(rows)` (336-354): Late/pikett rotation tracking (last 14 days)

**Main Parser**:
- `parse_schedule_to_payload(file_bytes, filename, today)` (248-334):
  - Reads Excel/CSV files
  - Normalizes column names (supports German Teams exports)
  - Expands date ranges
  - Splits past/future entries
  - Computes fairness hints

#### Teams Integration (lines 358-613)

**Dual-File Import** (lines 359-427):
- `parse_dual_schedule_files(shifts_bytes, timeoff_bytes, today)`:
  - Parses separate shifts and time-off files
  - Unified payload generation

**Helper Functions** (lines 429-487):
- `_parse_teams_file(file_bytes, filename, file_type, today)` (429-487):
  - Core Teams file parser
  - Handles both shift and time-off formats
  - Normalizes German column names

**Single-File Multi-Sheet Import** (lines 489-613) - **MOST POWERFUL**:
- `parse_teams_excel_multisheet(file_bytes, filename, today)`:
  - Detects and parses three sheets:
    - "Schichten" (Shifts)
    - "Arbeitsfreie Zeit" (Time-Off)
    - "Mitglieder" (Members)
  - Flexible sheet name matching (case-insensitive, German/English)
  - Extracts member data for validation and export
  - Returns unified payload with past/future split

#### Employee Auto-Population (lines 617-767)
- `find_duplicate_employee(project, name, email)` (618-642): Duplicate detection
- `auto_populate_employees_from_members(project, members_data)` (644-707):
  - Creates employees from Teams member list
  - Duplicate detection (name + email match)
  - Returns summary with added/existing counts

- `generate_schedule_preview(schedule_payload, employee_changes)` (709-767):
  - Generates statistics and sample entries
  - Combines schedule data with employee changes

#### Shift Pattern Detection (lines 772-948) - **AUTO-DETECTION**
- `detect_shift_patterns_from_schedule(schedule_payload, project)` (772-948):
  - **Automatic shift template creation**
  - Pattern analysis based on: start_time + end_time + role + color_code
  - Groups entries by pattern key
  - Tracks occurrences, weekdays, employees
  - Creates new ShiftTemplate objects
  - Filters out rare patterns (< 2 occurrences)
  - Returns detection summary with added/existing status

**Key Pattern Detection Logic** (lines 805-866):
- Pattern key: `"{start_time}-{end_time}||{role_identifier}||{color_code}"`
- Tracks: count, employees, dates, weekdays, color, label, notes
- Minimum threshold: 2 occurrences
- Infers role from label and notes

**Dependencies**: pandas, datetime, zoneinfo, models

---

### `prompt_templates.py` - System Prompt Construction (162 lines)

**Purpose**: Build comprehensive system prompts for LLM

**Key Functions**:

#### `get_solver_tool_definition_if_enabled()` (lines 12-39) 🆕
```python
def get_solver_tool_definition_if_enabled() -> str:
    # Check if solver mode is enabled
    if not st.session_state.get("enable_solver", False):
        return ""

    # Check if MiniZinc is available
    from solver_utils import check_minizinc_available
    from solver_models import SOLVER_TOOL_DEFINITION

    is_available, _ = check_minizinc_available()
    if not is_available:
        return ""

    return "\n\n" + SOLVER_TOOL_DEFINITION
```
- Checks session state for solver enablement
- Validates MiniZinc availability
- Returns solver tool definition for prompt inclusion
- Graceful fallback if unavailable

#### `build_system_prompt(project, schedule_payload, today_iso, planning_period)` (lines 111-161)
Main prompt assembly:
1. **Lines 117-138**: Build planning period context
   - Extract start/end dates
   - Calculate weekdays and total days
   - Format as markdown section

2. **Lines 140-146**: Format SYSTEM_TEMPLATE with:
   - Preamble from project rules
   - Teams color specification
   - Narrative rules
   - Output format instructions
   - Planning period context

3. **Lines 147**: Append compact JSON data

4. **Lines 149-154**: Append schedule addendum if available

5. **Lines 156-159**: Inject solver tool definition if enabled 🆕

**Templates**:
- `TEAMS_COLOR_SPEC` (lines 41-59): Color code reference for LLM (13 colors)
- `SYSTEM_TEMPLATE` (lines 61-95): Base prompt structure
- `SCHEDULE_ADDENDUM_TEMPLATE` (lines 97-109): Historical context injection

**Dependencies**: models, streamlit, solver_models, solver_utils

---

### `llm_manager.py` - LLM Integration with Tool Calling (535 lines) 🆕

**Purpose**: Advanced LLM management with extended thinking, reasoning, and tool execution

**Key Functions**:

#### Main Entry Point (lines 24-52)
```python
async def call_llm_with_reasoning(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
    on_chunk: Optional[Callable[[str], None]] = None,
    on_thinking: Optional[Callable[[str], None]] = None,
    enable_tools: bool = False,  # 🆕 Tool calling toggle
) -> Dict[str, Any]
```
- Returns dict with 'content', 'thinking', 'usage', optionally 'tool_calls'
- Model family dispatch (Claude, OpenAI, Generic)

#### Claude Integration with Tools (lines 55-116)
- `_call_claude(prompt, config, user_message, on_chunk, on_thinking, enable_tools)`:
  - **Lines 86-89**: Tool injection if enabled
  - **Lines 92-103**: Extended thinking configuration
    - budget_tokens for thinking budget
    - interleaved thinking mode (beta header)
  - **Line 114**: Tool call handling via `_handle_tool_calls_claude()`

#### Tool System (lines 351-534) 🆕

**Tool Management** (lines 353-383):
```python
def _get_available_tools() -> List[Dict[str, Any]]:
    """Returns list of tool definitions for LLM"""
    if st.session_state.get("enable_solver", False):
        from solver_utils import check_minizinc_available
        from solver_models import SolverRequest

        is_available, _ = check_minizinc_available()
        if is_available:
            tools.append({
                "name": "solve_with_minizinc",
                "description": "Generate optimized shift schedule...",
                "input_schema": SolverRequest.model_json_schema()
            })
    return tools
```

**Tool Execution** (lines 386-445):
```python
def _execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "solve_with_minizinc":
        # Inject weights from session state (lines 404-409)
        tool_input["rules"]["weights"] = st.session_state.solver_weights

        # Inject constraint rules (lines 411-422)
        tool_input["rules"]["no_consecutive_late"] = st.session_state.no_consecutive_late
        tool_input["rules"]["pikett_gap_days"] = st.session_state.pikett_gap_days
        tool_input["rules"]["fr_dispatcher_per_week"] = st.session_state.fr_dispatcher_per_week

        # Inject solver backend and timeout (lines 425-430)
        tool_input["options"]["solver"] = st.session_state.solver_backend
        tool_input["options"]["time_limit_ms"] = st.session_state.solver_timeout * 1000

        # Validate and execute (lines 432-435)
        request = SolverRequest.model_validate(tool_input)
        response = solve_with_minizinc(request)
        return response.model_dump()
```

**Tool Call Loop** (lines 448-534):
```python
async def _handle_tool_calls_claude(...) -> Dict[str, Any]:
    """Implements Claude tool use protocol"""
    # 1. Extract tool uses from response (lines 464-478)
    tool_uses = [block for block in response.content if block.type == "tool_use"]

    # 2. Execute each tool (lines 488-507)
    for tool_use in tool_uses:
        result = _execute_tool(tool_use["name"], tool_use["input"])
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": json.dumps(result, indent=2)
        })

    # 3. Send tool results back to LLM (lines 510-516)
    messages.append({"role": "user", "content": tool_results})

    # 4. Get final response (line 519)
    final_response = client.messages.create(**params)

    # 5. Return merged result (lines 522-534)
    return final_result  # Includes tool_calls metadata
```

#### Synchronous Wrapper (lines 341-348)
```python
def call_llm_sync(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
    enable_tools: bool = False,  # 🆕
) -> Dict[str, Any]:
    return asyncio.run(call_llm_with_reasoning(prompt, config, user_message, enable_tools=enable_tools))
```

**Dependencies**: anthropic, openai, solver_service, solver_models, solver_utils, streamlit, models

---

### `llm_client.py` - Multi-Provider LLM Client (426 lines)

**Purpose**: Unified interface for OpenAI, OpenRouter, Azure, and Custom endpoints

**Core Class**: `LLMClient` (lines 32-382)

**Initialization** (lines 45-95):
- `_create_client()` (56-95): Creates provider-specific OpenAI client
  - OpenAI: Standard OpenAI client
  - OpenRouter: Custom base_url + HTTP-Referer/X-Title headers
  - Azure: AzureOpenAI with endpoint and api_version
  - Custom: OpenAI client with custom base_url

**Model Management**:
- `fetch_models()` (97-115): Lists available models via API
- `_get_model_max_tokens(model_name)` (117-152): Token limits per model
  - Known limits for GPT-4, GPT-3.5, o1, Claude, Gemini, LLaMA

**Request Building**:
- `_build_completion_params(messages, stream, override_params)` (154-212):
  - Handles max_tokens with fallback to model limits
  - JSON mode toggle
  - Reasoning effort for o1/o3 models
  - Seed, stop sequences, penalties

**Completion Methods**:
- `complete(messages, max_retries, override_params)` (214-281):
  - Synchronous completion with retry logic
  - Exponential backoff for rate limits (3 retries)
  - Reasoning token extraction (o1/o3 models)

- `stream_complete(messages, on_token, on_complete, override_params)` (283-322):
  - Generator-based streaming
  - Optional callbacks per token and on completion

**Chat Interface**:
- `chat(user_message, session, system_prompt)` (324-382):
  - Manages ChatSession state
  - Adds system prompt on first message
  - Updates session with messages and token usage
  - Tracks reasoning_tokens separately

**Utilities**:
- `create_llm_client(config)` (385-395): Factory function
- `validate_provider_config(config)` (398-425): Provider-specific validation

**Dependencies**: openai (OpenAI, AzureOpenAI), models

---

### Solver Integration (4 files) 🆕

#### `solver_models.py` - JSON Contract (244 lines)

**Purpose**: Pydantic models defining the JSON contract between LLM and MiniZinc solver

**Core Models**:

**Enums** (lines 13-27):
- `SolverBackend`: GECODE, CHUFFED, COIN_BC
- `SolverStatus`: OPTIMAL, FEASIBLE, INFEASIBLE, UNKNOWN, TIMEOUT, ERROR

**EmployeeData** (lines 30-41):
```python
class EmployeeData(BaseModel):
    id: str
    fte: float = 1.0  # 0.0-1.0
    skills: List[str]
    languages: List[str]
    availability_mask: List[List[int]]  # [day][shift] -> 1/0
    target_weekly_shifts: Optional[int]
```

**ShiftData** (lines 43-57):
```python
class ShiftData(BaseModel):
    id: str
    role: str
    start_time: str  # "HH:MM"
    end_time: str
    is_late: bool  # Shifts starting after 09:00
    is_pikett: bool
    is_dispatcher: bool
    needs_french: bool
    required_per_day: List[int]  # [Mon-Sun]
```

**ConstraintRules** (lines 59-77):
```python
class ConstraintRules(BaseModel):
    no_consecutive_late: bool = True
    pikett_gap_days: int = 7
    fr_dispatcher_per_week: int = 1
    max_shifts_per_week: Optional[int] = 6
    weights: Dict[str, float] = {
        "fairness": 3.0,
        "late_violation": 2.0,
        "pikett_violation": 2.0,
        "preference_violation": 1.0,
        "overtime": 2.5
    }
```

**SolverOptions** (lines 79-87):
- solver: SolverBackend
- time_limit_ms: int (default 15000)
- all_solutions, intermediate_solutions, num_solutions

**SolverRequest** (lines 90-99):
```python
class SolverRequest(BaseModel):
    horizon: Dict[str, str]  # {from: YYYY-MM-DD, to: YYYY-MM-DD}
    employees: List[EmployeeData]
    shifts: List[ShiftData]
    rules: ConstraintRules
    options: SolverOptions
```

**Result Models** (lines 101-147):
- `Assignment`: date, employee_id, shift_id
- `Violation`: type, details, employee_id, severity
- `PenaltyBreakdown`: fairness, late, pikett, coverage, overtime penalties
- `SolverStats`: solver, nodes, failures, time_ms, timeout_reached
- `SolverResponse`: status, objective, breakdown, assignments, violations, stats, message

**Tool Definition** (lines 150-243):
- `SOLVER_TOOL_DEFINITION`: Complete markdown documentation for LLM
- Includes input/output schemas, status handling, best practices

**Dependencies**: pydantic

---

#### `solver_service.py` - MiniZinc Execution (357 lines)

**Purpose**: Bridge between LLM JSON requests and MiniZinc solver

**Main Entry Point** (lines 23-96):
```python
def solve_with_minizinc(request: SolverRequest) -> SolverResponse:
    # 1. Check availability (lines 36-42)
    is_available, message = check_minizinc_available()

    # 2. Build index maps (lines 50-52)
    emp_to_idx, idx_to_emp = _build_employee_index(request.employees)
    shift_to_idx, idx_to_shift = _build_shift_index(request.shifts)
    date_list = _build_date_range(request.horizon)

    # 3. Map data to MiniZinc parameters (lines 55-57)
    params = _build_minizinc_parameters(request, emp_to_idx, shift_to_idx, date_list)

    # 4. Load model and create instance (lines 60-69)
    model_path = Path(__file__).parent / "shift_schedule.mzn"
    model = minizinc.Model(model_path)
    solver = minizinc.Solver.lookup(request.options.solver.value)
    instance = minizinc.Instance(solver, model)

    # 5. Solve with timeout (lines 72-78)
    result = instance.solve(timeout=td(milliseconds=request.options.time_limit_ms))

    # 6. Parse result (lines 81-88)
    return _parse_minizinc_result(result, request, idx_to_emp, idx_to_shift, date_list, elapsed_ms)
```

**Index Building** (lines 99-126):
- `_build_employee_index(employees)` (99-103): emp_id → 1-based index
- `_build_shift_index(shifts)` (106-110): shift_id → 1-based index
- `_build_date_range(horizon)` (113-126): ISO date list from range

**Parameter Mapping** (lines 129-218):
```python
def _build_minizinc_parameters(request, emp_to_idx, shift_to_idx, date_list) -> Dict[str, Any]:
    # Dimensions (lines 140-142)
    E, D, S = len(employees), len(dates), len(shifts)

    # Employee attributes (lines 145-153)
    fte = [0.0] + [emp.fte for emp in employees]  # 1-indexed
    target_shifts = [0] + [emp.target_weekly_shifts or int(emp.fte * 5) for emp in employees]
    speaks_french = [False] + ["fr" in emp.languages for emp in employees]

    # Shift attributes (lines 156-158)
    is_late = [False] + [shift.is_late for shift in shifts]
    is_pikett = [False] + [shift.is_pikett for shift in shifts]
    needs_french = [False] + [shift.needs_french for shift in shifts]

    # Requirements array (lines 161-173): required[day][shift]
    # Availability mask (lines 176-193): available[emp][day][shift]

    # Return complete parameter dict (lines 199-218)
```

**Result Parsing** (lines 221-330):
```python
def _parse_minizinc_result(result, request, idx_to_emp, idx_to_shift, date_list, elapsed_ms):
    # Determine status (lines 234-254)
    if result.status == Status.OPTIMAL_SOLUTION:
        status = SolverStatus.OPTIMAL
    elif result.status == Status.SATISFIED:
        status = SolverStatus.FEASIBLE
    elif result.status == Status.UNSATISFIABLE:
        status = SolverStatus.INFEASIBLE

    # Extract objective and breakdown (lines 257-266)
    objective = float(result["total_penalty"])
    breakdown = PenaltyBreakdown(...)

    # Extract assignments from 3D variable x (lines 269-284)
    for e_idx in range(1, E+1):
        for d_idx in range(1, D+1):
            for s_idx in range(1, S+1):
                if x[e_idx-1][d_idx-1][s_idx-1] == 1:
                    assignments.append(Assignment(...))

    # Build violations list (lines 287-311)
    # Build stats (lines 314-320)
    # Return SolverResponse (lines 322-330)
```

**JSON Helper** (lines 335-356):
```python
def call_solver_from_json(request_json: str) -> str:
    """Alternative entry point for JSON string interface"""
    request_dict = json.loads(request_json)
    request = SolverRequest.model_validate(request_dict)
    response = solve_with_minizinc(request)
    return response.model_dump_json(indent=2)
```

**Dependencies**: minizinc, solver_models, solver_utils, pathlib, datetime

---

#### `solver_utils.py` - Detection & Installation (224 lines)

**Purpose**: Graceful detection and user guidance

**Detection Functions**:

**Availability Check** (lines 12-33):
```python
def check_minizinc_available() -> Tuple[bool, str]:
    try:
        import minizinc
        driver = minizinc.Driver.find()
        version = driver.minizinc_version
        return True, f"✅ MiniZinc {version} detected"
    except ImportError:
        return False, "❌ minizinc Python package not installed"
    except Exception as e:
        return False, f"❌ MiniZinc binary not found: {e}"
```

**Solver Discovery** (lines 36-59):
```python
def get_available_solvers() -> list[str]:
    solvers = []
    for solver_name in ["gecode", "chuffed", "coin-bc", "or-tools"]:
        try:
            solver = minizinc.Solver.lookup(solver_name)
            if solver:
                solvers.append(solver_name)
        except Exception:
            pass
    return solvers
```

**Solver Info** (lines 62-101):
```python
def get_solver_info() -> dict:
    is_available, message = check_minizinc_available()
    if not is_available:
        return {"available": False, "version": None, "solvers": [], "message": message}

    driver = minizinc.Driver.find()
    version = str(driver.minizinc_version)
    solvers = get_available_solvers()

    return {
        "available": True,
        "version": version,
        "solvers": solvers,
        "message": f"✅ MiniZinc {version} with {len(solvers)} solver(s): {', '.join(solvers)}"
    }
```

**Installation Guidance**:

**Platform-Specific Instructions** (lines 104-179):
```python
def get_installation_instructions() -> str:
    platform = sys.platform

    # macOS (lines 121-131)
    if platform == "darwin":
        instructions = "brew install minizinc"

    # Windows (lines 133-139)
    elif platform == "win32":
        instructions = "Download from https://www.minizinc.org/software.html"

    # Linux (lines 140-152)
    else:
        instructions = "sudo apt-get install minizinc"

    # Post-installation steps (lines 154-177)
```

**Fallback Message** (lines 204-223):
```python
SOLVER_UNAVAILABLE_MESSAGE = """
🔧 **Constraint Solver Mode Unavailable**
The constraint solver is an optional feature that requires MiniZinc.
**Current mode:** AI-based schedule generation (works great for most scenarios)
**Your app works perfectly without it!** The AI mode handles most scheduling needs.
"""
```

**Dependencies**: minizinc (optional), solver_models, sys

---

#### `shift_schedule.mzn` - Constraint Model (185 lines)

**Purpose**: Constraint programming model for optimal shift scheduling

**Parameters** (lines 12-43):
```minizinc
int: E;  % Number of employees
int: D;  % Number of days
int: S;  % Number of shift types

array[1..E] of float: fte;
array[1..E] of int: target_shifts;
array[1..E] of bool: speaks_french;

array[1..S] of bool: is_late;
array[1..S] of bool: is_pikett;
array[1..S] of bool: needs_french;

array[1..D, 1..S] of int: required;
array[1..E, 1..D, 1..S] of 0..1: available;

bool: enforce_no_consecutive_late;
int: pikett_min_gap_days;
int: min_french_per_week;

float: weight_coverage;
float: weight_fairness;
float: weight_late;
float: weight_pikett;
```

**Decision Variables** (line 48):
```minizinc
array[1..E, 1..D, 1..S] of var 0..1: x;
```

**Hard Constraints** (lines 52-60):
```minizinc
% AVAILABILITY
constraint forall(e in 1..E, d in 1..D, s in 1..S)(
    x[e,d,s] <= available[e,d,s]
);

% ONE SHIFT PER DAY
constraint forall(e in 1..E, d in 1..D)(
    sum(s in 1..S)(x[e,d,s]) <= 1
);
```

**Soft Constraints** (lines 66-117):
```minizinc
% Coverage deficit (lines 68-72)
array[1..D, 1..S] of var 0..E: coverage_deficit;

% Fairness deviation (lines 75-83)
array[1..E] of var int: shift_count;
array[1..E] of var 0..D*S: fairness_deviation;

% Consecutive late shifts (lines 86-98)
array[1..E, 1..D-1] of var 0..1: consecutive_late_violation;

% Pikett gap violations (lines 101-117)
array[1..E] of var 0..D: pikett_gap_violations;
```

**Domain-Specific Constraints** (lines 123-141):
```minizinc
% French language requirement (hard constraint per week)
constraint
    if min_french_per_week > 0 then
        forall(week in 0..(D-1) div 7)(
            sum(e in 1..E where speaks_french[e],
                d in week_start..week_end,
                s in 1..S where needs_french[s])(
                x[e,d,s]
            ) >= min_french_per_week
        )
    else
        true
    endif;
```

**Objective Function** (lines 146-157):
```minizinc
var float: total_penalty =
    weight_coverage * sum(d in 1..D, s in 1..S)(coverage_deficit[d,s]) +
    weight_fairness * sum(e in 1..E)(fairness_deviation[e]) +
    weight_late * sum(e in 1..E, d in 1..D-1)(consecutive_late_violation[e,d]) +
    weight_pikett * sum(e in 1..E)(pikett_gap_violations[e]);

solve minimize total_penalty;
```

**Output** (lines 162-184):
- JSON format with objective, breakdown, assignments
- Parsed by Python solver service

**Dependencies**: None (pure MiniZinc)

---

### `export_teams.py` - Teams Export (382 lines)

**Purpose**: Convert schedule entries to Teams-compatible Excel files

**Export Functions**:

**Dual-File Export** (lines 14-42):
```python
def export_to_teams_excel(schedule_entries, output_shifts, output_timeoff):
    """Export to two separate Excel files"""
    shifts = [e for e in schedule_entries if e.entry_type == "shift"]
    timeoffs = [e for e in schedule_entries if e.entry_type == "time_off"]

    if shifts:
        _export_shifts_file(shifts, output_shifts)
    if timeoffs:
        _export_timeoff_file(timeoffs, output_timeoff)
```

**Single-File Multi-Sheet Export** (lines 45-85) - **RECOMMENDED**:
```python
def export_to_teams_excel_multisheet(schedule_entries, output_file, members=None):
    """Export to single file with three sheets: Schichten, Arbeitsfreie Zeit, Mitglieder"""
    shifts_df = _prepare_shifts_dataframe(shifts)
    timeoff_df = _prepare_timeoff_dataframe(timeoffs)
    members_df = _prepare_members_dataframe(schedule_entries, members)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        shifts_df.to_excel(writer, sheet_name="Schichten", index=False)
        timeoff_df.to_excel(writer, sheet_name="Arbeitsfreie Zeit", index=False)
        members_df.to_excel(writer, sheet_name="Mitglieder", index=False)
```

**Data Preparation**:
- `_prepare_shifts_dataframe(shifts)` (168-199): Formats shift data
- `_prepare_timeoff_dataframe(timeoffs)` (202-235): Formats time-off data
- `_prepare_members_dataframe(schedule_entries, members)` (238-262): Member extraction

**Teams Formatting**:
- Date format: M/D/YYYY (e.g., "1/20/2025")
- Time format: HH:MM (24-hour)
- Color codes: "1. Weiß", "2. Blau", etc.
- Required fields: Mitglied, E-Mail (geschäftlich), Gruppe, Themenfarbe

**LLM Output Conversion** (lines 265-313):
```python
def schedule_entries_from_llm_output(llm_output: Dict[str, Any]) -> List[ScheduleEntry]:
    """Convert LLM JSON to ScheduleEntry objects"""
    entries = []
    assignments = llm_output.get("assignments", [])

    for assignment in assignments:
        entry = ScheduleEntry(
            employee_name=assignment.get("employee_name"),
            employee_email=assignment.get("employee_email"),
            start_date=assignment.get("date"),
            start_time=assignment.get("start_time"),
            end_time=assignment.get("end_time"),
            color_code=assignment.get("color_code"),
            label=assignment.get("role"),
            entry_type="shift"
        )
        entries.append(entry)

    return entries
```

**Sample Export** (lines 316-377):
- `create_sample_export()`: Creates test files for validation

**Dependencies**: pandas, datetime, models

---

### `preview.py` - Calendar Preview (430 lines)

**Purpose**: Teams Shifts-style visual calendar and conflict detection

**Main Functions**:

#### `render_calendar_preview(schedule_entries, start_date, end_date, title)` (lines 16-348)

**Navigation Controls** (lines 40-76):
- Previous/Next week buttons with view offset
- "Go to Latest" finds most recent entry date
- Session state: `calendar_view_offset_{title}`

**Grid Generation** (lines 104-125):
- Employee × date grid
- Collects entries per employee per date

**HTML Rendering** (lines 128-347):
- Teams Shifts color scheme (13 colors)
- Sticky headers (week, month, dates)
- Employee cells with total hours calculation
- Shift blocks with hover effects

**CSS Styling** (lines 129-230):
```css
.teams-schedule { /* Dark theme table */ }
.shift-block { /* Color-coded shift cells */ }
.color-1 { background-color: #ffffff; } /* White */
.color-2 { background-color: #0078d4; } /* Blue */
/* ... 13 colors total */
```

**Hours Calculation** (lines 233-250):
- Parses start/end times
- Subtracts unpaid breaks
- Sums per employee

#### `render_statistics(schedule_entries)` (lines 350-373)
Displays:
- Total shifts
- Time-off entries
- Unique employees
- Days covered

#### `render_conflicts(schedule_entries)` (lines 375-429)

**Conflict Detection** (lines 391-429):
```python
def _detect_conflicts(schedule_entries):
    # Group by employee and date (lines 396-405)
    emp_date_map = {}

    # Check for conflicts (lines 408-428)
    for emp, date_map in emp_date_map.items():
        for date_str, entries in date_map.items():
            # Shift during time-off (severity 3)
            if shifts_on_day and timeoffs_on_day:
                conflicts.append(...)

            # Multiple shifts (severity 1)
            if len(shifts_on_day) > 1:
                conflicts.append(...)
```

**Dependencies**: streamlit, streamlit.components.v1, pandas, datetime, models

---

### `translations.py` - UI Translations (456 lines)

**Purpose**: Complete English/German translations for UI

**Structure**:
- `TRANSLATIONS` dict (lines 6-451): Two-level dict (lang → key → text)
- Supported languages: "en", "de"

**Translation Categories**:
- General (app_title, app_caption)
- Sidebar (project_name, version, save/load)
- Tab names (11 tabs)
- Employee tab (25+ keys)
- Shifts tab (15+ keys)
- Rules, Import, Planning, Prompt tabs
- LLM Settings tab (30+ keys)
- Chat tab (15+ keys)
- Generate, Preview, Export tabs
- Messages (success, error, warnings)

**Function**:
- `get_text(key, lang)` (453-455): Retrieves translation with fallback to English

**Dependencies**: None

---

### `mcp_config.py` - MCP Integration (133 lines)

**Purpose**: Model Context Protocol server configuration

**Note**: MCP integration is configured but not fully implemented. Placeholder for future enhancement.

**Functions**:
- `validate_mcp_config(server)` (12-25): Validates server config
- `format_mcp_tools_for_prompt(mcp_servers)` (28-55): Formats for prompt inclusion
- `connect_to_mcp_servers(servers)` (58-97): Placeholder connection logic
- `get_mcp_server_examples()` (100-132): Example configurations

**Example Servers**:
- Filesystem, Web Search, Database, Git

**Dependencies**: models (MCPServerConfig)

---

## Data Flows

### 1. Schedule Import Flow

```
User uploads Excel → parse_teams_excel_multisheet()
                            ↓
                  Normalize columns (_normalize_cols)
                            ↓
                  Detect sheets (Schichten, Arbeitsfreie Zeit, Mitglieder)
                            ↓
                  Parse each sheet (_parse_teams_file)
                            ↓
                  Split by today (past/future)
                            ↓
                  Compute fairness hints (_compute_fairness_hints)
                            ↓
                  Store in session_state.schedule_payload
                            ↓
                  Auto-populate employees (auto_populate_employees_from_members)
                            ↓
                  Auto-detect shift patterns (detect_shift_patterns_from_schedule)
                            ↓
                  Generate preview (generate_schedule_preview)
```

### 2. Schedule Generation Flow (LLM with Solver)

```
User configures solver → Enable in tab 5
                            ↓
                  Set weights, rules, backend, timeout
                            ↓
User clicks Generate → call_llm_sync(enable_tools=True)
                            ↓
                  build_system_prompt() includes SOLVER_TOOL_DEFINITION
                            ↓
                  call_llm_with_reasoning(enable_tools=True)
                            ↓
        ┌──────────────────────────────────────┐
        │ LLM processes prompt                 │
        │ Decides to call solve_with_minizinc  │
        │ Constructs SolverRequest JSON        │
        └──────────────────────────────────────┘
                            ↓
                  _execute_tool("solve_with_minizinc", tool_input)
                            ↓
                  Inject session state config (weights, rules, backend, timeout)
                            ↓
                  SolverRequest.model_validate(tool_input)
                            ↓
                  solve_with_minizinc(request)
                            ↓
        ┌──────────────────────────────────────┐
        │ Build MiniZinc parameters            │
        │ Load shift_schedule.mzn model        │
        │ Create solver instance               │
        │ Execute with timeout                 │
        │ Parse result → SolverResponse        │
        └──────────────────────────────────────┘
                            ↓
                  SolverResponse.model_dump() returned as tool result
                            ↓
                  Tool result sent back to LLM
                            ↓
                  LLM interprets solver output, formats final schedule
                            ↓
                  Final response with tool_calls metadata
                            ↓
                  Display in UI with tool usage section
```

### 3. Export Flow

```
Generated schedule → User selects export format
                            ↓
                  export_to_teams_excel_multisheet()
                            ↓
                  Separate shifts and time-off
                            ↓
                  _prepare_shifts_dataframe()
                  _prepare_timeoff_dataframe()
                  _prepare_members_dataframe()
                            ↓
                  Format dates (M/D/YYYY)
                  Format times (HH:MM)
                  Format colors ("1. Weiß")
                            ↓
                  Write to Excel with three sheets:
                  - Schichten
                  - Arbeitsfreie Zeit
                  - Mitglieder
                            ↓
                  User downloads file
```

---

## Key Integration Points

### 1. Solver ↔ LLM Integration

**Location**: `llm_manager.py` lines 353-534

**Flow**:
1. Tool definition registered in `_get_available_tools()` (lines 353-383)
2. LLM receives tool in system prompt via `get_solver_tool_definition_if_enabled()` (prompt_templates.py lines 12-39)
3. LLM decides to call tool and returns `tool_use` stop reason
4. `_handle_tool_calls_claude()` extracts tool calls (lines 448-534)
5. `_execute_tool()` calls `solve_with_minizinc()` (lines 386-445)
6. Solver result returned as JSON to LLM
7. LLM continues with tool results

**Session State Injection** (lines 404-430):
- Weights from UI sliders
- Constraint rules (consecutive late, pikett gaps, French dispatchers)
- Solver backend and timeout

### 2. Teams Import ↔ Employee Management

**Location**: `utils.py` lines 644-767

**Flow**:
1. Parse Teams Excel → extract members sheet
2. `auto_populate_employees_from_members()` creates Employee objects (lines 644-707)
3. Duplicate detection via `find_duplicate_employee()` (name + email match) (lines 618-642)
4. New employees added to project.employees
5. UI refreshes with new employees

### 3. Shift Pattern Detection ↔ Shift Templates

**Location**: `utils.py` lines 772-948

**Flow**:
1. Parse schedule data (past + future entries)
2. Group shift entries by pattern key (time + role + color)
3. Track occurrences, weekdays, employees (lines 805-866)
4. Create ShiftTemplate objects for patterns with ≥2 occurrences
5. Add to project.shifts (skip if already exists)
6. Return detection summary

### 4. Calendar Preview ↔ Schedule Data

**Location**: `preview.py` lines 16-348

**Flow**:
1. Convert raw entries to ScheduleEntry objects
2. Build employee × date grid (lines 104-125)
3. Generate HTML table with Teams styling (lines 128-347)
4. Apply color codes from TEAMS_COLOR_NAMES
5. Calculate employee hours from shift times (lines 233-250)
6. Render with streamlit.components.v1

### 5. Prompt Construction ↔ Project Data

**Location**: `prompt_templates.py` lines 111-161

**Flow**:
1. `project.as_compact_json()` → minimal JSON (models.py lines 202-208)
2. Build planning period context (dates, weekdays) (lines 118-138)
3. Format SYSTEM_TEMPLATE with preamble, rules, output format (lines 140-146)
4. Append schedule addendum if available (lines 149-154)
5. Inject solver tool definition if enabled (lines 156-159)
6. Return complete prompt string

---

## External Dependencies

**Core Requirements** (`requirements.txt`):
- `streamlit>=1.28.0` - Web UI framework
- `pandas>=2.0.0` - Data manipulation
- `pydantic>=2.0.0` - Data validation
- `openpyxl>=3.1.0` - Excel I/O
- `openai>=1.0.0` - LLM client SDK
- `requests>=2.31.0` - HTTP requests
- `minizinc>=0.10.0` - Constraint solver (optional)

**Optional Dependencies**:
- `anthropic>=0.40.0` - Claude extended thinking
- MCP package - Model Context Protocol (not yet installed)

---

## Configuration Files

### `CLAUDE.md` - Project Instructions
Comprehensive guide for Claude Code sessions:
- Required reading: projectstructure.md, knowledge.md
- Development setup
- Architecture overview
- Teams color codes
- Version control workflow

### `knowledge.md` - Technical Knowledge Base
Technical specifications and integration patterns

### `requirements.txt` - Python Dependencies
All required and optional packages

### Documentation Files 🆕
- **SOLVER_SETUP.md**: Installation guide for MiniZinc
- **SOLVER_INTEGRATION_SUMMARY.md**: Architecture and design decisions
- **SOLVER_INTEGRATION_COMPLETE.md**: Testing guide and examples

---

## Testing & Documentation

### Testing Strategy
- Manual testing via Streamlit UI
- Solver validation via MiniZinc test cases
- LLM integration testing via Chat tab

---

## Entry Points

### Main Application
**Command**: `streamlit run app.py`
- Launches web UI on http://localhost:8501
- Initializes session state (lines 68-105)
- Loads default project

### Solver Execution
**Location**: `solver_service.py:23`
- Called by `_execute_tool()` in llm_manager.py (line 434)
- Can be tested standalone via Python import

### Export Utility
**Location**: `export_teams.py:380`
- `create_sample_export()` for testing
- Can be run standalone: `python export_teams.py`

---

## Architectural Patterns

### 1. Pydantic Models Everywhere
All data structures use Pydantic for validation:
- Type safety at runtime
- Automatic JSON serialization
- `model_dump()` for dict conversion
- `model_validate()` for parsing

### 2. Session State Management
Streamlit session state for persistence:
- `st.session_state.project` - Current project
- `st.session_state.schedule_payload` - Imported schedule
- `st.session_state.generated_entries` - LLM output
- `st.session_state.llm_client` - LLM client instance
- `st.session_state.enable_solver` - Solver mode toggle

### 3. Graceful Degradation
Optional features fail gracefully:
- MiniZinc solver: Check availability before use
- MCP integration: Placeholder for future
- Extended thinking: Model-specific features

### 4. Multi-Provider Abstraction
LLM client supports multiple providers:
- Unified `LLMClient` interface
- Provider-specific initialization
- Common completion and streaming methods

### 5. Tool Calling Protocol
Claude-compatible tool calling:
- Tool definition in system prompt
- Stop reason detection
- Tool execution and result injection
- Conversation continuation

### 6. Data Transformation Pipeline
Import → Transform → Generate → Export:
- Flexible column detection
- Normalization layers
- Bidirectional conversion (Teams ↔ Internal ↔ Solver)

---

## Performance Considerations

### 1. Schedule Parsing
- Capped at 800 past entries and 800 future entries
- Fairness hints computed only from last 14 days
- Column normalization cached during parsing

### 2. Calendar Rendering
- Dynamic height based on employee count (max 800px)
- Week-by-week view (7 days at a time)
- HTML rendering via components.html

### 3. LLM Client
- Automatic max_tokens fallback to model limits
- Exponential backoff for rate limits (3 retries)
- Streaming support for better UX

### 4. Solver Execution
- Default timeout: 15 seconds
- Configurable time limit per request
- Early termination on timeout

---

## Summary

**Shift Prompt Studio** is a sophisticated shift planning application that combines:

1. **AI-Driven Generation**: LLM-powered schedule creation with context-aware prompts
2. **Mathematical Optimization**: Optional constraint solver for provably optimal schedules
3. **Teams Integration**: Full import/export compatibility with Microsoft Teams Shifts
4. **Interactive Refinement**: Chat interface for conversational schedule adjustments
5. **Visual Preview**: Teams Shifts-style calendar with conflict detection

The architecture is modular, with clear separation between data models, utilities, LLM integration, solver execution, and UI. Key design principles include graceful degradation, multi-provider support, and extensive validation through Pydantic models.

**For Claude Code sessions**: Always consult this file first to locate functionality with exact line numbers, then refer to `knowledge.md` for implementation guidance.

---

**End of Project Structure Documentation**
