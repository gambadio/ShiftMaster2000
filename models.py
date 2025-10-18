from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import date

ShiftID = str

class TeamsColor(str, Enum):
    """Microsoft Teams Shift Colors with German names"""
    WEISS = "1"              # Weiß - Operation Lead
    BLAU = "2"               # Blau - Contact Team, Dispatcher (07:00-16:00)
    GRUEN = "3"              # Grün - Contact Team, SOB roles
    LILA = "4"               # Lila - Late shifts (10:00-19:00)
    ROSA = "5"               # Rosa - Special assignments
    GELB = "6"               # Gelb - Late shifts (09:00-18:00)
    DUNKELBLAU = "8"         # Dunkelblau - Project work
    DUNKELGRUEN = "9"        # Dunkelgrün - WoVe, PCV roles
    DUNKELVIOLETT = "10"     # Dunkelviolett - Pikett
    DUNKELROSA = "11"        # Dunkelrosa - People Developer
    DUNKELGELB = "12"        # Dunkelgelb - Livechat shifts
    GRAU = "13"              # Grau - Time-off

TEAMS_COLOR_NAMES = {
    "1": "Weiß",
    "2": "Blau",
    "3": "Grün",
    "4": "Lila",
    "5": "Rosa",
    "6": "Gelb",
    "8": "Dunkelblau",
    "9": "Dunkelgrün",
    "10": "Dunkelviolett",
    "11": "Dunkelrosa",
    "12": "Dunkelgelb",
    "13": "Grau"
}

class ShiftTemplate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: ShiftID
    role: str
    start_time: str  # "HH:MM"
    end_time: str    # "HH:MM"
    weekdays: List[str] = Field(default_factory=lambda: ["Mon","Tue","Wed","Thu","Fri"])
    required_count: Dict[str, int] = Field(default_factory=dict)  # per weekday required headcount
    notes: Optional[str] = None
    # Teams-specific fields
    color_code: Optional[str] = None  # "1" to "13" for Teams colors
    unpaid_break_minutes: Optional[int] = None
    teams_label: Optional[str] = None  # Display label for Teams
    shared_status: Optional[str] = "1. Geteilt"  # Default shared status
    concurrent_shifts: List[str] = Field(default_factory=list)  # IDs of shifts that can run simultaneously

class Employee(BaseModel):
    id: str
    name: str
    email: Optional[str] = None  # Required for Teams export
    percent: Optional[int] = None  # e.g., 60, 80, 100
    roles: List[str] = Field(default_factory=list)  # allowed roles
    languages: List[str] = Field(default_factory=list)
    earliest_start: Optional[str] = None  # "07:00"
    latest_end: Optional[str] = None      # "19:00"
    weekday_blockers: Dict[str, str] = Field(default_factory=dict)  # {"Tue": "school 7:00-12:00"}
    shift_time_preferences: List[Tuple[str, str]] = Field(default_factory=list)  # [("07:00","16:00")]
    hard_constraints: List[str] = Field(default_factory=list)  # free-text
    soft_preferences: List[str] = Field(default_factory=list)  # free-text
    tags: List[str] = Field(default_factory=list)
    # Teams-specific fields
    group: Optional[str] = None  # Department/team grouping
    teams_color: Optional[str] = None  # Optional color override

class ScheduleEntry(BaseModel):
    """Unified representation of shifts and time-off"""
    model_config = ConfigDict(extra="ignore")
    employee_name: str
    employee_email: Optional[str] = None
    group: Optional[str] = None
    start_date: str  # ISO format YYYY-MM-DD
    start_time: Optional[str] = None  # HH:MM
    end_date: str  # ISO format YYYY-MM-DD
    end_time: Optional[str] = None  # HH:MM
    color_code: Optional[str] = None  # "1" to "13"
    label: Optional[str] = None  # Bezeichnung
    unpaid_break: Optional[int] = None
    notes: Optional[str] = None
    shared: Optional[str] = "1. Geteilt"
    entry_type: str = "shift"  # "shift" or "time_off"
    reason: Optional[str] = None  # For time-off entries (Grund für arbeitsfreie Zeit)

class PlanningPeriod(BaseModel):
    """Date range for schedule generation"""
    start_date: date
    end_date: date

class MCPServerConfig(BaseModel):
    """MCP Server configuration"""
    name: str
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)

class ProviderType(str, Enum):
    """Supported LLM provider types"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    AZURE = "azure"
    CUSTOM = "custom"

class LLMProviderConfig(BaseModel):
    """Provider-specific configuration"""
    model_config = ConfigDict(extra="ignore")

    provider: ProviderType = ProviderType.OPENAI
    api_key: str = ""

    # OpenAI/OpenRouter/Custom
    base_url: Optional[str] = None  # Custom endpoint or OpenRouter URL

    # Azure-specific
    azure_endpoint: Optional[str] = None  # https://YOUR-RESOURCE.openai.azure.com/
    azure_deployment: Optional[str] = None  # Deployment name
    api_version: str = "2024-10-21"  # Azure API version

    # OpenRouter-specific
    http_referer: Optional[str] = None  # For rankings
    x_title: Optional[str] = None  # App name on openrouter.ai

    # Model selection
    model: str = "gpt-4o"  # Default model or deployment name
    available_models: List[str] = Field(default_factory=list)  # Cached model list

    def get_base_url(self) -> Optional[str]:
        """Get the appropriate base URL for this provider"""
        if self.provider == ProviderType.OPENROUTER:
            return "https://openrouter.ai/api/v1"
        return self.base_url

class LLMConfig(BaseModel):
    """Complete LLM configuration for scheduling tasks"""
    model_config = ConfigDict(extra="ignore")

    # Provider configuration
    provider_config: LLMProviderConfig = Field(default_factory=LLMProviderConfig)

    # Generation parameters
    temperature: float = 0.2
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Response format
    json_mode: bool = True  # Request JSON output
    enable_streaming: bool = True  # Stream responses

    # Reasoning/Thinking parameters (OpenAI o1/o3, OpenRouter, Claude)
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" (OpenAI/OpenRouter)
    reasoning_max_tokens: Optional[int] = None  # Max tokens for reasoning (OpenRouter/Claude)
    reasoning_exclude: bool = False  # Exclude reasoning from response (OpenRouter)

    # Claude-specific extended thinking
    budget_tokens: Optional[int] = None  # Claude: 1024-10000+ for extended thinking

    # Advanced features
    seed: Optional[int] = None  # For reproducible outputs
    stop_sequences: List[str] = Field(default_factory=list)

    # MCP integration
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)
    enable_mcp: bool = False

class ChatMessage(BaseModel):
    """Chat message in conversation history"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: Optional[str] = None
    reasoning_tokens: Optional[int] = None  # For o1/o3 models

class ChatSession(BaseModel):
    """Chat session state for interactive planning"""
    messages: List[ChatMessage] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    session_id: Optional[str] = None

class RuleSet(BaseModel):
    preamble: str = "You are a meticulous, constraint-aware shift planner."
    narrative_rules: str = ""  # large free-text rules
    output_format_instructions: str = (
        "Return a valid JSON schedule. Top-level keys: dates[], assignments[], "
        "violations[], and notes. Each assignment maps (date, role, shift_id) -> employee_id."
    )
    token_budgets: Dict[str, int] = Field(default_factory=lambda: {"max_prompt_chars": 120000})

class Project(BaseModel):
    name: str = "Untitled"
    version: str = "2.0"  # Updated version
    employees: List[Employee] = Field(default_factory=list)
    shifts: List[ShiftTemplate] = Field(default_factory=list)
    global_rules: RuleSet = Field(default_factory=RuleSet)
    custom_data: Dict[str, Any] = Field(default_factory=dict)
    # New fields for enhanced functionality
    llm_config: Optional[LLMConfig] = None
    planning_period: Optional[PlanningPeriod] = None

    def as_compact_json(self) -> Dict[str, Any]:
        return {
            "employees":[e.model_dump(exclude_none=True) for e in self.employees],
            "shifts":[s.model_dump(exclude_none=True) for s in self.shifts],
            "meta":{"version": self.version, "project": self.name},
        }
