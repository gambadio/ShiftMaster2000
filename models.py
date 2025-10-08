from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict

ShiftID = str

class ShiftTemplate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: ShiftID
    role: str
    start_time: str  # "HH:MM"
    end_time: str    # "HH:MM"
    weekdays: List[str] = Field(default_factory=lambda: ["Mon","Tue","Wed","Thu","Fri"])
    required_count: Dict[str, int] = Field(default_factory=dict)  # per weekday required headcount
    notes: Optional[str] = None

class Employee(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
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
    version: str = "1.0"
    employees: List[Employee] = Field(default_factory=list)
    shifts: List[ShiftTemplate] = Field(default_factory=list)
    global_rules: RuleSet = Field(default_factory=RuleSet)
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    def as_compact_json(self) -> Dict[str, Any]:
        return {
            "employees":[e.model_dump(exclude_none=True) for e in self.employees],
            "shifts":[s.model_dump(exclude_none=True) for s in self.shifts],
            "meta":{"version": self.version, "project": self.name},
        }
