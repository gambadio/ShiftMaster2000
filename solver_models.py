"""
Pydantic models for MiniZinc constraint solver tool integration.

These models define the JSON contract between the LLM and the constraint solver.
The LLM sends structured data, the solver returns structured results.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from enum import Enum


class SolverBackend(str, Enum):
    """Available MiniZinc solver backends"""
    GECODE = "gecode"
    CHUFFED = "chuffed"
    COIN_BC = "coin-bc"


class SolverStatus(str, Enum):
    """Possible solver result statuses"""
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class EmployeeData(BaseModel):
    """Canonicalized employee data for solver"""
    model_config = ConfigDict(extra="forbid")

    id: str
    fte: float = 1.0  # 0.0-1.0 or percentage/100
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    # Availability mask: [day][shift] -> bool (available=1, unavailable=0)
    availability_mask: List[List[int]] = Field(default_factory=list)
    target_weekly_shifts: Optional[int] = None  # For fairness calculation


class ShiftData(BaseModel):
    """Canonicalized shift template data for solver"""
    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    start_time: str  # "HH:MM"
    end_time: str  # "HH:MM"
    is_late: bool = False  # Shifts starting after 09:00
    is_pikett: bool = False  # On-call duty
    is_dispatcher: bool = False
    needs_french: bool = False  # Requires French-speaking employee
    # Required headcount per day [Mon, Tue, Wed, Thu, Fri, Sat, Sun]
    required_per_day: List[int] = Field(default_factory=list)


class ConstraintRules(BaseModel):
    """Soft constraint toggles and penalty weights"""
    model_config = ConfigDict(extra="forbid")

    # Constraint toggles
    no_consecutive_late: bool = True
    pikett_gap_days: int = 7  # Minimum days between pikett assignments
    fr_dispatcher_per_week: int = 1  # Min French speakers on dispatcher per week
    max_shifts_per_week: Optional[int] = 6

    # Penalty weights for soft constraints (higher = more important)
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "fairness": 3.0,
        "late_violation": 2.0,
        "pikett_violation": 2.0,
        "preference_violation": 1.0,
        "overtime": 2.5
    })


class SolverOptions(BaseModel):
    """Solver execution options"""
    model_config = ConfigDict(extra="forbid")

    solver: SolverBackend = SolverBackend.CHUFFED
    time_limit_ms: int = 15000  # 15 seconds default
    all_solutions: bool = False  # Find all solutions (satisfaction only)
    intermediate_solutions: bool = False  # Return optimization progress
    num_solutions: int = 1  # How many solutions to find


class SolverRequest(BaseModel):
    """Complete request payload for solve_with_minizinc tool"""
    model_config = ConfigDict(extra="forbid")

    horizon: Dict[str, str] = Field(..., description="Date range: {from: YYYY-MM-DD, to: YYYY-MM-DD}")
    employees: List[EmployeeData] = Field(..., description="List of canonicalized employee data")
    shifts: List[ShiftData] = Field(..., description="List of shift templates with requirements")
    rules: ConstraintRules = Field(default_factory=ConstraintRules, description="Constraint rules and weights")
    options: SolverOptions = Field(default_factory=SolverOptions, description="Solver execution options")


class Assignment(BaseModel):
    """A single shift assignment"""
    date: str  # YYYY-MM-DD
    employee_id: str
    shift_id: str


class Violation(BaseModel):
    """A soft constraint violation"""
    type: str  # "fairness", "late_consecutive", "pikett_gap", etc.
    details: str  # Human-readable description
    employee_id: Optional[str] = None
    severity: float = 1.0  # Penalty contribution


class PenaltyBreakdown(BaseModel):
    """Breakdown of objective function penalties"""
    fairness_penalty: float = 0.0
    late_violation_penalty: float = 0.0
    pikett_violation_penalty: float = 0.0
    coverage_penalty: float = 0.0
    overtime_penalty: float = 0.0
    preference_penalty: float = 0.0
    total: float = 0.0


class SolverStats(BaseModel):
    """Solver execution statistics"""
    solver: str
    nodes: Optional[int] = None  # Search tree nodes explored
    failures: Optional[int] = None  # Failed branches
    time_ms: int = 0
    timeout_reached: bool = False


class SolverResponse(BaseModel):
    """Response from solve_with_minizinc tool"""
    model_config = ConfigDict(extra="ignore")

    status: SolverStatus
    objective: Optional[float] = None  # Lower is better (penalty sum)
    breakdown: Optional[PenaltyBreakdown] = None
    assignments: List[Assignment] = Field(default_factory=list)
    violations: List[Violation] = Field(default_factory=list)
    stats: SolverStats
    message: str = ""  # Human-readable status message


# Tool definition for LLM system prompt
SOLVER_TOOL_DEFINITION = """
## solve_with_minizinc

**Purpose:** Generate an optimized shift schedule using constraint programming.

**When to use:**
- Creating schedules with complex fairness and rotation requirements
- Optimizing assignments under hard constraints (coverage, availability, skills)
- Need provably optimal or near-optimal solutions
- Handling infeasibility analysis and conflict detection

**Input Schema (JSON):**
```json
{
  "horizon": {
    "from": "YYYY-MM-DD",
    "to": "YYYY-MM-DD"
  },
  "employees": [
    {
      "id": "employee_id",
      "fte": 1.0,
      "skills": ["dispatcher", "contact"],
      "languages": ["de", "fr"],
      "availability_mask": [[1,1,1,0], [1,1,1,0], ...]  // [day][shift]
    }
  ],
  "shifts": [
    {
      "id": "shift_id",
      "role": "Contact Team",
      "start_time": "07:00",
      "end_time": "16:00",
      "is_late": false,
      "is_pikett": false,
      "needs_french": false,
      "required_per_day": [2, 2, 2, 2, 2, 0, 0]  // Mon-Sun
    }
  ],
  "rules": {
    "no_consecutive_late": true,
    "pikett_gap_days": 7,
    "fr_dispatcher_per_week": 1,
    "weights": {
      "fairness": 3.0,
      "late_violation": 2.0,
      "overtime": 2.5
    }
  },
  "options": {
    "solver": "chuffed",
    "time_limit_ms": 15000
  }
}
```

**Output Schema (JSON):**
```json
{
  "status": "OPTIMAL",
  "objective": 12.5,
  "breakdown": {
    "fairness_penalty": 8.0,
    "late_violation_penalty": 4.5,
    "coverage_penalty": 0.0,
    "total": 12.5
  },
  "assignments": [
    {"date": "2025-01-20", "employee_id": "alice", "shift_id": "dispatch-0700"}
  ],
  "violations": [
    {"type": "fairness", "details": "Bob assigned 3 shifts, target was 5", "severity": 2.0}
  ],
  "stats": {
    "solver": "chuffed",
    "nodes": 15234,
    "time_ms": 3456
  },
  "message": "Optimal solution found within 3.5 seconds"
}
```

**Handling Results:**
- `status = "OPTIMAL"`: Best possible schedule found
- `status = "FEASIBLE"`: Valid schedule found, may not be optimal (timeout)
- `status = "INFEASIBLE"`: No valid schedule exists - analyze violations to suggest fixes
- `status = "TIMEOUT"`: Solver ran out of time, may need to relax constraints

**Best Practices:**
1. Build availability masks carefully (respect employee time windows, blockers)
2. If INFEASIBLE, check: coverage requirements vs. available staff, skill matches, conflicting constraints
3. Adjust weights to balance fairness vs. other objectives
4. Use higher time_limit_ms for larger problems (7+ days, 20+ employees)
"""
