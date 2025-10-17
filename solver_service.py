"""
MiniZinc constraint solver service for shift scheduling.

This module provides the bridge between the LLM's JSON requests and the MiniZinc solver.
The LLM decides "what to solve", this module decides "how to solve".
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from solver_models import (
    SolverRequest, SolverResponse, SolverStatus, Assignment,
    Violation, PenaltyBreakdown, SolverStats, SolverBackend
)
from solver_utils import check_minizinc_available

logger = logging.getLogger(__name__)


def solve_with_minizinc(request: SolverRequest) -> SolverResponse:
    """
    Execute constraint optimization using MiniZinc solver.

    This is the main entry point called by the LLM tool system.

    Args:
        request: Validated SolverRequest with all scheduling data

    Returns:
        SolverResponse with assignments, violations, and statistics
    """
    # Check if MiniZinc is available
    is_available, message = check_minizinc_available()
    if not is_available:
        return SolverResponse(
            status=SolverStatus.ERROR,
            stats=SolverStats(solver="none", time_ms=0),
            message=f"MiniZinc not available: {message}"
        )

    try:
        # Import minizinc only when needed (graceful degradation)
        import minizinc
        from datetime import timedelta as td

        # Build index maps
        emp_to_idx, idx_to_emp = _build_employee_index(request.employees)
        shift_to_idx, idx_to_shift = _build_shift_index(request.shifts)
        date_list = _build_date_range(request.horizon)

        # Map data to MiniZinc parameters
        params = _build_minizinc_parameters(
            request, emp_to_idx, shift_to_idx, date_list
        )

        # Load model
        model_path = Path(__file__).parent / "shift_schedule.mzn"
        model = minizinc.Model(model_path)

        # Select solver
        solver = minizinc.Solver.lookup(request.options.solver.value)

        # Create instance and set parameters
        instance = minizinc.Instance(solver, model)
        for key, value in params.items():
            instance[key] = value

        # Solve with timeout
        start_time = datetime.now()
        result = instance.solve(
            timeout=td(milliseconds=request.options.time_limit_ms),
            intermediate_solutions=request.options.intermediate_solutions,
            all_solutions=request.options.all_solutions,
        )
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Parse result
        return _parse_minizinc_result(
            result,
            request,
            idx_to_emp,
            idx_to_shift,
            date_list,
            elapsed_ms
        )

    except Exception as e:
        logger.exception("Solver execution failed")
        return SolverResponse(
            status=SolverStatus.ERROR,
            stats=SolverStats(solver=request.options.solver.value, time_ms=0),
            message=f"Solver error: {str(e)}"
        )


def _build_employee_index(employees: List) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build bidirectional mapping between employee IDs and 1-based indices"""
    emp_to_idx = {emp.id: idx + 1 for idx, emp in enumerate(employees)}
    idx_to_emp = {idx: emp_id for emp_id, idx in emp_to_idx.items()}
    return emp_to_idx, idx_to_emp


def _build_shift_index(shifts: List) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build bidirectional mapping between shift IDs and 1-based indices"""
    shift_to_idx = {shift.id: idx + 1 for idx, shift in enumerate(shifts)}
    idx_to_shift = {idx: shift_id for shift_id, idx in shift_to_idx.items()}
    return shift_to_idx, idx_to_shift


def _build_date_range(horizon: Dict[str, str]) -> List[str]:
    """Build list of dates from horizon (inclusive)"""
    from datetime import date

    start = date.fromisoformat(horizon["from"])
    end = date.fromisoformat(horizon["to"])

    dates = []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)

    return dates


def _build_minizinc_parameters(
    request: SolverRequest,
    emp_to_idx: Dict[str, int],
    shift_to_idx: Dict[str, int],
    date_list: List[str]
) -> Dict[str, Any]:
    """
    Build MiniZinc parameter dictionary from request.

    Converts high-level request data into the low-level arrays expected by the model.
    """
    E = len(request.employees)
    D = len(date_list)
    S = len(request.shifts)

    # Employee attributes (1-indexed arrays)
    fte = [0.0] + [emp.fte for emp in request.employees]  # Prepend dummy for 1-indexing
    target_shifts = [0] + [
        emp.target_weekly_shifts or int(emp.fte * 5)  # Default: FTE * 5 shifts/week
        for emp in request.employees
    ]
    speaks_french = [False] + [
        "fr" in [lang.lower() for lang in emp.languages]
        for emp in request.employees
    ]

    # Shift attributes (1-indexed arrays)
    is_late = [False] + [shift.is_late for shift in request.shifts]
    is_pikett = [False] + [shift.is_pikett for shift in request.shifts]
    needs_french = [False] + [shift.needs_french for shift in request.shifts]

    # Requirements array: required[day][shift]
    # Convert from per-shift required_per_day to 2D array
    required = []
    for day_idx in range(1, D + 1):
        day_reqs = []
        for shift_idx in range(1, S + 1):
            shift = request.shifts[shift_idx - 1]
            # Map day index to weekday (assuming date_list[0] is Monday=0)
            day_of_week = (day_idx - 1) % 7  # 0=Mon, 6=Sun
            if day_of_week < len(shift.required_per_day):
                day_reqs.append(shift.required_per_day[day_of_week])
            else:
                day_reqs.append(0)  # No requirement if not specified
        required.append(day_reqs)

    # Availability mask: available[emp][day][shift]
    # If availability_mask is provided, use it; otherwise assume all available
    available = []
    for emp_idx, emp in enumerate(request.employees, start=1):
        emp_avail = []
        for day_idx in range(1, D + 1):
            day_avail = []
            for shift_idx in range(1, S + 1):
                # Check if availability mask is provided
                if emp.availability_mask and len(emp.availability_mask) >= day_idx - 1:
                    day_mask = emp.availability_mask[day_idx - 1]
                    if len(day_mask) >= shift_idx - 1:
                        day_avail.append(day_mask[shift_idx - 1])
                    else:
                        day_avail.append(1)  # Default: available
                else:
                    day_avail.append(1)  # Default: available
            emp_avail.append(day_avail)
        available.append(emp_avail)

    # Constraint parameters
    rules = request.rules
    weights = rules.weights

    return {
        "E": E,
        "D": D,
        "S": S,
        "fte": fte,
        "target_shifts": target_shifts,
        "speaks_french": speaks_french,
        "is_late": is_late,
        "is_pikett": is_pikett,
        "needs_french": needs_french,
        "required": required,
        "available": available,
        "enforce_no_consecutive_late": rules.no_consecutive_late,
        "pikett_min_gap_days": rules.pikett_gap_days,
        "min_french_per_week": rules.fr_dispatcher_per_week,
        "weight_coverage": weights.get("coverage", 5.0),
        "weight_fairness": weights.get("fairness", 3.0),
        "weight_late": weights.get("late_violation", 2.0),
        "weight_pikett": weights.get("pikett_violation", 2.0),
    }


def _parse_minizinc_result(
    result,
    request: SolverRequest,
    idx_to_emp: Dict[int, str],
    idx_to_shift: Dict[int, str],
    date_list: List[str],
    elapsed_ms: int
) -> SolverResponse:
    """
    Parse MiniZinc result and convert back to high-level response.
    """
    from minizinc import Status

    # Determine status
    if result.status == Status.OPTIMAL_SOLUTION:
        status = SolverStatus.OPTIMAL
        message = "Optimal solution found"
    elif result.status == Status.SATISFIED or result.status == Status.ALL_SOLUTIONS:
        status = SolverStatus.FEASIBLE
        message = "Feasible solution found (may not be optimal)"
    elif result.status == Status.UNSATISFIABLE:
        status = SolverStatus.INFEASIBLE
        message = "No feasible solution exists - constraints are unsatisfiable"
        return SolverResponse(
            status=status,
            stats=SolverStats(
                solver=request.options.solver.value,
                time_ms=elapsed_ms
            ),
            message=message
        )
    else:
        status = SolverStatus.UNKNOWN
        message = f"Solver terminated with status: {result.status}"

    # Extract objective value
    objective = float(result["total_penalty"]) if "total_penalty" in result else None

    # Build penalty breakdown
    breakdown = PenaltyBreakdown(
        fairness_penalty=float(result.get("fairness_deviation_total", 0)),
        late_violation_penalty=float(result.get("late_violations", 0)),
        pikett_violation_penalty=float(result.get("pikett_violations", 0)),
        coverage_penalty=float(result.get("coverage_deficit_total", 0)),
        total=objective or 0.0
    )

    # Extract assignments
    assignments = []
    x = result["x"]  # 3D array[E][D][S]

    E = len(request.employees)
    D = len(date_list)
    S = len(request.shifts)

    for e_idx in range(1, E + 1):
        for d_idx in range(1, D + 1):
            for s_idx in range(1, S + 1):
                if x[e_idx - 1][d_idx - 1][s_idx - 1] == 1:
                    assignments.append(Assignment(
                        date=date_list[d_idx - 1],
                        employee_id=idx_to_emp[e_idx],
                        shift_id=idx_to_shift[s_idx]
                    ))

    # Build violations list
    violations = []
    if breakdown.coverage_penalty > 0:
        violations.append(Violation(
            type="coverage",
            details=f"{breakdown.coverage_penalty} total coverage deficit across all shifts",
            severity=breakdown.coverage_penalty
        ))
    if breakdown.fairness_penalty > 0:
        violations.append(Violation(
            type="fairness",
            details=f"{breakdown.fairness_penalty} total fairness deviation from target shift counts",
            severity=breakdown.fairness_penalty
        ))
    if breakdown.late_violation_penalty > 0:
        violations.append(Violation(
            type="consecutive_late",
            details=f"{int(breakdown.late_violation_penalty)} consecutive late shift violations",
            severity=breakdown.late_violation_penalty
        ))
    if breakdown.pikett_violation_penalty > 0:
        violations.append(Violation(
            type="pikett_gap",
            details=f"{int(breakdown.pikett_violation_penalty)} pikett gap violations",
            severity=breakdown.pikett_violation_penalty
        ))

    # Build stats
    stats = SolverStats(
        solver=request.options.solver.value,
        nodes=result.statistics.get("nodes", 0) if hasattr(result, "statistics") else None,
        failures=result.statistics.get("failures", 0) if hasattr(result, "statistics") else None,
        time_ms=elapsed_ms,
        timeout_reached=(elapsed_ms >= request.options.time_limit_ms * 0.95)
    )

    return SolverResponse(
        status=status,
        objective=objective,
        breakdown=breakdown,
        assignments=assignments,
        violations=violations,
        stats=stats,
        message=message
    )


# === Helper function for LLM integration ===

def call_solver_from_json(request_json: str) -> str:
    """
    Call solver from JSON string (for LLM tool calling).

    Args:
        request_json: JSON string with SolverRequest data

    Returns:
        JSON string with SolverResponse data
    """
    try:
        request_dict = json.loads(request_json)
        request = SolverRequest.model_validate(request_dict)
        response = solve_with_minizinc(request)
        return response.model_dump_json(indent=2)
    except Exception as e:
        error_response = SolverResponse(
            status=SolverStatus.ERROR,
            stats=SolverStats(solver="none", time_ms=0),
            message=f"Error parsing request or executing solver: {str(e)}"
        )
        return error_response.model_dump_json(indent=2)
