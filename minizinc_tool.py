"""
MiniZinc Tool for LLM-based Constraint Solving

Allows the LLM to write and execute MiniZinc models for schedule optimization.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import json
import tempfile
import os
from datetime import datetime


# OpenAI Function Calling schema for MiniZinc
MINIZINC_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_minizinc",
        "description": """Execute a MiniZinc constraint satisfaction or optimization model.

Use this tool to solve complex scheduling constraints like:
- Employee availability and time-off conflicts
- Shift coverage requirements (minimum/maximum staff per shift)
- Fairness distribution (equal distribution of shifts, late shifts, weekend work)
- Minimum rest periods between shifts
- Role-based assignment constraints
- Weekly hour limits

The model should:
1. Define decision variables (who works which shift)
2. Define constraints based on the scheduling rules
3. Optionally define an objective function to optimize

Return the solution as JSON that can be converted to schedule entries.""",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": """Complete MiniZinc model code. Example:
```minizinc
% Decision variables
array[1..num_employees, 1..num_days] of var 0..num_shifts: schedule;

% Constraints
constraint forall(d in 1..num_days)(
    sum(e in 1..num_employees)(schedule[e,d] > 0) >= min_coverage
);

% Solve
solve satisfy;

% Output
output [show(schedule)];
```"""
                },
                "data": {
                    "type": "object",
                    "description": "Optional data parameters to pass to the model (employees, shifts, dates, etc.)",
                    "additionalProperties": True
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                }
            },
            "required": ["model"]
        }
    }
}


def check_minizinc_available() -> tuple:
    """
    Check if MiniZinc is available on the system.

    Returns:
        Tuple of (is_available: bool, message: str)
    """
    try:
        import minizinc
        solver = minizinc.Solver.lookup("gecode")
        return True, f"MiniZinc available with solver: {solver.name}"
    except ImportError:
        return False, "MiniZinc Python package not installed. Install with: pip install minizinc"
    except Exception as e:
        return False, f"MiniZinc not properly configured: {e}. Please install MiniZinc: https://www.minizinc.org/software.html"


def run_minizinc(
    model: str,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute a MiniZinc model and return the solution.

    Args:
        model: Complete MiniZinc model code
        data: Optional data parameters for the model
        timeout: Timeout in seconds

    Returns:
        Dictionary with:
        - success: bool
        - solution: dict (if successful)
        - error: str (if failed)
        - statistics: dict (solver statistics)
    """
    try:
        import minizinc
    except ImportError:
        return {
            "success": False,
            "error": "MiniZinc Python package not installed. Install with: pip install minizinc",
            "solution": None,
            "statistics": {}
        }

    try:
        # Create a temporary file for the model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mzn', delete=False) as f:
            f.write(model)
            model_file = f.name

        try:
            # Load the model
            mzn_model = minizinc.Model(model_file)

            # Find a solver (prefer Gecode, fall back to others)
            try:
                solver = minizinc.Solver.lookup("gecode")
            except:
                try:
                    solver = minizinc.Solver.lookup("chuffed")
                except:
                    solver = minizinc.Solver.lookup("coinbc")

            # Create instance
            instance = minizinc.Instance(solver, mzn_model)

            # Add data if provided
            if data:
                for key, value in data.items():
                    instance[key] = value

            # Solve with timeout
            from datetime import timedelta
            result = instance.solve(timeout=timedelta(seconds=timeout))

            # Check result status
            if result.status == minizinc.Status.SATISFIED or result.status == minizinc.Status.OPTIMAL_SOLUTION:
                # Extract solution
                solution = {}
                if hasattr(result, 'solution') and result.solution:
                    # Convert solution to dict
                    for var_name in dir(result.solution):
                        if not var_name.startswith('_'):
                            try:
                                value = getattr(result.solution, var_name)
                                # Convert numpy arrays if present
                                if hasattr(value, 'tolist'):
                                    value = value.tolist()
                                solution[var_name] = value
                            except:
                                pass

                return {
                    "success": True,
                    "solution": solution,
                    "status": str(result.status),
                    "error": None,
                    "statistics": {
                        "solve_time": str(result.statistics.get("solveTime", "unknown")),
                        "nodes": result.statistics.get("nodes", 0),
                        "failures": result.statistics.get("failures", 0)
                    }
                }

            elif result.status == minizinc.Status.UNSATISFIABLE:
                return {
                    "success": False,
                    "error": "The constraints are unsatisfiable. The scheduling requirements cannot be met with the given constraints.",
                    "solution": None,
                    "status": "UNSATISFIABLE",
                    "statistics": {}
                }

            elif result.status == minizinc.Status.UNKNOWN:
                return {
                    "success": False,
                    "error": f"Solver timed out after {timeout} seconds. Try simplifying the model or increasing timeout.",
                    "solution": None,
                    "status": "TIMEOUT",
                    "statistics": {}
                }

            else:
                return {
                    "success": False,
                    "error": f"Solver returned status: {result.status}",
                    "solution": None,
                    "status": str(result.status),
                    "statistics": {}
                }

        finally:
            # Clean up temp file
            try:
                os.unlink(model_file)
            except:
                pass

    except minizinc.MiniZincError as e:
        return {
            "success": False,
            "error": f"MiniZinc error: {str(e)}",
            "solution": None,
            "statistics": {}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution error: {str(e)}",
            "solution": None,
            "statistics": {}
        }


def process_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a tool call from the LLM and return the result.

    Args:
        tool_call: The tool call object from OpenAI API
            {
                "id": "call_xxx",
                "type": "function",
                "function": {
                    "name": "run_minizinc",
                    "arguments": "{\"model\": \"...\", \"data\": {...}}"
                }
            }

    Returns:
        Tool result to send back to the LLM
    """
    try:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")

        if function_name != "run_minizinc":
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

        # Execute the model
        model = args.get("model", "")
        data = args.get("data")
        timeout = args.get("timeout", 30)

        if not model:
            return {
                "success": False,
                "error": "No model provided"
            }

        return run_minizinc(model, data, timeout)

    except Exception as e:
        return {
            "success": False,
            "error": f"Tool call processing error: {str(e)}"
        }


# Example model for schedule optimization
EXAMPLE_SCHEDULE_MODEL = """
% Example: Simple shift assignment with coverage constraints

% Parameters
int: num_employees;
int: num_days;
int: min_coverage;  % minimum employees per day
set of int: EMPLOYEES = 1..num_employees;
set of int: DAYS = 1..num_days;

% Decision variables: 1 if employee e works on day d, 0 otherwise
array[EMPLOYEES, DAYS] of var 0..1: works;

% Constraint: Each day must have at least min_coverage employees
constraint forall(d in DAYS)(
    sum(e in EMPLOYEES)(works[e, d]) >= min_coverage
);

% Constraint: Each employee works at most 5 days
constraint forall(e in EMPLOYEES)(
    sum(d in DAYS)(works[e, d]) <= 5
);

% Objective: Minimize total assignments (or balance workload)
var int: total_shifts = sum(e in EMPLOYEES, d in DAYS)(works[e, d]);

solve minimize total_shifts;

output ["Schedule: ", show(works), "\\nTotal shifts: ", show(total_shifts)];
"""
