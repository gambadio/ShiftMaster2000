% ============================================================================
% Shift Schedule Optimization Model
% ============================================================================
% Fixed MiniZinc model for shift scheduling with fairness and rotation constraints
% The LLM provides data via JSON, this model provides the optimization logic
%
% Decision: x[emp, day, shift] = 1 if employee emp works shift on day, 0 otherwise
% Objective: Minimize weighted penalties (fairness, late shifts, coverage gaps, etc.)
% ============================================================================

% === PARAMETERS (Set from Python via JSON) ===

% Dimensions
int: E;  % Number of employees
int: D;  % Number of days in planning horizon
int: S;  % Number of shift types

% Employee attributes
array[1..E] of float: fte;  % FTE per employee (0.0-1.0)
array[1..E] of int: target_shifts;  % Target number of shifts per employee for fairness

% Shift attributes
array[1..S] of bool: is_late;  % True if shift is a "late" shift (starts after 09:00)
array[1..S] of bool: is_pikett;  % True if shift is on-call/pikett duty
array[1..S] of bool: needs_french;  % True if shift requires French language skills

% Requirements and availability
array[1..D, 1..S] of int: required;  % Required headcount per day/shift
array[1..E, 1..D, 1..S] of 0..1: available;  % Availability mask: 1=available, 0=not available

% Employee language skills
array[1..E] of bool: speaks_french;  % True if employee speaks French

% Constraint toggles and parameters
bool: enforce_no_consecutive_late;  % Toggle for consecutive late shift prevention
int: pikett_min_gap_days;  % Minimum days between pikett assignments (0 = no constraint)
int: min_french_per_week;  % Minimum French-speaking dispatchers per week (0 = no constraint)

% Penalty weights (higher = more important to satisfy)
float: weight_coverage;  % Under-coverage penalty
float: weight_fairness;  % Fairness deviation penalty
float: weight_late;  % Consecutive late shifts penalty
float: weight_pikett;  % Pikett gap violation penalty

% === DECISION VARIABLES ===

% Main assignment variable: x[e,d,s] = 1 if employee e works shift s on day d
array[1..E, 1..D, 1..S] of var 0..1: x;

% === HARD CONSTRAINTS ===

% 1. AVAILABILITY: Can only assign employees when they are available
constraint forall(e in 1..E, d in 1..D, s in 1..S)(
    x[e,d,s] <= available[e,d,s]
);

% 2. ONE SHIFT PER DAY: Employee works at most one shift per day
constraint forall(e in 1..E, d in 1..D)(
    sum(s in 1..S)(x[e,d,s]) <= 1
);

% 3. COVERAGE: Must meet or exceed required headcount (soft version: allow under-staffing with penalty)
% Note: Making this soft allows finding solutions even when requirements are impossible
% The penalty in the objective will drive towards meeting requirements

% === SOFT CONSTRAINTS & PENALTY VARIABLES ===

% Soft constraint 1: Coverage deficit
array[1..D, 1..S] of var 0..E: coverage_deficit;
constraint forall(d in 1..D, s in 1..S)(
    coverage_deficit[d,s] = max(0, required[d,s] - sum(e in 1..E)(x[e,d,s]))
);

% Soft constraint 2: Fairness (deviation from target)
array[1..E] of var int: shift_count;
constraint forall(e in 1..E)(
    shift_count[e] = sum(d in 1..D, s in 1..S)(x[e,d,s])
);

array[1..E] of var 0..D*S: fairness_deviation;
constraint forall(e in 1..E)(
    fairness_deviation[e] = abs(shift_count[e] - target_shifts[e])
);

% Soft constraint 3: Consecutive late shifts
array[1..E, 1..D-1] of var 0..1: consecutive_late_violation;
constraint forall(e in 1..E, d in 1..D-1)(
    if enforce_no_consecutive_late then
        % Violation occurs if employee works late shifts on consecutive days
        consecutive_late_violation[e,d] =
            bool2int(
                (sum(s in 1..S where is_late[s])(x[e,d,s]) >= 1) /\
                (sum(s in 1..S where is_late[s])(x[e,d+1,s]) >= 1)
            )
    else
        consecutive_late_violation[e,d] = 0
    endif
);

% Soft constraint 4: Pikett gap (minimum days between pikett assignments)
% Count pikett gap violations per employee
array[1..E] of var 0..D: pikett_gap_violations;
constraint forall(e in 1..E)(
    if pikett_min_gap_days > 0 then
        pikett_gap_violations[e] = sum(d in 1..D-pikett_min_gap_days)(
            % Check if employee has pikett within the gap window after day d
            bool2int(
                (sum(s in 1..S where is_pikett[s])(x[e,d,s]) >= 1) /\
                (sum(d2 in d+1..min(d+pikett_min_gap_days, D), s in 1..S where is_pikett[s])(
                    x[e,d2,s]
                ) >= 1)
            )
        )
    else
        pikett_gap_violations[e] = 0
    endif
);

% === ADDITIONAL DOMAIN-SPECIFIC CONSTRAINTS ===

% French language requirement for specific shifts (per week)
% This is a hard constraint: must have at least N French speakers per week on French-required shifts
constraint
    if min_french_per_week > 0 then
        % For each week (assuming 7-day weeks)
        forall(week in 0..(D-1) div 7)(
            let {
                int: week_start = week * 7 + 1,
                int: week_end = min((week + 1) * 7, D)
            } in
            % Count French-speaking employees on French-required shifts this week
            sum(e in 1..E where speaks_french[e],
                d in week_start..week_end,
                s in 1..S where needs_french[s])(
                x[e,d,s]
            ) >= min_french_per_week
        )
    else
        true
    endif
;

% === OBJECTIVE FUNCTION ===

% Minimize total weighted penalties
var float: total_penalty =
    % Coverage deficit penalty
    weight_coverage * sum(d in 1..D, s in 1..S)(coverage_deficit[d,s]) +

    % Fairness penalty (sum of deviations from target)
    weight_fairness * sum(e in 1..E)(fairness_deviation[e]) +

    % Late shift consecutive penalty
    weight_late * sum(e in 1..E, d in 1..D-1)(consecutive_late_violation[e,d]) +

    % Pikett gap penalty
    weight_pikett * sum(e in 1..E)(pikett_gap_violations[e]);

solve minimize total_penalty;

% === OUTPUT (Parsed by Python) ===

output [
    "{\n",
    "  \"objective\": ", show(total_penalty), ",\n",
    "  \"coverage_deficit_total\": ", show(sum(d in 1..D, s in 1..S)(coverage_deficit[d,s])), ",\n",
    "  \"fairness_deviation_total\": ", show(sum(e in 1..E)(fairness_deviation[e])), ",\n",
    "  \"late_violations\": ", show(sum(e in 1..E, d in 1..D-1)(consecutive_late_violation[e,d])), ",\n",
    "  \"pikett_violations\": ", show(sum(e in 1..E)(pikett_gap_violations[e])), ",\n",
    "  \"assignments\": [\n"
] ++
[
    if d == 1 /\ s == 1 then "" else "," endif ++
    if fix(x[e,d,s]) == 1 then
        "    {\"emp\": " ++ show(e) ++ ", \"day\": " ++ show(d) ++ ", \"shift\": " ++ show(s) ++ "}\n"
    else
        ""
    endif
    | e in 1..E, d in 1..D, s in 1..S
] ++
[
    "  ]\n",
    "}\n"
];
