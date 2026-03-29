"""SDP solver dispatch via cvxpy.

Thin wrapper around cvxpy's Problem.solve() that maps solver names from
config to cvxpy solver constants. Supports CLARABEL (bundled, recommended),
SCS (bundled, ADMM), MOSEK (commercial), and others.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
from colorama import Fore

# Map config string -> cvxpy solver constant
SOLVER_MAP: dict[str, str] = {
    "clarabel": cp.CLARABEL,
    "scs": cp.SCS,
    "mosek": cp.MOSEK,
    "cvxopt": cp.CVXOPT,
    "copt": cp.COPT,
}


def sdp_scalar(expr: cp.Expression | float | int) -> cp.Expression | np.ndarray:
    """Reshape a scalar cvxpy expression to (1,1) for use in cp.bmat."""
    if isinstance(expr, cp.Expression):
        return cp.reshape(expr, (1, 1), order="C")
    return np.array([[float(expr)]])


def solve_sdp(
    prob: cp.Problem,
    solver_name: str = "clarabel",
    solver_opts: dict[str, Any] | None = None,
    verbose: bool = False,
) -> str:
    """Solve a cvxpy Problem with the specified solver.

    Args:
        prob: cvxpy Problem (already constructed with objective and constraints)
        solver_name: one of 'clarabel', 'scs', 'mosek', 'cvxopt', 'copt'
        solver_opts: solver-specific options (e.g. max_iter, tol)
        verbose: print solver output

    Returns:
        status string from cvxpy (e.g. 'optimal', 'infeasible', 'unbounded')
    """
    solver = SOLVER_MAP.get(solver_name.lower(), cp.CLARABEL)
    opts = solver_opts or {}
    try:
        prob.solve(solver=solver, verbose=verbose, **opts)
    except cp.error.SolverError as e:
        print(Fore.RED + f"cvxpy solver error ({solver_name}): {e}" + Fore.RESET)
        return "solver_error"

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(Fore.RED + f"cvxpy ({solver_name}): {prob.status}" + Fore.RESET)

    return prob.status
