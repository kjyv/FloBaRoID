from __future__ import print_function
from builtins import str
from builtins import range

from typing import Tuple, List

import sympy
from sympy import Basic, BlockDiagMatrix, Symbol, sympify
from distutils.version import LooseVersion
old_sympy = LooseVersion(sympy.__version__) < LooseVersion('0.7.4')

import numpy as np

import cvxopt
from cvxopt import matrix
import lmi_sdp

epsilon_sdptol = 1e-6

from colorama import Fore, Back

#simplified LMI definitions (works with newer sympy, lmi_sdp variants do not)
def LMI_PD(lhs, rhs=0):
    # type: (sympy.Eq, sympy.Eq) -> (sympy.Eq)
    if old_sympy:
        lmi = lmi_sdp.LMI_PD(lhs, rhs)
    else:
        lmi = lhs > sympify(rhs)

    return lmi

def LMI_PSD(lhs, rhs=0):
    # type: (sympy.Eq, sympy.Eq) -> (sympy.Eq)
    if old_sympy:
        lmi = lmi_sdp.LMI_PSD(lhs, rhs)
    else:
        lmi = lhs >= sympify(rhs)
    return lmi

##copied some methods from lmi_sdp here for compatibility changes
def lmi_to_coeffs(lmi, variables, split_blocks=False):
    # type: (List[sympy.Matrix], List[Symbol], bool) -> List[sympy.Matrix]
    """Transforms LMIs from symbolic to numerical.

    Parameters
    ----------
    lmi: symbolic LMI or Matrix, or a list of them
    variables: list of symbols
    split_blocks: bool or string
        If set to True, function tries to subdivide each LMI into
        smaller diagonal blocks. If set to 'BlockDiagMatrix',
        BlockDiagMatrix's are split into their diagonal blocks but the
        funtion does not try to subdivide them any further.

    Returns
    -------
    coeffs: list of numerical LMIs
        List of numerical LMIs where each one is a pair where the first
        element is a list of numpy arrays corresponding to the coefficients of
        each variable, and the second element is a numpy array with zero order
        coefficients (constants not  multipling by any variable). The
        numerical coefficients are extracted from the matrix `M` of the
        canonical PSD (or PD) LMI form `M>=0` (or `M>0`).

    Example
    -------
    >>> from sympy import Matrix
    >>> from sympy.abc import x, y, z
    >>> from lmi_sdp import LMI_PSD, lmi_to_coeffs
    >>> vars = [x, y, z]
    >>> m = Matrix([[x+3, y-2], [y-2, z]])
    >>> lmi = LMI_PSD(m)
    >>> lmi_to_coeffs(lmi, vars)
    [([array([[ 1.,  0.],
           [ 0.,  0.]]), array([[ 0.,  1.],
           [ 1.,  0.]]), array([[ 0.,  0.],
           [ 0.,  1.]])], array([[ 3., -2.],
           [-2.,  0.]]))]
    """

    if old_sympy:
        return lmi_sdp.lmi_to_coeffs(lmi, variables, split_blocks)

    if isinstance(lmi, Basic):
        lmis = [lmi]
    else:
        lmis = list(lmi)

    slms = []  # SLM stands for 'Symmetric Linear Matrix'
    for lmi in lmis:
        if lmi.is_Matrix:
            lmi = LMI_PSD(lmi)
        lm = lmi.canonical.gts
        slms.append(lm)

    if split_blocks:
        orig_slms = slms
        slms = []
        for slm in orig_slms:
            if isinstance(slm, BlockDiagMatrix):
                if split_blocks == 'BlockDiagMatrix':
                    slms += slm.diag
                else:
                    slms += sum([d.get_diag_blocks() for d in slm.diag], [])
            else:
                slms += slm.get_diag_blocks()

    coeffs = [lmi_sdp.lm_sym_to_coeffs(slm, variables) for slm in slms]

    return coeffs

def to_cvxopt(objective_func, lmis, variables, objective_type='minimize',
              split_blocks=True):
    """Prepare objective and LMI to be used with cvxopt SDP solver.

    Parameters
    ----------
    objective_func: symbolic linear expression
    lmi: symbolic LMI or Matrix, or a list of them
    variables: list of symbols
        The variable symbols which form the LMI/SDP space.
    objective_type: 'maximize' or 'minimize', defaults to 'minimize'
    split_blocks: bool
        If set to True, function tries to subdivide each LMI into
        smaller diagonal blocks

    Returns
    -------
    c, Gs, hs: parameters ready to be input to cvxopt.solvers.sdp()
    """
    if cvxopt is None:
        raise lmi_sdp.sdp.NotAvailableError(to_cvxopt.__name__)

    obj_coeffs = lmi_sdp.objective_to_coeffs(objective_func, variables,
                                             objective_type)
    lmi_coeffs = lmi_to_coeffs(lmis, variables, split_blocks)

    c = matrix(obj_coeffs)

    Gs = []
    hs = []

    for (LMis, LM0) in lmi_coeffs:
        Gs.append(matrix([(-LMi).flatten().astype(float).tolist()
                          for LMi in LMis]))
        hs.append(matrix(LM0.astype(float).tolist()))

    return c, Gs, hs


def to_sdpa_sparse(objective_func, lmis, variables, objective_type='minimize',
                   split_blocks=True, comment=None):
    """Put problem (objective and LMIs) into SDPA sparse format."""
    obj_coeffs = lmi_sdp.objective_to_coeffs(objective_func, variables,
                                             objective_type)
    lmi_coeffs = lmi_to_coeffs(lmis, variables, split_blocks)

    s = lmi_sdp.sdp._sdpa_header(obj_coeffs, lmi_coeffs, comment)

    def _print_sparse(x, b, m, sign=1):
        s = ''
        shape = m.shape
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                e = sign*m[i, j]
                if e != 0:
                    s += '%d %d %d %d %s\n' % (x, b, i+1, j+1, str(e))
        return s

    for b in range(len(lmi_coeffs)):
        s += _print_sparse(0, b+1, lmi_coeffs[b][1], sign=-1)
    for x in range(len(obj_coeffs)):
        for b in range(len(lmi_coeffs)):
            s += _print_sparse(x+1, b+1, lmi_coeffs[b][0][x])

    return s

def cvxopt_conelp(objf, lmis, variables, primalstart=None):
    # type: (List[Symbol], List[sympy.Eq], List[Symbol], np._ArrayLike) -> Tuple[np.matrix, str]
    ''' using cvxopt conelp to solve SDP program

        a more exact but possibly less robust solver than dsdp5

        TODO: is currently not using primalstart argument (benefits?)
        primalstart['sl'] - initial value of u?
        primalsstart['x'] - initial values of x
        primalsstart['ss'] - value like hs for initial x values (lmis replaced with primal values
        and converted to cvxopt matrix format), must be within constraints

        Notes:
         - Errors of the form "Rank(A) < p or Rank([G; A]) < n" mean that there are linear
           dependencies in the problem matrix A, i.e. too many base parameters/columns are
           determined (depends on proper base regressor, dependent e.g. on minTol value. If data has
           too little dependencies, use structural regressor) or in the constraints G (one might
           need to add constraints).
    '''

    import cvxopt.solvers
    c, Gs, hs = to_cvxopt(objf, lmis, variables)
    cvxopt.solvers.options['maxiters'] = 100
    cvxopt.solvers.options['show_progress'] = False
    #cvxopt.solvers.options['feastol'] = 1e-5

    sdpout = cvxopt.solvers.sdp(c, Gs=Gs, hs=hs)
    state = sdpout['status']
    if sdpout['status'] == 'optimal':
        #print("(does not necessarily mean feasible)")
        pass
    elif primalstart is not None:
        # return primalstart if no solution was found
        print(Fore.RED + '{}'.format(sdpout['status']) + Fore.RESET)
        sdpout['x'] = np.reshape( np.concatenate(([0], primalstart)), (len(primalstart)+1, 1) )
    return np.matrix(sdpout['x']), state


def cvxopt_dsdp5(objf, lmis, variables, primalstart=None, wide_bounds=False):
    # type: (List[Symbol], List[sympy.Eq], List[Symbol], np._ArrayLike, bool) -> Tuple[np.matrix, str]
    # using cvxopt interface to dsdp5
    # (not using primal atm)
    import cvxopt.solvers
    c, Gs, hs = to_cvxopt(objf, lmis, variables)
    cvxopt.solvers.options['dsdp']= {'DSDP_GapTolerance': epsilon_sdptol, 'DSDP_Monitor': 10}
    if wide_bounds:
        sdpout = cvxopt.solvers.sdp(c, Gs=Gs, hs=hs, beta=10e15, gama=10e15, solver='dsdp')
    else:
        sdpout = cvxopt.solvers.sdp(c, Gs=Gs, hs=hs, solver='dsdp')
    state = sdpout['status']
    if sdpout['status'] == 'optimal':
        print("{}".format(sdpout['status']))
        #print("(does not necessarily mean feasible)")
    else:
        print(Fore.RED + '{}'.format(sdpout['status']) + Fore.RESET)
    return np.matrix(sdpout['x']), state


def dsdp5(objf, lmis, variables, primalstart=None, wide_bounds=False):
    # type: (List[Symbol], List[sympy.Eq], List[Symbol], np._ArrayLike, bool) -> Tuple[np.matrix, str]
    ''' use dsdp5 directly (faster than cvxopt, can use starting points, more robust) '''
    import subprocess
    import os

    sdpadat = to_sdpa_sparse(objf, lmis, variables)
    import uuid
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sdpa_dat_{}'.format(uuid.uuid4()))
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, 'sdp.dat-s'), 'w') as f:
        f.write(sdpadat)

    if primalstart is None:
        primalstart = np.zeros(len(variables)-1)
    with open(os.path.join(dir, 'primal.dat'), 'wb') as f:
            np.savetxt(f, primalstart)

    # change options to allow for far away solutions
    if wide_bounds:
        bounds = ['-boundy', '1e15', '-penalty', '1e15']
    else:
        bounds = []

    try:
        result = subprocess.check_output(['dsdp5', 'sdp.dat-s', '-save', 'dsdp5.out', '-gaptol',
                                         '{}'.format(epsilon_sdptol)] + bounds +
                                         ['-y0', 'primal.dat'],
                                         cwd = dir).decode('utf-8')
        state = 'optimal'
    except subprocess.CalledProcessError as e:
        print("DSDP stopped early: {}".format(e.returncode))
        state = 'stopped'
        result = e.output

    error = list()
    for s in result.split('\n'):
        if 'DSDP Terminated Due to' in s:
            error.append(s)
            state = 'infeasible'
        if 'DSDP Primal Unbounded, Dual Infeasible' in s:
            error.append(s)
            state = 'infeasible'
        if 'DSDP Converged' in s:
            # there can be converged and dual infeasible messages but errors come last
            state = 'optimal'

    #if state != 'optimal':
        #if there were errors, print all the output
    #    print(result)

    if error:
        state = 'infeasible'
        print(Fore.RED + error[0] + Fore.RESET)
    else:
        print(state)
    outfile = open(os.path.join(dir, 'dsdp5.out'), 'r').readlines()
    sol = [float(v) for v in outfile[0].split()]

    # remove tmp dir again
    import shutil
    shutil.rmtree(dir)

    return np.matrix(sol).T, state


#set a default solver
solve_sdp = cvxopt_conelp # choose one from dsdp5, cvxopt_dsdp5, cvxopt_conelp

# it seems cvxopt_conelp and cvxopt_dsdp5 are working well when the data is good whereas dsdp5
# sometimes fails that situation completely. However, in some bad data situations dsdp5 performs very well
# (with changed bounds) where the other two don't work.
