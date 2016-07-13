import cvxopt
from cvxopt import matrix
from sympy import Basic, BlockDiagMatrix
import lmi_sdp
import time
import numpy as np

epsilon_sdptol = 1e-7

##copied some methods from lmi_sdp here for compat changes
def lmi_to_coeffs(lmi, variables, split_blocks=False):
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
    if isinstance(lmi, Basic):
        lmis = [lmi]
    else:
        lmis = list(lmi)

    slms = []  # SLM stands for 'Symmetric Linear Matrix'
    for lmi in lmis:
        if lmi.is_Matrix:
            lmi = LMI(lmi)
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
        raise NotAvailableError(to_cvxopt.__name__)

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

def cvxopt_conelp(objf, lmis, variables, primalstart=None):
    # using cvxopt conelp (start point must be feasible (?), no structure exploitation,
    import cvxopt.solvers
    c, Gs, hs = to_cvxopt(objf, lmis, variables)
    tic = time.time()
    sdpout = cvxopt.solvers.sdp(c, Gs=Gs, hs=hs, primalstart=primalstart)
    toc = time.time()
    print(sdpout['status'], '(\'optimal\' does not necessarily mean feasible)')
    print('Elapsed time: %.2f s'%(toc-tic))
    return np.matrix(sdpout['x'])


def cvxopt_dsdp5(objf, lmis, variables, primalstart=None):
    # using cvxopt interface to dsdp (might be faster, can use non-feasible starting point (but not from cvxopt?), but can't handle equalities)
    import cvxopt.solvers
    c, Gs, hs = to_cvxopt(objf, lmis, variables)
    cvxopt.solvers.options['DSDP_GapTolerance'] = epsilon_sdptol
    tic = time.time()
    sdpout = cvxopt.solvers.sdp(c, Gs=Gs, hs=hs, solver='dsdp')
    toc = time.time()
    print(sdpout['status'], '(\'optimal\' does not necessarily mean feasible)')
    print('Elapsed time: %.2f s'%(toc-tic))
    return np.matrix(sdpout['x'])

#set a default solver
solve_sdp = cvxopt_conelp
