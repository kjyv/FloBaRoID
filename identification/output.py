#-*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
from typing import Tuple

import sys
import os
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import colorama
from colorama import Fore, Style

from identification import helpers

np.core.arrayprint._line_width = 160

# redefine unicode for testing in python2/3
if sys.version_info >= (3, 0):
    unicode = str

# color triplets
color_triplets_6 = [
           [ 0.29019608,  0.43529412,  0.89019608],
           [ 0.52156863,  0.58431373,  0.88235294],
           [ 0.70980392,  0.73333333,  0.89019608],
           [ 0.90196078,  0.68627451,  0.7254902 ],
           [ 0.87843137,  0.48235294,  0.56862745],
           [ 0.82745098,  0.24705882,  0.41568627],
          ]

#set some more colors for higher DOF
from palettable.tableau import Tableau_10, Tableau_20
colors = Tableau_10.mpl_colors[0:6] + Tableau_20.mpl_colors + Tableau_20.mpl_colors

#swap some values for aesthetics
colors[2], colors[0] = colors[0], colors[2]

class OutputConsole(object):
    def __init__(self, idf):
        self.idf = idf

        if not idf.opt['useEssentialParams']:
            idf.stdEssentialIdx = list(range(0, idf.model.num_identified_params))
            idf.stdNonEssentialIdx = []

        #if requested, load params from other urdf for comparison
        if idf.urdf_file_real:
            self.xStdReal = idf.xStdReal
            self.xBaseReal = idf.xBaseReal

        p_idf = idf.model.identified_params
        if idf.opt['showStandardParams']:
            # convert params to COM-relative instead of frame origin-relative (linearized parameters)
            if idf.opt['outputBarycentric']:
                if idf.opt['identifyGravityParamsOnly']:
                    xStd_full = idf.model.xStdModel.copy()
                    xStd_full[p_idf] = idf.model.xStd
                    xStd = idf.paramHelpers.paramsLink2Bary(xStd_full)
                    self.xStd = xStd[p_idf]
                else:
                    self.xStd = idf.paramHelpers.paramsLink2Bary(idf.model.xStd)
                self.xStdModel = idf.paramHelpers.paramsLink2Bary(idf.model.xStdModel)
                if idf.urdf_file_real:
                    self.xStdReal = idf.paramHelpers.paramsLink2Bary(self.xStdReal)
            else:
                self.xStd = idf.model.xStd
                self.xStdModel = idf.model.xStdModel

    def printStdParams(self, summary_only=False):
        idf = self.idf

        if not idf.opt['showStandardParams']:
            return

        if not summary_only:
            if idf.opt['outputBarycentric']:
                print("Barycentric (relative to COM) Standard Parameters")
            else:
                print("Linear (relative to Frame) Standard Parameters")

        # collect values for parameters
        description = idf.model.generator.getDescriptionOfParameters()
        if idf.opt['identifyFriction']:
            for i in range(0, idf.model.num_dofs):
                description += "Parameter {}: Constant friction / offset of joint {}\n".format(
                        i+idf.model.num_model_params,
                        idf.model.jointNames[i]
                )

            for i in range(0, idf.model.num_dofs*2):
                description += "Parameter {}: Velocity dep. friction joint {}\n".format(
                        i+idf.model.num_dofs+idf.model.num_model_params,
                        idf.model.jointNames[i%idf.model.num_dofs]
                )

        idx_ep = 0  #count essential params
        lines = list()
        sum_diff_r_pc_ess = 0
        sum_diff_r_pc_all = 0
        sum_pc_delta_all = 0
        sum_pc_delta_ess = 0
        descriptions = description.replace(r'Parameter ', '#').split('\n')
        for idx_p in range(idf.model.num_identified_params):
            idx_p_full = idf.model.identified_params[idx_p]
            d = descriptions[idx_p_full]

            if idf.opt['outputBarycentric']:
                d = d.replace(r'first moment', 'center')
            # add symbol for each parameter
            d = d.replace(r':', ': {} -'.format(idf.model.param_syms[idx_p_full]))

            # print beginning of each link block in green
            if idx_p_full % 10 == 0 and idx_p_full < idf.model.num_model_params:
                d = Fore.GREEN + d

            # get some error values for each parameter
            approx = self.xStd[idx_p]
            apriori = self.xStdModel[idx_p_full]
            diff = approx - apriori

            if idf.urdf_file_real:
                real = self.xStdReal[idx_p_full]
                # set real params that are 0 to some small value
                #if real == 0: real = 0.01

                # get error percentage (new to real)
                # so if 100% are the real value, how big is the error
                diff_real = approx - real
                if real != 0:
                    diff_r_pc = (100 * diff_real) / real
                else:
                    diff_r_pc = (100 * diff_real) / 0.01

                # add to final error percent sum
                sum_diff_r_pc_all += np.abs(diff_r_pc)
                if idx_p in idf.stdEssentialIdx:
                    sum_diff_r_pc_ess += np.abs(diff_r_pc)

                # get error percentage (new to apriori)
                diff_apriori = apriori - real
                #if apriori == 0: apriori = 0.01
                if diff_apriori != 0:
                    pc_delta = np.abs((100/diff_apriori)*diff_real)
                elif np.abs(diff_real) > 0:
                    # if there was no error between apriori and real
                    #pc_delta = np.abs((100/0.01)*diff_real)
                    pc_delta = 100 + np.abs(diff_r_pc)
                else:
                    # both real and a priori are zero, error is still at 100% (of zero error)
                    pc_delta = 100
                sum_pc_delta_all += pc_delta
                if idx_p in idf.stdEssentialIdx:
                    sum_pc_delta_ess += pc_delta
            else:
                # get percentage difference between apriori and identified values
                # (shown when real values are not known)
                if apriori != 0:
                    diff_pc = (100 * diff) / apriori
                else:
                    diff_pc = (100 * diff) / 0.01

            #values for each line
            if idf.opt['useEssentialParams'] and idx_ep < idf.num_essential_params and idx_p in idf.stdEssentialIdx:
                sigma = idf.p_sigma_x[idx_ep]
            else:
                sigma = 0.0

            if idf.urdf_file_real and idf.opt['constrainToConsistent']:
                if idx_p_full in idf.model.non_id:
                    idf.sdp.constr_per_param[idx_p_full].append('nID')
                vals = [real, apriori, approx, diff, np.abs(diff_r_pc), pc_delta, sigma, ' '.join(idf.sdp.constr_per_param[idx_p_full]), d]
            elif idf.urdf_file_real:
                vals = [real, apriori, approx, diff, np.abs(diff_r_pc), pc_delta, sigma, d]
            elif idf.opt['constrainToConsistent']:
                if idx_p_full in idf.model.non_id:
                    idf.sdp.constr_per_param[idx_p_full].append('nID')
                vals = [apriori, approx, diff, diff_pc, ' '.join(idf.sdp.constr_per_param[idx_p_full]), d]
            elif idf.opt['useEssentialParams']:
                vals = [apriori, approx, diff, diff_pc, sigma, d]
            else:
                vals = [apriori, approx, diff, diff_pc, d]
            lines.append(vals)

            if idf.opt['useEssentialParams'] and idx_p in idf.stdEssentialIdx:
                idx_ep += 1

        if idf.urdf_file_real and idf.opt['constrainToConsistent']:
            column_widths = [13, 13, 13, 7, 7, 7, 6, 8, 45]
            precisions = [8, 8, 8, 4, 1, 1, 3, 0, 0]
        elif idf.urdf_file_real:
            column_widths = [13, 13, 13, 7, 7, 7, 6, 45]
            precisions = [8, 8, 8, 4, 1, 1, 3, 0]
        elif idf.opt['constrainToConsistent']:
            column_widths = [13, 13, 7, 7, 8, 45]
            precisions = [8, 8, 4, 1, 0, 0]
        elif idf.opt['useEssentialParams']:
            column_widths = [13, 13, 7, 7, 6, 45]
            precisions = [8, 8, 4, 1, 3, 0]
        else:
            column_widths = [13, 13, 7, 7, 45]
            precisions = [8, 8, 4, 1, 0]

        if not summary_only:
            # print column header
            template = ''
            for w in range(0, len(column_widths)):
                template += '|{{{}:{}}}'.format(w, column_widths[w])
            if idf.urdf_file_real and idf.opt['constrainToConsistent']:
                print(template.format("'Real'", "A priori", "Ident", "Change", "%e", "Δ%e", "%σ", "Constr", "Description"))
            elif idf.urdf_file_real:
                print(template.format("'Real'", "A priori", "Ident", "Change", "%e", "Δ%e", "%σ", "Description"))
            elif idf.opt['constrainToConsistent']:
                print(template.format("A priori", "Ident", "Change", "%e", "Constr", "Description"))
            elif idf.opt['useEssentialParams']:
                print(template.format("A priori", "Ident", "Change", "%e", "%σ", "Description"))
            else:
                print(template.format("A priori", "Ident", "Change", "%e", "Description"))

            # print values/description
            template = ''
            for w in range(0, len(column_widths)):
                if(type(lines[0][w]) in [str, unicode, list]):
                    # strings don't have precision
                    template += '|{{{}:{}}}'.format(w, column_widths[w])
                else:
                    template += '|{{{}:{}.{}f}}'.format(w, column_widths[w], precisions[w])
            idx_p = 0
            for l in lines:
                t = template.format(*l)
                if idx_p in idf.stdNonEssentialIdx:
                    t = Style.DIM + t
                if idx_p in idf.stdEssentialIdx:
                    t = Style.BRIGHT + t
                print(t, end=' ')
                idx_p += 1
                print(Style.RESET_ALL)
            print("\n")

    def printBaseParams(self, summary_only=False):
        idf = self.idf

        if not idf.opt['showBaseParams'] or summary_only or idf.opt['estimateWith'] in ['urdf', 'std_direct']:
            return

        print("Base Parameters and Corresponding standard columns")
        if not idf.opt['useEssentialParams']:
            baseEssentialIdx = list(range(0, idf.model.num_base_params))
            baseNonEssentialIdx = []
            xBase_essential = idf.model.xBase
        else:
            baseEssentialIdx = idf.baseEssentialIdx
            baseNonEssentialIdx = idf.baseNonEssentialIdx
            xBase_essential = idf.xBase_essential

        # collect values for parameters
        idx_ep = 0
        lines = list()
        sum_error_all_base = 0
        for idx_p in range(0, idf.model.num_base_params):
            if idf.opt['useEssentialParams']: # and xBase_essential[idx_p] != 0:
                new = xBase_essential[idx_p]
            else:
                new = idf.model.xBase[idx_p]
            old = idf.model.xBaseModel[idx_p]
            diff = new - old
            if idf.urdf_file_real:
                real = self.xBaseReal[idx_p]
                error = new - real
                sum_error_all_base += np.abs(error)

            # collect linear dependencies for this param
            #deps = np.where(np.abs(idf.linear_deps[idx_p, :])>0.1)[0]
            #dep_factors = idf.linear_deps[idx_p, deps]

            if idf.opt['showBaseEqns']:
                param_columns = " = "
                param_columns += "{}".format(idf.model.base_deps[idx_p])
                #for p in range(0, len(deps)):
                #    param_columns += ' {:.4f}*|{}|'.format(dep_factors[p], idf.P[idf.num_base_params:][deps[p]])
            else:
                param_columns = ""

            if idf.opt['useEssentialParams']:
                if idx_p in idf.baseEssentialIdx:
                    sigma = idf.p_sigma_x[idx_ep]
                else:
                    sigma = 0
            else:
                sigma = idf.p_sigma_x[idx_p]

            if idf.urdf_file_real:
                lines.append([idx_p, real, old, new, diff, error, sigma, param_columns])
            else:
                lines.append([idx_p, old, new, diff, sigma, param_columns])

            if idf.opt['useEssentialParams'] and idx_p in idf.baseEssentialIdx:
                idx_ep+=1

        if idf.urdf_file_real:
            column_widths = [3, 13, 13, 13, 7, 7, 6, 30]   # widths of the columns
            precisions = [0, 8, 8, 8, 4, 4, 3, 0]         # numerical precision
        else:
            column_widths = [3, 13, 13, 7, 6, 30]   # widths of the columns
            precisions = [0, 8, 8, 4, 3, 0]         # numerical precision

        if not summary_only:
            # print column header
            template = ''
            for w in range(0, len(column_widths)):
                template += '|{{{}:{}}}'.format(w, column_widths[w])
            if idf.urdf_file_real:
                print(template.format("#", "Real", "A priori", "Ident", "Change", "Error", "%σ", "Description"))
            else:
                print(template.format("#", "A priori", "Ident", "Change", "%σ", "Description"))

            # print values/description
            template = ''
            for w in range(0, len(column_widths)):
                if(type(lines[0][w]) in [str, unicode]):
                    # strings don't have precision
                    template += '|{{{}:{}}}'.format(w, column_widths[w])
                else:
                    template += '|{{{}:{}.{}f}}'.format(w, column_widths[w], precisions[w])
            idx_p = 0
            for l in lines:
                t = template.format(*l)
                if idx_p in baseNonEssentialIdx:
                    t = Style.DIM + t
                elif idx_p in baseEssentialIdx:
                    t = Style.BRIGHT + t
                if idf.opt['showEssentialSteps'] and l[-2] == np.max(idf.p_sigma_x):
                    t = Fore.CYAN + t
                print(t, end=' ')
                idx_p+=1
                print(Style.RESET_ALL)

    def printLatex(self):
        ''' print standard params also as latex table '''
        idf = self.idf
        if not idf.opt['outputLatex']:
            return

        print('As Latex:')
        import inspect
        print(inspect.cleandoc(r"""
            \begin{table}[h]
                \caption{Identified standard parameters. Non-identifiable parameters are marked
                         with *. These have no effect on dynamics and are determined only to satisfy
                         consistency constraints.}
                \begin{center}
        """))
        header = inspect.cleandoc(r"""
                \begin{minipage}[t]{0.32\linewidth}
                    \resizebox{0.97\textwidth}{!}{%
                    \begin{tabular}[t]{c c c}
                        \hline
                        \rule{0pt}{12pt} Parameter & Prior & Identified \\[2pt]
                        \hline\rule{0pt}{12pt}
        """)
        footer = inspect.cleandoc(r"""
                        \hline
                    \end{tabular}}
                \end{minipage}
        """)
        print(header)

        #print table rows
        for idx_p in range(10, idf.model.num_identified_params):
            #if idx_p == len(idf.model.identifiable) // 2:
            if idx_p-10 in [(idf.model.num_identified_params-10) // 3, ((idf.model.num_identified_params-10) // 3)*2]:
                #start new table after half of params
                print(footer)
                print(header)
            #if idx_p in idf.model.identifiable:
            #add another underscore for proper subscripts
            import re
            param = str(idf.model.param_syms[idx_p])
            p = re.compile(r"([0-9]+)(.*)")
            param = p.sub(r'{\1\2}', param)
            nonid = '*' if idx_p in idf.model.non_id else ''
            real = self.xStdReal if idf.urdf_file_real else self.xStdModel
            print("        ${}$    & ${:.4f}$ & ${:.4f}${} \\\\".format(param, real[idx_p], self.xStd[idx_p], nonid))

        print(footer)
        print(inspect.cleandoc(r"""
                \end{center}
            \end{table}
        """))
        print("")

    def printStats(self, summary_only=False):
        idf = self.idf
        if idf.opt['selectBlocksFromMeasurements']:
            if len(idf.data.usedBlocks):
                print("used {} of {} blocks: {}".format(len(idf.data.usedBlocks),
                                                        len(idf.data.usedBlocks)+len(idf.data.unusedBlocks),
                                                        [b for (b,bs,cond,linkConds) in idf.data.usedBlocks]))
            else:
                print("\ncurrent block: {}".format(idf.data.block_pos))
            #print "unused blocks: {}".format(idf.unusedBlocks)
            print("condition number: {}".format(la.cond(idf.model.YBase)))

        if idf.opt['identifyGravityParamsOnly']:
            fric = idf.model.num_dofs * idf.opt['identifyFriction']
            sum_id = np.sum(idf.model.xStd[0:idf.model.num_identified_params-fric:4])
        else:
            sum_id = np.sum(idf.model.xStd[0:idf.model.num_model_params:10])

        print(Style.BRIGHT + "Parameters" + Style.RESET_ALL)
        sum_apriori = np.sum(idf.model.xStdModel[0:idf.model.num_model_params:10])
        print("Estimated overall mass: {} kg vs. a priori {} kg".format(sum_id, sum_apriori), end="")
        if idf.urdf_file_real:
            print(" vs. real {} kg".format(np.sum(self.xStdReal[0:idf.model.num_model_params:10])))
        else:
            print()

        if idf.opt['showStandardParams']:
            if idf.opt['showTriangleConsistency']:
                cons_apriori = idf.paramHelpers.checkPhysicalConsistency(idf.model.xStdModel, full=True)
                cons_ident = idf.paramHelpers.checkPhysicalConsistency(idf.model.xStd)
                print("Consistency (including triangle inequality):")
            else:
                cons_apriori = idf.paramHelpers.checkPhysicalConsistencyNoTriangle(idf.model.xStdModel, full=True)
                cons_ident = idf.paramHelpers.checkPhysicalConsistencyNoTriangle(idf.model.xStd)

            if False in list(cons_apriori.values()):
                print(Fore.RED + "A priori parameters are not physical consistent!" + Fore.RESET)
                print("Per-link physical consistency (a priori): {}".format(cons_apriori))
            else:
                print("A priori parameters are physical consistent")

            if False in list(cons_ident.values()):
                print("Identified parameters are not physical consistent,")
                print("per-link physical consistency (identified): {}".format(cons_ident))
            else:
                print("Identified parameters are physical consistent")

        if idf.opt['identifyGravityParamsOnly']:
            p_idf = idf.model.identified_params
        else:
            p_idf = idf.model.identifiable
        if idf.urdf_file_real:
            if idf.opt['showStandardParams']:
                #if idf.opt['useEssentialParams']:
                #    print("Mean relative error of essential std params: {}%".\
                #            format(sum_diff_r_pc_ess / len(idf.stdEssentialIdx)))
                #print("Mean relative error of all std params: {}%".format(sum_diff_r_pc_all/len(idf.model.xStd)))

                #if idf.opt['useEssentialParams']:
                #    print("Mean error delta (a priori error vs approx error) of essential std params: {}%".\
                #            format(sum_pc_delta_ess/len(idf.stdEssentialIdx)))
                #print("Mean error delta (a priori error vs approx error) of all std params: {}%".\
                #        format(sum_pc_delta_all/len(idf.model.xStd)))
                sq_error_apriori = np.square(la.norm(self.xStdReal[p_idf] - idf.model.xStdModel[p_idf]))
                if idf.opt['identifyGravityParamsOnly']:
                    xStd_full = idf.model.xStdModel.copy()
                    xStd_full[p_idf] = idf.model.xStd
                    sq_error_idf = np.square(la.norm(self.xStdReal[p_idf] - xStd_full[p_idf]))
                else:
                    sq_error_idf = np.square(la.norm(self.xStdReal[p_idf] - idf.model.xStd[p_idf]))
                print("Squared distance of identifiable std parameter vectors (identified, a priori) to real: {} vs. {}".\
                        format(sq_error_idf, sq_error_apriori))
                #sq_error_apriori = np.square(la.norm(xStdReal - idf.model.xStdModel))
                #sq_error_idf = np.square(la.norm(xStdReal - idf.model.xStd))
                #print( "Squared distance of std parameter vectors (identified, a priori) to real: {} vs. {}".\
                #        format(sq_error_idf, sq_error_apriori))
            if idf.opt['showBaseParams'] and not summary_only and idf.opt['estimateWith'] not in ['urdf', 'std_direct']:
                #print("Mean error (a priori - approx) of all base params: {:.5f}".\
                #        format(sum_error_all_base/len(idf.model.xBase)))
                sq_error_apriori = np.square(la.norm(self.xBaseReal - idf.model.xBaseModel))
                sq_error_idf = np.square(la.norm(self.xBaseReal - idf.model.xBase))
                print("Squared distance of base parameter vectors (identified, a priori) to real: {} vs. {}".\
                        format(sq_error_idf, sq_error_apriori))
        else:
            if idf.opt['showStandardParams'] and not summary_only:
                if idf.opt['identifyGravityParamsOnly']:
                    xStd_full = idf.model.xStdModel.copy()
                    xStd_full[p_idf] = idf.model.xStd
                    sq_error_apriori = np.square(la.norm(xStd_full[p_idf] - idf.model.xStdModel[p_idf]))
                else:
                    sq_error_apriori = np.square(la.norm(self.xStd[p_idf] - idf.model.xStdModel[p_idf]))
                print("Squared distance of identifiable std parameter vectors to a priori: {}".\
                        format(sq_error_apriori))
            if idf.opt['showBaseParams'] and not summary_only and idf.opt['estimateWith'] not in ['urdf', 'std_direct']:
                sq_error_apriori = np.square(la.norm(idf.model.xBase - idf.model.xBaseModel))
                print("Squared distance of base parameter vectors (identified vs. a priori): {}".\
                        format(sq_error_apriori))


        print(Style.BRIGHT + "\nTorque prediction errors" + Style.RESET_ALL)
        # get percentual error (i.e. how big is the error relative to the measured magnitudes)
        idf.estimateRegressorTorques(estimateWith='urdf')   #estimate torques with CAD params
        idf.estimateRegressorTorques()   #estimate torques again with identified parameters
        idf.apriori_error = sla.norm(idf.tauAPriori-idf.model.tauMeasured)*100/sla.norm(idf.model.tauMeasured)
        idf.res_error = sla.norm(idf.tauEstimated-idf.model.tauMeasured)*100/sla.norm(idf.model.tauMeasured)
        print("Relative mean residual error: {}% vs. A priori: {}%".\
                format(idf.res_error, idf.apriori_error))

        idf.abs_apriori_error = np.mean(sla.norm(idf.tauAPriori-idf.model.tauMeasured, axis=1))
        idf.abs_res_error = idf.base_error #np.mean(sla.norm(idf.tauEstimated-idf.model.tauMeasured, axis=1))
        print("Absolute mean residual error: {} vs. A priori: {}".format(idf.abs_res_error, idf.abs_apriori_error))

        torque_limits = []
        for joint in idf.model.jointNames:
            torque_limits.append(idf.model.limits[joint]['torque'])
        idf.abs_apriori_error = helpers.getNRMSE(idf.model.tauMeasured, idf.tauAPriori, limits=torque_limits)
        idf.abs_res_error = helpers.getNRMSE(idf.model.tauMeasured, idf.tauEstimated, limits=torque_limits)
        print("NRMS of residual error: {}% vs. A priori: {}%".format(idf.abs_res_error, idf.abs_apriori_error))


    def render(self, summary_only=False):
        """Output results on the console, tables of identified parameters and some statistics"""

        colorama.init(autoreset=False)
        self.printStdParams(summary_only)
        self.printBaseParams(summary_only)
        self.printLatex()
        self.printStats(summary_only)


class OutputMatplotlib(object):
    def __init__(self, datasets, text=None):
        self.datasets = datasets
        self.text = text

    def render(self, idf, filename='output.html'):
        progress_inst = helpers.Progress(idf.opt)
        self.progress = progress_inst.progress

        if idf.opt['outputFilename']:
            filename = idf.opt['outputFilename']

        if idf.opt['outputAs'] == 'html':
            # write matplotlib/d3 plots to html file
            import matplotlib
            import matplotlib.pyplot as plt, mpld3
            import matplotlib.axes

            from mpld3 import plugins
            from jinja2 import Environment, FileSystemLoader
        elif idf.opt['outputAs'] in ['pdf', 'interactive', 'tikz']:
            # show plots in separate matplotlib windows
            import matplotlib
            if idf.opt['outputAs'] == 'pdf':
                from matplotlib.backends.backend_pdf import PdfPages
                pp = PdfPages(filename)
            import matplotlib.pyplot as plt
            import matplotlib.axes
        else:
            print("No proper output method given. Not plotting.")
            return

        font_size = 10
        if idf.opt['outputAs'] in ['pdf', 'tikz']:
            if idf.opt['plotPerJoint']:
                font_size = 30
            else:
                font_size = 12
            matplotlib.rcParams.update({'font.size': font_size})
            matplotlib.rcParams.update({'axes.labelsize': font_size -5})
            matplotlib.rcParams.update({'axes.linewidth': font_size / 15.})
            matplotlib.rcParams.update({'axes.titlesize': font_size -2})
            matplotlib.rcParams.update({'legend.fontsize': font_size -2})
            matplotlib.rcParams.update({'xtick.labelsize': font_size -5})
            matplotlib.rcParams.update({'ytick.labelsize': font_size -5})
            matplotlib.rcParams.update({'lines.linewidth': font_size / 15.})
            matplotlib.rcParams.update({'patch.linewidth': font_size / 15.})
            matplotlib.rcParams.update({'grid.linewidth': font_size / 20.})


        # skip some samples so graphs don't get too large/detailed TODO: change skip so that some
        # maximum number of points is plotted (determined by screen etc.)
        skip = 5

        #create figures and plots
        figures = list()
        for ds in self.progress(range(len(self.datasets))):
            group = self.datasets[ds]
            fig, axes = plt.subplots(len(group['dataset']), sharex=True, sharey=True)
            # scale unified scaling figures to same ranges and add some margin
            if group['unified_scaling']:
                ymin = 0
                ymax = 0
                for i in range(len(group['dataset'])):
                    ymin = np.min((np.min(group['dataset'][i]['data']), ymin)) * 1.05
                    ymax = np.max((np.max(group['dataset'][i]['data']), ymax)) * 1.05

            #plot each group of data
            for d_i in range(len(group['dataset'])):
                d = group['dataset'][d_i]
                if not issubclass(type(axes), matplotlib.axes.SubplotBase):
                    ax = axes[d_i]
                else:
                    ax = axes
                    axes = [axes]
                if idf.opt['outputAs'] != 'tikz':
                    ax.set_title(d['title'])
                if group['unified_scaling']:
                    ax.set_ylim([ymin, ymax])
                for data_i in range(0, len(d['data'])):
                    if len(d['data'][data_i].shape) > 1:
                        #data matrices
                        for i in range(0, d['data'][data_i].shape[1]):
                            l = group['labels'][i] if data_i == 0 else ''
                            if i < 6 and 'contains_base' in group and group['contains_base']:
                                ls = 'dashed'
                            else:
                                ls = '-'
                            dashes = ()      # type: Tuple
                            if idf.opt['plotErrors']:
                                if idf.opt['plotPrioriTorques']:
                                    n = 3
                                else:
                                    n = 2
                                if i == n:
                                    ls = 'dashed'
                                    dashes = (3, 0.5)
                            ax.plot(d['time'][::skip], d['data'][data_i][::skip, i], label=l,
                                    color=colors[i], alpha=1-(data_i/2.0), linestyle=ls,
                                    dashes=dashes)
                    else:
                        #data vector
                        ax.plot(d['time'][::skip], d['data'][data_i][::skip],
                                label=group['labels'][d_i], color=colors[0], alpha=1-(data_i/2.0))

                ax.grid(which='both', linestyle="dotted", alpha=0.8)
                if 'y_label' in group:
                    ax.set_ylabel(group['y_label'])

            if idf.opt['outputAs'] != 'tikz':
                ax.set_xlabel("Time (s)")

            plt.setp([a.get_xticklabels() for a in axes[:-1]], visible=False)
            #plt.setp([a.get_yticklabels() for a in axes], fontsize=8)

            if idf.opt['plotLegend']:
                handles, labels = ax.get_legend_handles_labels()
                if idf.opt['outputAs'] == 'html':
                    #TODO: show legend properly (see mpld3 bug #274)
                    #leg = fig.legend(handles, labels, loc='upper right', fancybox=True, fontsize=10, title='')
                    leg = axes[0].legend(handles, labels, loc='upper right', fancybox=True, fontsize=10, title='', prop={'size': 8})
                else:
                    leg = plt.figlegend(handles, labels, loc='upper right', fancybox=True,
                            fontsize=font_size, title='', prop={'size': font_size-3})
                    leg.draggable()

            fig.subplots_adjust(hspace=2)
            fig.set_tight_layout(True)

            if idf.opt['outputAs'] == 'html':
                plugins.clear(fig)
                plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=False),
                                plugins.MousePosition(fontsize=14, fmt=".5g"))
                figures.append(mpld3.fig_to_html(fig))
            elif idf.opt['outputAs'] == 'interactive':
                plt.show(block=False)
            elif idf.opt['outputAs'] == 'pdf':
                pp.savefig(plt.gcf())
            elif idf.opt['outputAs'] == 'tikz':
                from matplotlib2tikz import save as tikz_save
                tikz_save('{}_{}_{}.tex'.format(filename,
                    group['dataset'][0]['title'].replace('_','-'), ds // idf.model.num_dofs),
                    figureheight = '\\figureheight', figurewidth = '\\figurewidth', show_info=False)

        if idf.opt['outputAs'] == 'html':
            path = os.path.dirname(os.path.abspath(__file__))
            template_environment = Environment(autoescape=False,
                                               loader=FileSystemLoader(os.path.join(path, '../output')),
                                               trim_blocks=False)

            context = { 'figures': figures, 'text': self.text }
            outfile = os.path.join(path, '..', 'output', filename)
            import codecs
            with codecs.open(outfile, 'w', 'utf-8') as f:
                html = template_environment.get_template("templates/index.html").render(context)
                f.write(html)

            print("Saved output at file://{}".format(outfile))
        elif idf.opt['outputAs'] == 'interactive':
            #keep non-blocking plot windows open
            plt.show()
        elif idf.opt['outputAs'] == 'pdf':
            pp.close()

    def openURL(self):
        import subprocess, time

        time.sleep(1)
        print("Opening output...")
        #call(["open", '"http://127.0.0.1:8000"'])
        filepath = "http://127.0.0.1:8080/output/output.html"

        if sys.platform.startswith('darwin'):
            subprocess.call(('open', filepath))
        elif os.name == 'nt':
            os.startfile(filepath)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', filepath))

    def runServer(self):
        import http.server
        import socketserver
        import threading

        port = 8080
        Handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), Handler)
        threading.Thread(target=self.openURL).start()
        print("serving on port {}, press ctrl-c to stop".format(port))
        httpd.serve_forever()

