#-*- coding: utf-8 -*-

import os
from IPython import embed
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import colorama
from colorama import Fore, Back, Style

#plot colors
colors = []
"""
#some random colors
colors += [[ 0.97254902,  0.62745098,  0.40784314],
          [ 0.0627451 ,  0.53333333,  0.84705882],
          [ 0.15686275,  0.75294118,  0.37647059],
          [ 0.90980392,  0.37647059,  0.84705882],
          [ 0.84705882,  0.        ,  0.1254902 ],
          [ 0.18823529,  0.31372549,  0.09411765],
          [ 0.50196078,  0.40784314,  0.15686275]
         ]
"""

# color triplets
color_triplets_6 = [
           [ 0.29019608,  0.43529412,  0.89019608],
           [ 0.52156863,  0.58431373,  0.88235294],
           [ 0.70980392,  0.73333333,  0.89019608],
           [ 0.90196078,  0.68627451,  0.7254902 ],
           [ 0.87843137,  0.48235294,  0.56862745],
           [ 0.82745098,  0.24705882,  0.41568627],
          ]

grayscale_6 = [
               [ 0.        ,  0.        ,  0.        ],
               [ 0.13333333,  0.13333333,  0.13333333],
               [ 0.26666667,  0.26666667,  0.26666667],
               [ 0.4       ,  0.4       ,  0.4       ],
               [ 0.53333333,  0.53333333,  0.53333333],
               [ 0.67058824,  0.67058824,  0.67058824]
              ]

#set some more colors for higher DOF
from palettable.tableau import Tableau_10, Tableau_20
colors += Tableau_10.mpl_colors[0:6] + Tableau_20.mpl_colors

class OutputConsole(object):
    @staticmethod
    def render(idf, summary_only=False):
        """Do some pretty printing."""

        colorama.init(autoreset=False)

        if not idf.opt['useEssentialParams']:
            idf.stdEssentialIdx = range(0, idf.model.num_params)
            idf.stdNonEssentialIdx = []

        import iDynTree
        #if requested, load params from other urdf for comparison
        if idf.urdf_file_real:
            dc = iDynTree.DynamicsComputations()
            dc.loadRobotModelFromFile(idf.urdf_file_real)
            tmp = iDynTree.VectorDynSize(idf.model.num_params)
            dc.getModelDynamicsParameters(tmp)
            xStdReal = tmp.toNumPy()
            xBaseReal = np.dot(idf.model.Binv, xStdReal)

        if idf.opt['showStandardParams']:
            # convert params to COM-relative instead of frame origin-relative (linearized parameters)
            if idf.opt['outputBarycentric']:
                xStd = idf.paramHelpers.paramsLink2Bary(idf.model.xStd)
                xStdModel = idf.paramHelpers.paramsLink2Bary(idf.model.xStdModel)
                if not summary_only:
                    print("Barycentric (relative to COM) Standard Parameters")
            else:
                xStd = idf.model.xStd
                xStdModel = idf.model.xStdModel
                if not summary_only:
                    print("Linear (relative to Frame) Standard Parameters")

            # collect values for parameters
            description = idf.model.generator.getDescriptionOfParameters()
            idx_p = 0   #count (std) params
            idx_ep = 0  #count essential params
            lines = list()
            sum_diff_r_pc_ess = 0
            sum_diff_r_pc_all = 0
            sum_pc_delta_all = 0
            sum_pc_delta_ess = 0
            for d in description.replace(r'Parameter ', '#').split('\n'):
                if idf.opt['outputBarycentric']:
                    d = d.replace(r'first moment', 'center')
                #add symbol for each parameter
                d = d.replace(r':', ': {} -'.format(idf.model.param_syms[idx_p]))

                #print beginning of each link block in green
                if idx_p % 10 == 0:
                    d = Fore.GREEN + d

                #get some error values for each parameter
                approx = xStd[idx_p]
                apriori = xStdModel[idx_p]
                diff = approx - apriori

                if idf.urdf_file_real:
                    real = xStdReal[idx_p]
                    # set real params that are 0 to some small value
                    #if real == 0: real = 0.01

                    # get error percentage (new to real)
                    # so if 100% are the real value, how big is the error
                    diff_real = approx - real
                    if real != 0:
                        diff_r_pc = (100*diff_real)/real
                    else:
                        diff_r_pc = (100*diff_real)/0.01

                    #add to final error percent sum
                    sum_diff_r_pc_all += np.abs(diff_r_pc)
                    if idx_p in idf.stdEssentialIdx:
                        sum_diff_r_pc_ess += np.abs(diff_r_pc)

                    # get error percentage (new to apriori)
                    diff_apriori = apriori - real
                    #if apriori == 0: apriori = 0.01
                    if diff_apriori != 0:
                        pc_delta = np.abs((100/diff_apriori)*diff_real)
                    elif diff_real > 0.0001:
                        pc_delta = np.abs((100/0.01)*diff_real)
                    else:
                        #both real and a priori are zero, error is still at 100% (of zero error)
                        pc_delta = 100
                    sum_pc_delta_all += pc_delta
                    if idx_p in idf.stdEssentialIdx:
                        sum_pc_delta_ess += pc_delta
                else:
                    # get percentage difference between apriori and identified values
                    # (shown when real values are not known)
                    if apriori != 0:
                        diff_pc = (100*diff)/apriori
                    else:
                        diff_pc = (100*diff)/0.01

                #values for each line
                if idf.opt['useEssentialParams'] and idx_ep < idf.num_essential_params and idx_p in idf.stdEssentialIdx:
                    sigma = idf.p_sigma_x[idx_ep]
                else:
                    sigma = 0.0

                if idf.urdf_file_real:
                    vals = [real, apriori, approx, diff, np.abs(diff_r_pc), pc_delta, sigma, d]
                else:
                    vals = [apriori, approx, diff, diff_pc, sigma, d]
                lines.append(vals)

                if idf.opt['useEssentialParams'] and idx_p in idf.stdEssentialIdx:
                    idx_ep+=1
                idx_p+=1
                if idx_p == len(xStd):
                    break

            if idf.urdf_file_real:
                column_widths = [13, 13, 13, 7, 7, 7, 6, 45]
                precisions = [8, 8, 8, 4, 1, 1, 3, 0]
            else:
                column_widths = [13, 13, 7, 7, 6, 45]
                precisions = [8, 8, 4, 1, 3, 0]

            if not summary_only:
                # print column header
                template = ''
                for w in range(0, len(column_widths)):
                    template += '|{{{}:{}}}'.format(w, column_widths[w])
                if idf.urdf_file_real:
                    print template.format("'Real'", "A priori", "Approx", "Change", "%e", u"Δ%e".encode('utf-8'), u"%σ".encode('utf-8'), "Description")
                else:
                    print template.format("A priori", "Approx", "Change", "%e", u"%σ".encode('utf-8'), "Description")

                # print values/description
                template = ''
                for w in range(0, len(column_widths)):
                    if(type(lines[0][w]) == str):
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
                    print t,
                    idx_p+=1
                    print Style.RESET_ALL
                print "\n"

        ### print base params
        if idf.opt['showBaseParams'] and not summary_only and idf.opt['estimateWith'] not in ['urdf', 'std_direct']:
            print("Base Parameters and Corresponding standard columns")
            if not idf.opt['useEssentialParams']:
                baseEssentialIdx = range(0, idf.model.num_base_params)
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
                    real = xBaseReal[idx_p]
                    error = new - real
                    sum_error_all_base += np.abs(error)

                # collect linear dependencies for this param
                #deps = np.where(np.abs(idf.linear_deps[idx_p, :])>0.1)[0]
                #dep_factors = idf.linear_deps[idx_p, deps]

                param_columns = " = "
                param_columns += "{}".format(idf.model.base_deps[idx_p])
                #for p in range(0, len(deps)):
                #    param_columns += ' {:.4f}*|{}|'.format(dep_factors[p], idf.P[idf.num_base_params:][deps[p]])


                if idf.opt['useEssentialParams'] and idx_p in idf.baseEssentialIdx:
                    sigma = idf.p_sigma_x[idx_ep]
                else:
                    sigma = 0.0

                if idf.urdf_file_real:
                    lines.append((idx_p, real, old, new, diff, error, sigma, param_columns))
                else:
                    lines.append((old, new, diff, sigma, param_columns))

                if idf.opt['useEssentialParams'] and idx_p in idf.baseEssentialIdx:
                    idx_ep+=1

            if idf.urdf_file_real:
                column_widths = [3, 13, 13, 13, 7, 7, 6, 30]   # widths of the columns
                precisions = [0, 8, 8, 8, 4, 4, 3, 0]         # numerical precision
            else:
                column_widths = [13, 13, 7, 6, 30]   # widths of the columns
                precisions = [8, 8, 4, 3, 0]         # numerical precision

            if not summary_only:
                # print column header
                template = ''
                for w in range(0, len(column_widths)):
                    template += '|{{{}:{}}}'.format(w, column_widths[w])
                if idf.urdf_file_real:
                    print template.format("\#", "Real", "Model", "Approx", "Change", "Error", u"%σ".encode('utf-8'), "Description")
                else:
                    print template.format("Model", "Approx", "Change", u"%σ".encode('utf-8'), "Description")

                # print values/description
                template = ''
                for w in range(0, len(column_widths)):
                    if(type(lines[0][w]) == str):
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
                    print t,
                    idx_p+=1
                    print Style.RESET_ALL

        if idf.opt['selectBlocksFromMeasurements']:
            if len(idf.data.usedBlocks):
                print "used {} of {} blocks: {}".format(len(idf.data.usedBlocks),
                                                        len(idf.data.usedBlocks)+len(idf.data.unusedBlocks),
                                                        [b for (b,bs,cond,linkConds) in idf.data.usedBlocks])
            else:
                print "\ncurrent block: {}".format(idf.data.block_pos)
            #print "unused blocks: {}".format(idf.unusedBlocks)
            print "condition number: {}".format(la.cond(idf.model.YBase))

        if idf.opt['showStandardParams']:
            cons_apriori = idf.paramHelpers.checkPhysicalConsistency(idf.model.xStdModel)
            print("Per-link physical consistency (a priori): {}".format(cons_apriori))
            if False in cons_apriori.values():
                print Fore.RED + "a priori parameters not consistent!" + Fore.RESET
            print("Per-link physical consistency (identified): {}".format(idf.paramHelpers.checkPhysicalConsistencyNoTriangle(idf.model.xStd)))
            print("Per-link full physical consistency (identified): {}".format(idf.paramHelpers.checkPhysicalConsistency(idf.model.xStd)))

        print("Estimated overall mass: {} vs. apriori {}".format(np.sum(idf.model.xStd[0::10]), np.sum(idf.model.xStdModel[0::10])))

        if idf.urdf_file_real:
            if idf.opt['showStandardParams']:
                if idf.opt['useEssentialParams']:
                    print("Mean relative error of essential std params: {}%".\
                            format(sum_diff_r_pc_ess/len(idf.stdEssentialIdx)))
                print("Mean relative error of all std params: {}%".format(sum_diff_r_pc_all/len(idf.model.xStd)))

                if idf.opt['useEssentialParams']:
                    print("Mean error delta (apriori error vs approx error) of essential std params: {}%".\
                            format(sum_pc_delta_ess/len(idf.stdEssentialIdx)))
                print("Mean error delta (apriori error vs approx error) of all std params: {}%".\
                        format(sum_pc_delta_all/len(idf.model.xStd)))
                sq_error_apriori = np.square(la.norm(xStdReal - idf.model.xStdModel))
                sq_error_idf = np.square(la.norm(xStdReal - idf.model.xStd))
                print( "Squared distance of parameter vectors (apriori, identified) to real: {} vs. {}".\
                        format(sq_error_apriori, sq_error_idf))
            if idf.opt['showBaseParams'] and not summary_only and idf.opt['estimateWith'] not in ['urdf', 'std_direct']:
                print("Mean error (apriori - approx) of all base params: {:.5f}".\
                        format(sum_error_all_base/len(idf.model.xBase)))

        idf.estimateRegressorTorques(estimateWith='urdf')   #estimate torques with CAD params
        idf.apriori_error = sla.norm(idf.tauEstimated-idf.tauMeasured)*100/sla.norm(idf.tauMeasured)
        idf.estimateRegressorTorques()   #estimate torques again with identified parameters
        idf.res_error = sla.norm(idf.tauEstimated-idf.tauMeasured)*100/sla.norm(idf.tauMeasured)
        print("Relative residual error (torque prediction): {}% vs. A priori error: {}%".\
                format(idf.res_error, idf.apriori_error))

class OutputMatplotlib(object):
    def __init__(self, datasets, html=True):
        self.datasets = datasets
        self.html = html

    def render(self, filename='output.html'):
        if self.html:
            # write matplotlib/d3 plots to html file
            import matplotlib.pyplot as plt,mpld3
            from mpld3 import plugins
            from jinja2 import Environment, FileSystemLoader
        else:
            # show plots in separate matplotlib windows
            import matplotlib
            import matplotlib.pyplot as plt
        figures = list()
        for group in self.datasets:
            fig, axes = plt.subplots(len(group['dataset']), sharex=True)
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
                ax = axes[d_i]
                ax.set_title(d['title'])
                if group['unified_scaling']:
                    ax.set_ylim([ymin, ymax])
                for data_i in range(0, len(d['data'])):
                    if len(d['data'][data_i].shape) > 1:
                        #data matrices
                        for i in range(0, d['data'][data_i].shape[1]):
                            l = group['labels'][i] if data_i == 0 else ''
                            if i < 6 and group.has_key('contains_base') and group['contains_base']:
                                ls = '--'
                            else:
                                ls = '-'
                            ax.plot(d['time'], d['data'][data_i][:, i], label=l, color=colors[i], alpha=1-(data_i/2.0), linestyle=ls)
                    else:
                        #data vector
                        ax.plot(d['time'], d['data'][d_i], label=d['title'], color=colors[0], alpha=1-(data_i/2.0))
                if group.has_key('y_label'):
                    ax.set_ylabel(group['y_label'])

            ax.set_xlabel("Time (s)")

            plt.setp([a.get_xticklabels() for a in axes[:-1]], visible=False)
            if self.html:
                #TODO: show legend properly (see mpld3 bug #274)
                handles, labels = ax.get_legend_handles_labels()
                #leg = fig.legend(handles, labels, loc='upper right', fancybox=True, fontsize=10, title='')
                leg = axes[0].legend(handles, labels, loc='upper right', fancybox=True, fontsize=10, title='')
            else:
                handles, labels = ax.get_legend_handles_labels()
                leg = plt.figlegend(handles, labels, loc='upper right', fancybox=True, fontsize=10, title='')
            #plt.tight_layout()

            if self.html:
                plugins.clear(fig)
                plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=False),
                                plugins.MousePosition(fontsize=14))
                figures.append(mpld3.fig_to_html(fig))
            else:
                plt.show()

        if self.html:
            path = os.path.dirname(os.path.abspath(__file__))
            template_environment = Environment(autoescape=False,
                                               loader=FileSystemLoader(os.path.join(path, '../output')),
                                               trim_blocks=False)

            context = { 'figures': figures }
            outfile = os.path.join(path, '..', 'output', 'output.html')
            with open(outfile, 'w') as f:
                html = template_environment.get_template("templates/index.html").render(context)
                f.write(html)

            print("Saved output at file://{}".format(outfile))

    def openURL(self):
        import sys, subprocess, time

        time.sleep(1)
        print "Opening output..."
        #call(["open", '"http://127.0.0.1:8000"'])
        filepath = "http://127.0.0.1:8080/output/output.html"

        if sys.platform.startswith('darwin'):
            subprocess.call(('open', filepath))
        elif os.name == 'nt':
            os.startfile(filepath)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', filepath))

    def runServer(self):
        import SimpleHTTPServer
        import SocketServer
        import threading

        port = 8080
        Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
        httpd = SocketServer.TCPServer(("", port), Handler)
        threading.Thread(target=self.openURL).start()
        print("serving on port {}, press ctrl-c to stop".format(port))
        httpd.serve_forever()

