#-*- coding: utf-8 -*-

import os
import numpy as np

colors = [[ 0.97254902,  0.62745098,  0.40784314],
          [ 0.0627451 ,  0.53333333,  0.84705882],
          [ 0.15686275,  0.75294118,  0.37647059],
          [ 0.90980392,  0.37647059,  0.84705882],
          [ 0.84705882,  0.        ,  0.1254902 ],
          [ 0.18823529,  0.31372549,  0.09411765],
          [ 0.50196078,  0.40784314,  0.15686275]
         ]

class OutputConsole(object):
    @staticmethod
    def render(idf, summary_only=False):
        """Do some pretty printing."""

        import numpy.linalg as la
        import scipy.linalg as sla

        import colorama
        from colorama import Fore, Back, Style
        colorama.init(autoreset=False)

        if not idf.useEssentialParams:
            idf.stdEssentialIdx = range(0, idf.N_PARAMS)
            idf.stdNonEssentialIdx = []

        import iDynTree
        #if requested, load params from other urdf for comparison
        if idf.urdf_file_real:
            dc = iDynTree.DynamicsComputations()
            dc.loadRobotModelFromFile(idf.urdf_file_real)
            tmp = iDynTree.VectorDynSize(idf.N_PARAMS)
            dc.getModelDynamicsParameters(tmp)
            xStdReal = tmp.toNumPy()
            xBaseReal = np.dot(idf.B.T, xStdReal)

        if idf.showStandardParams:
            # convert params to COM-relative instead of frame origin-relative (linearized parameters)
            if idf.outputBarycentric:
                if not idf.robotranRegressor:
                  xStd = idf.helpers.paramsLink2Bary(idf.xStd)
                xStdModel = idf.helpers.paramsLink2Bary(idf.xStdModel)
                if not summary_only:
                    print("Barycentric (relative to COM) Standard Parameters")
            else:
                xStd = idf.xStd
                xStdModel = idf.xStdModel
                if not summary_only:
                    print("Linear (relative to Frame) Standard Parameters")

            # collect values for parameters
            description = idf.generator.getDescriptionOfParameters()
            idx_p = 0   #count (std) params
            idx_ep = 0  #count essential params
            lines = list()
            sum_diff_r_pc_ess = 0
            sum_diff_r_pc_all = 0
            sum_pc_delta_all = 0
            sum_pc_delta_ess = 0
            for d in description.replace(r'Parameter ', '# ').replace(r'first moment', 'center').split('\n'):
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
                if idf.useEssentialParams and idx_ep < idf.num_essential_params and idx_p in idf.stdEssentialIdx:
                    sigma = idf.p_sigma_x[idx_ep]
                else:
                    sigma = 0.0

                if idf.urdf_file_real:
                    vals = [real, apriori, approx, diff, np.abs(diff_r_pc), pc_delta, sigma, d]
                else:
                    vals = [apriori, approx, diff, diff_pc, sigma, d]
                lines.append(vals)

                if idf.useEssentialParams and idx_p in idf.stdEssentialIdx:
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
        if idf.showBaseParams and not summary_only and idf.estimateWith not in ['urdf', 'std_direct']:
            print("Base Parameters and Corresponding standard columns")
            if not idf.useEssentialParams:
                baseEssentialIdx = range(0, idf.num_base_params)
                baseNonEssentialIdx = []
                xBase_essential = idf.xBase
            else:
                baseEssentialIdx = idf.baseEssentialIdx
                baseNonEssentialIdx = idf.baseNonEssentialIdx
                xBase_essential = idf.xBase_essential

            # collect values for parameters
            idx_ep = 0
            lines = list()
            for idx_p in range(0, idf.num_base_params):
                if idf.useEssentialParams: # and xBase_essential[idx_p] != 0:
                    new = xBase_essential[idx_p]
                else:
                    new = idf.xBase[idx_p]
                old = idf.xBaseModel[idx_p]
                diff = new - old
                if idf.urdf_file_real:
                    real = xBaseReal[idx_p]
                    error = new - real

                # collect linear dependencies for this param
                deps = np.where(np.abs(idf.linear_deps[idx_p, :])>0.1)[0]
                dep_factors = idf.linear_deps[idx_p, deps]

                param_columns = ' |{}|'.format(idf.independent_cols[idx_p])
                if len(deps):
                    param_columns += " deps:"
                for p in range(0, len(deps)):
                    param_columns += ' {:.4f}*|{}|'.format(dep_factors[p], idf.P[idf.num_base_params:][deps[p]])

                if idf.useEssentialParams and idx_p in idf.baseEssentialIdx:
                    sigma = idf.p_sigma_x[idx_ep]
                else:
                    sigma = 0.0

                if idf.urdf_file_real:
                    lines.append((idx_p, real, old, new, diff, error, sigma, param_columns))
                else:
                    lines.append((old, new, diff, sigma, param_columns))

                if idf.useEssentialParams and idx_p in idf.baseEssentialIdx:
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
                    if idf.showEssentialSteps and l[-2] == np.max(idf.p_sigma_x):
                        t = Fore.CYAN + t
                    print t,
                    idx_p+=1
                    print Style.RESET_ALL

        if idf.selectBlocksFromMeasurements:
            if len(idf.usedBlocks):
                print "used {} of {} blocks: {}".format(len(idf.usedBlocks),
                                                        len(idf.usedBlocks)+len(idf.unusedBlocks),
                                                        [b for (b,bs,cond,linkConds) in idf.usedBlocks])
            else:
                print "\ncurrent block: {}".format(idf.block_pos)
            #print "unused blocks: {}".format(idf.unusedBlocks)
            print "condition number: {}".format(la.cond(idf.YBase))

        if idf.showStandardParams:
            print("Per-link physical consistency (a priori): {}".format(idf.helpers.checkPhysicalConsistency(idf.xStdModel)))
            print("Per-link physical consistency (identified): {}".format(idf.helpers.checkPhysicalConsistency(idf.xStd)))

        if idf.urdf_file_real:
            if idf.showStandardParams:
                if idf.useEssentialParams:
                    print("Mean relative error of essential std params: {}%".\
                            format(sum_diff_r_pc_ess/len(idf.stdEssentialIdx)))
                print("Mean relative error of all std params: {}%".format(sum_diff_r_pc_all/len(idf.xStd)))

                if idf.useEssentialParams:
                    print("Mean error delta (apriori error vs approx error) of essential std params: {}%".\
                            format(sum_pc_delta_ess/len(idf.stdEssentialIdx)))
                print("Mean error delta (apriori error vs approx error) of all std params: {}%".\
                        format(sum_pc_delta_all/len(idf.xStd)))

        idf.estimateRegressorTorques(estimateWith='urdf')
        idf.apriori_error = sla.norm(idf.tauEstimated-idf.tauMeasured)*100/sla.norm(idf.tauMeasured)
        idf.estimateRegressorTorques()
        idf.res_error = sla.norm(idf.tauEstimated-idf.tauMeasured)*100/sla.norm(idf.tauMeasured)
        print("Relative residual error (torque prediction): {}% vs. A priori error: {}%".\
                format(idf.res_error, idf.apriori_error))

class OutputMatplotlib(object):
    def __init__(self, datasets, jointNames):
        import matplotlib
        import matplotlib.pyplot as plt

        self.datasets = datasets
        # scale all figures to same ranges and add some margin
        self.ymin = np.min([datasets[0][0][0], datasets[1][0][0]]) * 1.05
        self.ymax = np.max([datasets[0][0][0], datasets[1][0][0]]) * 1.05
        self.jointNames = jointNames

    def render(self):
        """show plots in separate matplotlib windows"""
        for (data, time, title) in self.datasets:
            fig = plt.figure()
            plt.ylim([self.ymin, self.ymax])
            plt.title(title)
            for d_i in range(0, len(data)):
                if len(data[d_i].shape) > 1:
                    #data matrices
                    for i in range(0, data[d_i].shape[1]):
                        l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                        plt.plot(time, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
                else:
                    #data vector
                    plt.plot(time, data[d_i], label=title, color=colors[0], alpha=1-(d_i/2.0))

            leg = plt.legend(loc='best', fancybox=True, fontsize=10, title='')
            leg.draggable()
        plt.show()


class OutputHTML(object):
    def __init__(self, datasets, jointNames):
        self.datasets = datasets
        # scale all figures to same ranges and add some margin
        self.ymin = np.min([datasets[0][0][0], datasets[1][0][0]]) * 1.05
        self.ymax = np.max([datasets[0][0][0], datasets[1][0][0]]) * 1.05
        self.jointNames = jointNames

    def render(self, filename='output.html'):
        """write matplotlib/d3 plots to html file"""
        import matplotlib.pyplot as plt,mpld3
        from mpld3 import plugins
        from jinja2 import Environment, FileSystemLoader

        figures = list()
        for (data, time, title) in self.datasets:
            fig = plt.figure()
            plt.ylim([self.ymin, self.ymax])
            plt.title(title)
            for d_i in range(0, len(data)):
                if len(data[d_i].shape) > 1:
                    #data matrices
                    for i in range(0, data[d_i].shape[1]):
                        l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                        plt.plot(time, data[d_i][:, i], label=l, color=colors[i], alpha=1-(d_i/2.0))
                else:
                    #data vector
                    plt.plot(time, data[d_i], label=title, color=colors[0], alpha=1-(d_i/2.0))

            leg = plt.legend(loc='best', fancybox=True, fontsize=10, title='')
            leg.draggable()
            plugins.clear(fig)
            plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=False),
                            plugins.MousePosition(fontsize=14))
            figures.append(mpld3.fig_to_html(fig))

        path = os.path.dirname(os.path.abspath(__file__))
        template_environment = Environment(autoescape=False,
                                           loader=FileSystemLoader(os.path.join(path, 'output')),
                                           trim_blocks=False)

        context = { 'figures': figures }
        outfile = os.path.join(path, 'output', 'output.html')
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

