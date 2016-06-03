
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
from jinja2 import Environment, FileSystemLoader

from IPython import embed

class OutputHTML(object):
    def __init__(self, datasets, time, colors, jointNames):
        self.datasets = datasets
        self.T = time
        # scale all figures to same ranges and add some margin
        self.ymin = np.min([datasets[0][0][0], datasets[1][0][0]])
        self.ymin += self.ymin * 0.05
        self.ymax = np.max([datasets[0][0][0], datasets[1][0][0]])
        self.ymax += self.ymax * 0.05
        self.jointNames = jointNames
        self.colors = colors

    def render(self, filename='output.html'):
        figures = list()
        for (data, title) in self.datasets:
            fig = plt.figure()
            plt.ylim([self.ymin, self.ymax])
            plt.title(title)
            for d_i in range(0, len(data)):
                if len(data[d_i].shape) > 1:
                    #data matrices
                    for i in range(0, data[d_i].shape[1]):
                        l = self.jointNames[i] if d_i == 0 else ''  # only put joint names in the legend once
                        plt.plot(self.T, data[d_i][:, i], label=l, color=self.colors[i], alpha=1-(d_i/2.0))
                else:
                    #data vector
                    plt.plot(self.T, data[d_i], label=title, color=self.colors[0], alpha=1-(d_i/2.0))

            leg = plt.legend(loc='best', fancybox=True, fontsize=10, title='')
            leg.draggable()
            plugins.clear(fig)
            plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom(enabled=True),
                            plugins.MousePosition(fontsize=14))
            figures.append(mpld3.fig_to_html(fig))

        path = os.path.dirname(os.path.abspath(__file__))
        template_environment = Environment(autoescape=False,
                                           loader=FileSystemLoader(os.path.join(path, 'output')),
                                           trim_blocks=False)

        context = { 'figures': figures }
        with open(os.path.join(path, 'output', 'output.html'), 'w') as f:
            html = template_environment.get_template("templates/index.html").render(context)
            f.write(html)



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

