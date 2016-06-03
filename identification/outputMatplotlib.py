import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class OutputMatplotlib(object):
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

    def render(self):
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
        plt.show()
