import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, rcParams, rcParamsDefault
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class PRCurveEvaluator(Evaluator):

    def __init__(self, data=None, targets=None, scores=None):
        """
        data: The pandas dataframe containing all the necessary columns.

        targets: These should be the list of binary targets along with their 
        observability flags as follows. [('target1','target1_obs'),('target2', 'target2_obs)]
        Only the first target will be displayed in the graph at the moment.

        scores: List of score columns as follows ['score1', 'score2', 'score3']
        """
        self.data=data
        self.targets=targets
        self.scores=scores

    def d(self, data=None):
          self.data=data
          return self       

    def t(self, targets=None):
          self.targets=targets
          return self   

    def s(self, scores=None):
          self.scores=scores
          return self      

    def get_graph(self, isocurves='off', plot=True):

        # setup plot details
        rcParams.update(rcParamsDefault)
        colors = cycle(colors_)

        f, ax = plt.subplots(figsize=figsize_)
        lines = []
        labels = []

        n_scores=len(self.scores)
        
        if isocurves=='on':
            f_scores = np.linspace(0.2, 0.8, num=4)

            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02), axes=ax)
            lines.append(l)
            labels.append('iso-f1 curves')

        #plot each ROC curve for each score
        for i, color in zip(range(n_scores), colors):
            target_=self.targets[0]
            score_=self.scores[i]
            df_=self.data[self.data[target_[1]]==1] #filtering for only target-observable cases

            lr_precision, lr_recall, _ = metrics.precision_recall_curve(df_[target_[0]], df_[score_])
            l, = plt.plot(lr_recall, lr_precision, color=color, lw=2)
            lines.append(l)
            labels.append(f'{score_}')

        #plotting the base line
        no_skill = len(df_[df_[target_[0]]==1]) / len(df_)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--',color='blue')

        #set plotting parameters
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', axes=ax)
        plt.ylabel('Precision', axes=ax)
        plt.title(f'PR curves for target "{self.targets[0][0]}"', axes=ax)
        ax.legend(lines, labels) #, loc=(0, -.38), prop=dict(size=14)
        ax.grid(True)

        if plot==True:
            plt.show()  

        self.graph=f
        self.axis=ax

        plt.close()
        
        return self

    def get_table(self):
        pass

