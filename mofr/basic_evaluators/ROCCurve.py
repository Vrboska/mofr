import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class ROCCurveEvaluator(Evaluator):

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

    def get_graph(self, plot=True):

        # setup plot details
        colors = cycle(colors_)

        f, ax = plt.subplots(figsize=figsize_)
        lines = []
        labels = []

        n_scores=len(self.scores)
        

        #plot each ROC curve for each score
        for i, color in zip(range(n_scores), colors):
            target_=self.targets[0]
            score_=self.scores[i]
            df_=self.data[self.data[target_[1]]==1] #filtering for only target-observable cases

            _fpr, _tpr, _ = metrics.roc_curve(df_[target_[0]], df_[score_]) 
            l, = plt.plot(_fpr, _tpr, color=color, lw=2)
            lines.append(l)
            labels.append(f'{score_}')

        #plotting the base line
        _x=[x/len(_fpr) for x in range(len(_fpr))]
        _y=[x/len(_fpr) for x in range(len(_fpr))]
        plt.plot(_x, _y, linestyle='--', color='blue', axes=ax)
        
        #set plotting parameters
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim(-0.01,1.01)
        plt.ylim(-0.01,1.03)
        plt.xlabel('False Positive Rate', axes=ax)
        plt.ylabel('True Positive Rate', axes=ax)
        plt.title(f'ROC curves for target "{self.targets[0][0]}"', axes=ax)
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