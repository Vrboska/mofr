import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, rcParams, rcParamsDefault
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class LiftCurveEvaluator(Evaluator):

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
        rcParams.update(rcParamsDefault)
        colors = cycle(colors_)

        f, ax = plt.subplots(figsize=figsize_)
        lines = []
        labels = []

        n_scores=len(self.scores)
        x_= [(x/10) for x in range (1,11)] #x-axis with different lifts
        max_lift=1.1

        #plot each lift curve for each score
        for i, color in zip(range(n_scores), colors):
            target_=self.targets[0]
            score_=self.scores[i]
            df_=self.data[self.data[target_[1]]==1] #filtering for only target-observable cases

            lift_curve = [metrics.liftN(df_[target_[0]], df_[score_], x) for x in x_]
            max_lift=max(max(lift_curve), max_lift)
            l, = plt.plot(x_, lift_curve, color=color, lw=2)
            lines.append(l)
            labels.append(f'{score_}')

        #plotting the base line
        _x=[x/11 for x in range(12)]
        _y=[1 for x in range(12)]
        plt.plot(_x, _y, linestyle='--', color='blue')
                
        #set plotting parameters
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.02])
        plt.ylim([0.9, max_lift+0.2])
        plt.xlabel('Lift percentage', axes=ax)
        plt.ylabel('Lift', axes=ax)
        plt.title(f'Lifts for target "{self.targets[0][0]}"', axes=ax)
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