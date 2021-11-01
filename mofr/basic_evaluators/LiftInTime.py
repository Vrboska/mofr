import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class LiftInTimeEvaluator(Evaluator):

      def __init__(self, data=None, targets=None, scores=None, time_column=None):
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
            self.time_column=time_column

      def d(self, data=None):
            self.data=data
            return self       

      def t(self, targets=None):
            self.targets=targets
            return self   

      def s(self, scores=None):
            self.scores=scores
            return self      

      def tc(self, time_column=None):
            self.time_column=time_column
            return self 

      def get_graph(self):

            # setup plot details
            colors = cycle(colors_)

            plt.figure(figsize=figsize_)

            """
            The idea is to have a graph of Lift metric (metrics.lift) in time (x-axis chronologically ordered) for a given
            target and given scores, using the legend,labels and colors as in e.g. ROCCurve.py.
            """
            lines = []
            labels = []

            n_scores=len(self.scores)


            #plot each LIFT curve for each score
            for i, color in zip(range(n_scores), colors):
                  target_=self.targets[0]
                  score_=self.scores[i]
                  df_=self.data[self.data[target_[1]]==1] #filtering for only target-observable cases

                  lift_by_month=df_.groupby('month').apply(lambda x: metrics.lift(x[target_[0]], x[score_]) ).to_frame('LIFT')
                  lift_by_month.reset_index(level=0,inplace=True)
                  _x=lift_by_month['month'].apply(int)
                  _y=lift_by_month['LIFT']
                  l, = plt.plot(_x, _y, color=color, lw=2)
                  lines.append(l)
                  labels.append(f'{score_}')

            #set plotting parameters
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.25)
            plt.ticklabel_format(useOffset=False)
            plt.xticks(range(min(_x), max(_x)+1))
            #plt.xlim(min(_x)-0.1,max(_x)+0.1)
            #plt.ylim(-0.01,1.03)
            plt.xlabel('Month')
            plt.ylabel('LIFT')
            plt.title(f'LIFT (10%) in time for target "{self.targets[0][0]}"')
            plt.legend(lines, labels) #, loc=(0, -.38), prop=dict(size=14)

            plt.show()    

            return self

      def get_table(self):        
            """
            The idea is to have a table corresponding to the data shown in graph in a following format (or similar):
                                    
                              Lift on 'name of the target column'
            Time             Model1         Model2      Model3       
            ------------------------------------------------------------
            202001          2.23            2.25           ...
            202002          2.33            ...            ...
            202003          2.54            ...            ...
            ------------------------------------------------------------
            All             2.42            ...            ...


            Let's implement this using pandas.pivot_table since it will make it easier later to combine different
            tables into one.
            """
    
            def lift_zipped(x):
                  """Auxilliary function for calculating Lift in pivot table."""
                  list_=list(zip(*x))
                  target_=list_[0]
                  score_=list_[1]
                  ts=pd.DataFrame(target_, columns=['target_'])
                  ts['score_']=score_   
                              
                  return metrics.lift(ts['target_'], ts['score_'])
                  
            n_scores=len(self.scores)

            tables=[]

            #creating pivot table for each score
            for i in range(n_scores):          
                  target_=self.targets[0]
                  score_=self.scores[i]
                  df_=self.data[self.data[target_[1]]==1] #filtering for only target-observable cases

                  df_['target_score_']=list(zip(df_[target_[0]], df_[score_]))

                  pt=pd.pivot_table(df_, values=['target_score_'], index='month', columns=None, aggfunc=lift_zipped, fill_value=None, margins=True, dropna=True, margins_name='All')
                  pt.columns=[score_]
                  tables.append(pt)

            #gathering the final table together
            from functools import reduce
            final_table=reduce(lambda x, y: x.merge(y,left_index=True,right_index=True), tables)
            final_table=final_table.style.set_table_attributes("style='display:inline'").set_caption(f'Lift on target "{target_[0]}"')  
            self.table=final_table

            return self
