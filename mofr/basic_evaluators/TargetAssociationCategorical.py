import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_, max_categories_


class TargetAssociationCategoricalEvaluator(Evaluator):

    def __init__(self, data=None, targets=None, predictor_column=None, time_column=None):
      """
      data: The pandas dataframe containing all the necessary columns.

      targets: These should be the list of binary targets along with their 
      observability flags as follows. [('target1','target1_obs'),('target2', 'target2_obs)]
      Only the first target will be displayed in the graph at the moment.

      predictor_column: The name of the column containing the categorical predictor.
      There should be no more than 20 unique categories for this perecdictor. Binning should be used
      for predictors with higher number of unique categories. The predictor should be in string format or 
      at least convertible into string.

      time_column: The name of the column containing the time information. This should be an integer or at least convertibleto integer
      e.g. 2019 for year, 202110 for month, 20211001 for day etc.
      """
      self.data=data
      self.targets=targets
      self.predictor_column=predictor_column
      self.time_column=time_column 
 
    def d(self, data=None):
      self.data=data
      return self  

    def t(self, targets=None):
      self.targets=targets
      return self  

    def pc(self, predictor_column=None):
      self.predictor_column=predictor_column
      return self      

    def tc(self, time_column=None):
      self.time_column=time_column
      return self 

    def get_graph(self):

      # setup plot details
      colors = cycle(colors_)

      plt.figure(figsize=figsize_)

      """
      The idea is to have a graph of Share of each category in time (x-axis chronologically ordered) for the 
      given predictor
      """
      lines = []
      labels = []
      
      #set up data details
      target_=self.targets[0]
      df_=self.data[self.data[target_[1]]==1]#filtering for only target-observable cases
      df_[self.predictor_column]=df_[self.predictor_column].apply(str)
      df_[self.time_column]=df_[self.time_column].apply(int)
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique categories!'
      assert n_categories<=max_categories_, f'The predictor column specified has more than {max_categories_} unique categories!'

      #  produce tablepd.crosstab(index=df_[self.predictor_column], columns=df_[self.time_column], values=df_['one'], rownames=None, colnames=None, aggfunc=sum, margins=False, margins_name='All', dropna=True, normalize='columns').transpose() of distribution/share of each category in time
      crosstab_=pd.crosstab(index=df_[self.predictor_column], columns=df_[self.time_column], values=df_[target_[0]], rownames=None, colnames=None, aggfunc=sum, margins=False, margins_name='All', dropna=True, normalize='columns').transpose()

      #plot each curve for each category
      for i, color in zip(range(n_categories), colors):
          data_for_plot=crosstab_[categories[i]]
          l, = plt.plot(data_for_plot, color=color, lw=2)
          lines.append(l)
          labels.append(f'{categories[i]}')

      #set plotting parameters
      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.25)
      plt.ticklabel_format(useOffset=False)
      plt.xticks(range(min(crosstab_.index), max(crosstab_.index)+1))
      #plt.xlim(min(_x)-0.1,max(_x)+0.1)
      #plt.ylim(-0.01,1.03)
      plt.xlabel(self.time_column)
      plt.ylabel('Mean of the target')
      plt.title(f'Target association (default rate) of predictor "{self.predictor_column}" in time')
      plt.legend(lines, labels) #, loc=(0, -.38), prop=dict(size=14)
      plt.grid(True)

      plt.show()       

      return self
    

    def get_table(self):
      """
      The idea is to have a table corresponding to the data shown in graph in a following format (or similar):
                              
                      Distribution of 'categorical predictor' in time
      Time             A         B      C       
      ------------------------------------------------------------
      202001          0.23            0.25           ...
      202002          0.33            ...            ...
      202003          0.54            ...            ...
      ------------------------------------------------------------
      All             0.42            ...            ...
      """
      #set up data details
      target_=self.targets[0]
      df_=self.data[self.data[target_[1]]==1]#filtering for only target-observable cases
      df_[self.predictor_column]=df_[self.predictor_column].apply(str)
      df_[self.time_column]=df_[self.time_column].apply(int)
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique categories!'
      assert n_categories<=max_categories_, f'The predictor column specified has more than {max_categories_} unique categories!'

      #produce table of distribution/share of each category in time
      crosstab_=pd.crosstab(index=df_[self.predictor_column], columns=df_[self.time_column], values=df_[target_[0]], rownames=None, colnames=None, aggfunc=sum, margins=True, margins_name='All', dropna=True, normalize='columns').transpose()
      final_table=crosstab_.style.set_table_attributes("style='display:inline'").set_caption(f'Target association (default rate) of predictor "{self.predictor_column}" in time')  
      self.table=final_table
      
      return self