import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class StabilityInTimeContinuousEvaluator(Evaluator):

    def __init__(self, data=None, predictor_column=None, time_column=None):
      """
      data: The pandas dataframe containing all the necessary columns.

      predictor_column: The name of the column containing the continuous predictor.
      The predictor should be in float format or at least convertible into float.

      time_column: The name of the column containing the time information. This should be an integer or at least convertible to integer
      e.g. 2019 for year, 202110 for month, 20211001 for day etc.
      """
      self.data=data
      self.predictor_column=predictor_column
      self.time_column=time_column 
 
    def d(self, data=None):
      self.data=data
      return self  

    def pc(self, predictor_column=None):
      self.predictor_column=predictor_column
      return self      

    def tc(self, time_column=None):
      self.time_column=time_column
      return self 

    def get_graph(self):

      #percentile functions for the pivot table
      def percentile_10(x):
          return np.percentile(x,10)
      def percentile_25(x):
          return np.percentile(x,25)
      def percentile_50(x):
          return np.percentile(x,50)
      def percentile_75(x):
          return np.percentile(x,75)
      def percentile_90(x):
          return np.percentile(x,90)

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
      df_=self.data
      df_[self.predictor_column]=df_[self.predictor_column].apply(float)
      df_[self.time_column]=df_[self.time_column].apply(int)
      #df_['one']=1
      categories=['percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90']
      n_categories=len(categories)

      #  produce table of distribution/share in time
      pt=pd.pivot_table(df_, values=self.predictor_column, index=self.time_column, columns=None, aggfunc=[percentile_10,percentile_25,percentile_50, percentile_75, percentile_90], fill_value=None, margins=False, dropna=True, margins_name='All')
      pt.columns=[pt.columns[x][0] for x in range(len(pt.columns))]

      #plot each curve for each category
      for i, color in zip(range(n_categories), colors):
          data_for_plot=pt[categories[i]]
          l, = plt.plot(data_for_plot, color=color, lw=2)
          lines.append(l)
          labels.append(f'{categories[i]}')

      #set plotting parameters
      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.25)
      plt.ticklabel_format(useOffset=False)
      plt.xticks(range(min(pt.index), max(pt.index)+1))
      #plt.xlim(min(_x)-0.1,max(_x)+0.1)
      #plt.ylim(-0.01,1.03)
      plt.xlabel(self.time_column)
      plt.ylabel('Percentiles')
      plt.title(f'Distribution of predictor "{self.predictor_column}" in time')
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
      #percentile functions for the pivot table
      def percentile_10(x):
          return np.percentile(x,10)
      def percentile_25(x):
          return np.percentile(x,25)
      def percentile_50(x):
          return np.percentile(x,50)
      def percentile_75(x):
          return np.percentile(x,75)
      def percentile_90(x):
          return np.percentile(x,90)
      
      #set up data details
      df_=self.data
      df_[self.predictor_column]=df_[self.predictor_column].apply(float)
      df_[self.time_column]=df_[self.time_column].apply(int)
      #df_['one']=1
      categories=['percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90']
      n_categories=len(categories)

      #  produce table of distribution/share in time
      pt=pd.pivot_table(df_, values=self.predictor_column, index=self.time_column, columns=None, aggfunc=[percentile_10,percentile_25,percentile_50, percentile_75, percentile_90], fill_value=None, margins=False, dropna=True, margins_name='All')
      pt.columns=[pt.columns[x][0] for x in range(len(pt.columns))]

      #produce table of distribution/share of each category in time
      final_table=pt.style.set_table_attributes("style='display:inline'").set_caption(f'Distribution of predictor "{self.predictor_column}" in time')  
      self.table=final_table
      
      return self