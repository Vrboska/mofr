import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class HistogramContinuousEvaluator(Evaluator):

    def __init__(self, data=None, predictor_column=None):
      """
      data: The pandas dataframe containing all the necessary columns.

      predictor_column: The name of the column containing the categorical predictor.
      There should be no more than 20 unique categories for this perecdictor. Binning should be used
      for predictors with higher number of unique categories. The predictor should be in string format or 
      at least convertible into string.
      """
      self.data=data
      self.predictor_column=predictor_column
 
    def d(self, data=None):
      self.data=data
      return self  

    def pc(self, predictor_column=None):
      self.predictor_column=predictor_column
      return self      


    def get_graph(self):

      # setup plot details
      plt.figure(figsize=figsize_)
      
      #set up data details
      df_=self.data
      df_[self.predictor_column]=df_[self.predictor_column].apply(float)

      #  produce histogram
      n, bins, patches = plt.hist(df_[self.predictor_column], bins='doane', density=False, facecolor='b', alpha=0.75, edgecolor='black')

      plt.xlabel('Values')
      plt.ylabel('Number of observations')
      plt.title(f'Histogram of predictor "{self.predictor_column}"')
      plt.grid(True)

      plt.show()       

      return self
    

    def get_table(self):
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
      df_['']=self.predictor_column
      categories=['percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90']
      n_categories=len(categories)

      #  produce table of distribution/share in time
      pt=pd.pivot_table(df_, values=self.predictor_column, index='', columns=None, aggfunc=[percentile_10,percentile_25,percentile_50, percentile_75, percentile_90], fill_value=None, margins=False, dropna=True, margins_name='All')
      pt.columns=[pt.columns[x][0] for x in range(len(pt.columns))]
      pt=pt.transpose()

      #produce table of distribution/share of each category in time
      final_table=pt.style.set_table_attributes("style='display:inline'").set_caption(f'Percentiles of predictor "{self.predictor_column}"')  
      self.table=final_table
      
      return self