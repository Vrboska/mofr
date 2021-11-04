import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_


class HistogramCategoricalEvaluator(Evaluator):

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
      df_[self.predictor_column]=df_[self.predictor_column].apply(str)
      df_['one']=1
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique categories!'
      assert n_categories<=20, 'The predictor column specified has more than 20 unique categories!'

      #  produce table of distribution/share of each category in time
      table=df_['categorical_predictor'].value_counts(dropna=False, normalize=True)


      #set plotting parameters
      plt.bar(table.index,table.values, color=colors_)

      plt.xlabel('Categories')
      plt.ylabel('Share of the given category')
      plt.title(f'Distribution of predictor "{self.predictor_column}"')
      plt.grid(True)

      plt.show()       

      return self
    

    def get_table(self):

      #set up data details
      df_=self.data
      df_[self.predictor_column]=df_[self.predictor_column].apply(str)
      df_['one']=1
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique categories!'
      assert n_categories<=20, 'The predictor column specified has more than 20 unique categories!'

      #  produce table of distribution/share of each category in time
      table1=df_['categorical_predictor'].value_counts(dropna=False, normalize=True)
      table2=df_['categorical_predictor'].value_counts(dropna=False, normalize=False)

      final_table=pd.concat([table1, table2], axis=1)
      final_table.columns=[final_table.columns[0]+' %', final_table.columns[1]]
      self.table=final_table
      
      return self