import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import figsize_, colors_, max_categories_


class TargetAssociationContinuousEvaluator(Evaluator):

    def __init__(self, data=None, targets=None, predictor_column=None, time_column=None):
      """
      data: The pandas dataframe containing all the necessary columns.

      targets: These should be the list of binary targets along with their 
      observability flags as follows. [('target1','target1_obs'),('target2', 'target2_obs)]
      Only the first target will be displayed in the graph at the moment.

      predictor_column: The name of the column containing the continuous predictor.
      The predictor should be in float format or at least convertible into float.

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

    def get_graph(self, plot=True):

      # setup plot details
      colors = cycle(colors_)

      f, ax = plt.subplots(figsize=figsize_)

      """
      The idea is to have a graph of predictor values on the x-axis and the log odds with respect to target variable
      on the y-axis. One line for each month/time_unit. The steeper the lines the stronger the predictor. The assumption of the 
      logistic regression model is that these lines should be straight lines (linear shape).
      """
      lines = []
      labels = []
      
      #set up data details
      target_=self.targets[0]
      df_=self.data[self.data[target_[1]]==1]#filtering for only target-observable cases
      df_[self.predictor_column]=df_[self.predictor_column].apply(float)
      df_[self.time_column]=df_[self.time_column].apply(int)
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique values!'

      #binning the numerical predictor into 5 intervals of equal length (not equal number of observations)
      df_[self.predictor_column+'_binned']=pd.cut(df_[self.predictor_column], bins=5)

      #auxilliary function for calculating logodds
      def logodds_(x):
        mean_=np.mean(x)
        if mean_==1.0:
            return -10.0
        if mean_==0.0:
            return 10.0
        else:
            return np.log(mean_/(1-mean_))

      #  produce table 
      crosstab_=pd.crosstab(index=df_[self.predictor_column+'_binned'], columns=df_[self.time_column], values=df_[target_[0]], rownames=None, colnames=None, aggfunc=logodds_, margins=True, margins_name='All', dropna=True, normalize=False).iloc[:-1]

      #plot each curve for each category
      for i, color in zip(range(len(crosstab_.columns)), colors):
          data_for_plot=crosstab_[crosstab_.columns[i]]
          l, = plt.plot(range(len(data_for_plot.index)), data_for_plot.values, color=color, lw=2, axes=ax)
          lines.append(l)
          labels.append(f'{crosstab_.columns[i]}')

      #set plotting parameters
      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.25)
      plt.ticklabel_format(useOffset=False)
      plt.xticks(range(len(data_for_plot.index)),crosstab_.index, axes=ax)
      #plt.xlim(min(_x)-0.1,max(_x)+0.1)
      #plt.ylim(-0.01,1.03)
      plt.xlabel(self.predictor_column+'_binned', axes=ax)
      plt.ylabel('Logodds of the target variable', axes=ax)
      plt.title(f'Logodds of the target variable "{target_[0]}" vs. the predictor "{self.predictor_column}" values', axes=ax)
      ax.legend(lines, labels) #, loc=(0, -.38), prop=dict(size=14)
      ax.grid(True)

      if plot==True:
        plt.show()  

      self.graph=f
      self.axis=ax

      plt.close()        

      return self
    

    def get_table(self):
      #set up data details
      target_=self.targets[0]
      df_=self.data[self.data[target_[1]]==1]#filtering for only target-observable cases
      df_[self.predictor_column]=df_[self.predictor_column].apply(float)
      df_[self.time_column]=df_[self.time_column].apply(int)
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique values!'

      #binning the numerical predictor into 5 intervals of equal length (not equal number of observations)
      df_[self.predictor_column+'_binned']=pd.cut(df_[self.predictor_column], bins=5)

      #auxilliary function for calculating logodds
      def logodds_(x):
        mean_=np.mean(x)
        if mean_==1.0:
            return -10.0
        if mean_==0.0:
            return 10.0
        else:
            return np.log(mean_/(1-mean_))

      #  produce table 
      crosstab_=pd.crosstab(index=df_[self.predictor_column+'_binned'], columns=df_[self.time_column], values=df_[target_[0]], rownames=None, colnames=None, aggfunc=['count','sum','mean',logodds_], margins=True, margins_name='All', dropna=True, normalize=False)
      final_table=crosstab_.style.set_table_attributes("style='display:inline'").set_caption(f'Different aggregations of target variable "{target_[0]}" vs. the predictor "{self.predictor_column}" values')  
      self.table=final_table
      
      return self