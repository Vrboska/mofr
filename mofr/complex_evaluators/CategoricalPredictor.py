import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, rcParams, rcParamsDefault
import matplotlib
from itertools import cycle

import mofr.metrics as metrics
from mofr.evaluator import Evaluator
from mofr.basic_evaluators.settings import big_figsize_,figsize_, colors_, max_categories_
from mofr.basic_evaluators.HistogramCategorical import HistogramCategoricalEvaluator



class CategoricalPredictorEvaluator(Evaluator):

    def __init__(self, data=None, targets=None, predictor_column=None, time_column=None):
      """
      data: The pandas dataframe containing all the necessary columns.

      predictor_column: The name of the column containing the categorical predictor.
      There should be no more than 20 unique categories for this predictor. Binning should be used
      for predictors with higher number of unique categories. The predictor should be in string format or 
      at least convertible into string.

      targets: These should be the list of binary targets along with their 
      observability flags as follows. [('target1','target1_obs'),('target2', 'target2_obs)]
      Only the first target will be displayed in the graph at the moment.

      time_column: The name of the column containing the time information. This should be an integer or at least convertible to integer
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

      rcParams.update({'font.size': 24})
      rcParams.update({'font.weight': 'bold'})
      
      fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=big_figsize_)
      fig.suptitle('Categorical Predictor Evaluation', size=40)  
 

      #set up data details
      target_=self.targets[0]
      df_=self.data
      df_[self.predictor_column]=df_[self.predictor_column].apply(str)
      df_[self.time_column]=df_[self.time_column].apply(int)
      df_['one']=1
      categories=list(df_[self.predictor_column].unique())
      n_categories=len(categories)  

      #assert the correct number of categories
      assert n_categories>=2,  'The predictor column specified has less than 2 unique categories!'
      assert n_categories<=max_categories_, f'The predictor column specified has more than {max_categories_} unique categories!'

      # Histogram part
      #  produce table of distribution/share of each category in time
      table=df_['categorical_predictor'].value_counts(dropna=False, normalize=True)


      #set plotting parameters
      ax1.bar(table.index,table.values, color=colors_)

      plt.xlabel('Categories', axes=ax1)
      plt.ylabel('Share of the given category', axes=ax1)
      ax1.set_title(f'Distribution of predictor "{self.predictor_column}"')
      ax1.grid(True)      

      # Stability in time part
      # produce table of distribution/share of each category in time
      crosstab_=pd.crosstab(index=df_[self.predictor_column], columns=df_[self.time_column], values=df_['one'], rownames=None, colnames=None, aggfunc=sum, margins=False, margins_name='All', dropna=True, normalize='columns').transpose()
      colors = cycle(colors_)
      lines = []
      labels = []  

      #plot each curve for each category
      for i, color in zip(range(n_categories), colors):
          data_for_plot=crosstab_[categories[i]]
          l, = ax2.plot(data_for_plot, color=color, lw=2)
          lines.append(l)
          labels.append(f'{categories[i]}')

      #set plotting parameters
      fig = plt.gcf()
      fig.subplots_adjust(bottom=0.25)
      ax2.ticklabel_format(useOffset=False)
      ax2.set_xticks(range(min(crosstab_.index), max(crosstab_.index)+1))
      #plt.xlim(min(_x)-0.1,max(_x)+0.1)
      #plt.ylim(-0.01,1.03)
      ax2.set_xlabel(self.time_column)
      ax2.set_ylabel('Share of the given category')
      ax2.set_title(f'Distribution of predictor "{self.predictor_column}" in time')
      ax2.legend(lines, labels) #, loc=(0, -.38), prop=dict(size=14)
      ax2.grid(True)

      # Target Association part
      #  produce tablepd.crosstab(index=df_[self.predictor_column], columns=df_[self.time_column], values=df_['one'], rownames=None, colnames=None, aggfunc=sum, margins=False, margins_name='All', dropna=True, normalize='columns').transpose() of distribution/share of each category in time
      crosstab_=pd.crosstab(index=df_[self.predictor_column], columns=df_[self.time_column], values=df_[target_[0]], rownames=None, colnames=None, aggfunc=sum, margins=False, margins_name='All', dropna=True, normalize='columns').transpose()
      colors = cycle(colors_)
      lines = []
      labels = []  

      #plot each curve for each category
      for i, color in zip(range(n_categories), colors):
          data_for_plot=crosstab_[categories[i]]
          l, = ax3.plot(data_for_plot, color=color, lw=2)
          lines.append(l)
          labels.append(f'{categories[i]}')

      #set plotting parameters
      ax3.ticklabel_format(useOffset=False)
      ax3.set_xticks(range(min(crosstab_.index), max(crosstab_.index)+1))
      #plt.xlim(min(_x)-0.1,max(_x)+0.1)
      #plt.ylim(-0.01,1.03)
      ax3.set_xlabel(self.time_column)
      ax3.set_ylabel('Mean of the target')
      ax3.set_title(f'Target association (default rate) of predictor "{self.predictor_column}" in time')
      ax3.legend(lines, labels) #, loc=(0, -.38), prop=dict(size=14)
      ax3.grid(True)      

      plt.show()
      plt.close()
      
      rcParams.update(rcParamsDefault)


      return self
    

    def get_table(self):

      # Histogram part 
      from mofr.basic_evaluators.HistogramCategorical import HistogramCategoricalEvaluator
      hcae=HistogramCategoricalEvaluator()
      hcae.d(self.data).pc(self.predictor_column)
      hcae.get_table()
      display(hcae.table)

      # Stability in time part
      from mofr.basic_evaluators.StabilityInTimeCategorical import StabilityInTimeCategoricalEvaluator
      sitcae=StabilityInTimeCategoricalEvaluator()
      sitcae.d(self.data).pc(self.predictor_column).tc(self.time_column)
      sitcae.get_table()
      display(sitcae.table)

      # Target Association part
      from mofr.basic_evaluators.TargetAssociationCategorical import TargetAssociationCategoricalEvaluator
      tacae=TargetAssociationCategoricalEvaluator()
      tacae.d(self.data).t([(self.targets[0][0], self.targets[0][1])]).pc(self.predictor_column).tc(self.time_column)
      tacae.get_table()
      display(tacae.table)

      return self