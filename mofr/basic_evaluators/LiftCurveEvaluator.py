import ptdmf.basic_evaluators #calling what is in the init file

class LiftCurveEvaluator(Evaluator):

    def __init__(self, df=None, targets=None, scores=None):
        """
        df: The pandas dataframe containing all the necessary columns.

        targets: These should be the list of binary targets along with their 
        observability flags as follows. [('target1','target1_obs'),('target2', 'target2_obs)]
        Only the first target will be displayed in the graph at the moment.

        scores: List of score columns as follows ['score1', 'score2', 'score3']
        """
        self.df=df
        self.targets=targets
        self.scores=scores

    def d(self, df=None):
          self.df=df
          return df       

    def t(self, targets=None):
          self.targets=targets
          return self   

    def s(self, scores=None):
          self.scores=scores
          return self      

    def get_graph(self):

        # setup plot details
        colors = colors_

        plt.figure(figsize=figsize_)
        lines = []
        labels = []

        n_scores=len(self.scores)
        x_= [(x/10) for x in range (1,11)] #x-axis with different lifts

        #plot each lift curve for each score
        for i, color in zip(range(n_scores), colors[n_scores]):
            target_=self.targets[0]
            score_=self.scores[i]
            df_=self.df[self.df[target_[1]==1]] #filtering for only target-observable cases

            lift_curve = [metrics.liftN(df_[target[0]], df_[score_], x) for x in x_]
            l, = plt.plot(x_, lift_curve, color=color, lw=2)
            lines.append(l)
            labels.append(f'{score_}')

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Lift percentage')
        plt.ylabel('Lift')
        plt.title(f'Lift for target {self.targets[0][0]}')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

        plt.show()
        
        return self

    def get_table(self):
        pass