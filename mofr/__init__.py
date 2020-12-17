"""
ptdmf - a shortcut for Prague Tech Data Modelling Framework
============================================================

The general idea for Modelling Framework is to have a comprehensive and unified python framework
for creating and evaluating predictive models within our team. 

This will be achieved by programming the simplest outputs (or, as I call them, basic evaluators) 
first and then combining these simplest outputs into more sophisticated ones (complex evaluators). 
Every evaluator will inherit from Evaluator abstract base class and will have methods get_graph 
and get_table.

For basic evaluators, each  will be a basic output unit related to predictive modelling tasks
e.g. StabilityInTimeEvaluator, ROCcurveEvaluator, LiftEvaluator etc.
We can think of basic evaluators as sth. that produces a single graph and/or single table.

Complex evaluators such as ContinuousPredictorEvaluator or ScoreComparisonEvaluator will
essentially combine 3-4 basic evaluators into one useful output containing 3-4 
graphs/tables that will allow the user to gain complex insights in one coherent output.


Additionally, there will be additional packages and modules added depending on the team's needs. 

"""