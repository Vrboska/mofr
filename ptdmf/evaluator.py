"""Implementing an abstract base class for our evaluator classes"""

import abc

class Evaluator(abc.ABC):
    """Implementing an abstract base class for our evaluator classes"""

    @abc.abstractmethod
    def get_graph(self):
        """Every evaluator will need to produce a graph; ideally a matplotlib figure"""
        pass

    @abc.abstractmethod
    def get_table(self):
        """Every evaluator will need to produce a table containing numbers in the graph; ideally pd.pivot_table"""
        pass