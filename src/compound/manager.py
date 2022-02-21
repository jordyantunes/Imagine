from .probability import MarkovChain
from typing import List, Dict, Tuple, Union
from copy import copy, deepcopy
import numpy as np


OutcomeList = List[Tuple(str, Union[str,None], List[np.array])]

class ProbabilityManager:
    epochs:List[int]
    current_epoch:int
    epoch_probability:List[MarkovChain]
    goal_descriptions:List[str]
    initial_probabilities:MarkovChain

    def __init__(self, goal_descriptions:List[str]) -> None:
        self.epochs = []
        self.current_epoch = -1

        self.goal_descriptions = copy(goal_descriptions)
        self.initial_probabilities = MarkovChain(self.goal_descriptions)


    def new_epoch(self, outcome_list:OutcomeList):
        # get last probs
        probs = deepcopy(self.epoch_probability[-1])
        self.current_epoch = self.epochs[-1] + 1
        
        # update epochs
        self.epochs.append(self.current_epoch)
        self.epoch_probability.append(probs)

        for class_name, given_class_name, outcomes in outcome_list:
            probs.update_outcomes(outcomes, class_name, given_class_name)

    def get_n_highest_improvement(self, n:int) -> Tuple:
        current_epoch = self.epoch_probability[-1]
        last_epoch = self.epoch_probability[-2]

        chain = MarkovChain(self.goal_descriptions)
        chain.probabilities = current_epoch.probabilities - last_epoch.probabilities
        (x, y) = chain.get_n_highest_probabilities(n)
        
        probabilities = chain.probabilities[(x, y)]

        pairs = zip(x, y)

        class_pairs = [(
            chain.index_map[x],
            chain.index_map[y]
        ) for x, y in pairs]

        return probabilities, class_pairs