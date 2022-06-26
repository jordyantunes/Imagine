from .probability import MarkovChain
from typing import List, Dict, Tuple, Union
from copy import copy, deepcopy
import numpy as np
from mpi4py import MPI


OutcomeList = List[Tuple[str, Union[str,None], List[np.array]]]

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
        self.epoch_probability = [self.initial_probabilities]
        self.rank = MPI.COMM_WORLD.Get_rank()


    def new_epoch(self, outcome_list:OutcomeList):
        # get last probs
        probs = deepcopy(self.epoch_probability[-1])
        self.current_epoch = 1 if len(self.epochs) == 0 else self.epochs[-1] + 1
        
        all_outcome_list = MPI.COMM_WORLD.gather(outcome_list, root=0)

        if self.rank == 0:
            print("All outcomes:", all_outcome_list)
            single_all_outcome_list = []
            for i in range(len(all_outcome_list)):
                single_all_outcome_list += all_outcome_list[i]

            # update epochs
            self.epochs.append(self.current_epoch)

            for class_name, given_class_name, outcomes in single_all_outcome_list:
                probs.update_outcomes([outcomes], class_name, given_class_name)
            
            self.epoch_probability.append(probs)

        if len(self.epochs) > 2:
            print("Removing previous epochs from history")
            del self.epochs[:-2]
            del self.epoch_probability[:-2]

        self.current_epoch = MPI.COMM_WORLD.bcast(self.current_epoch, root=0)
        self.epochs = MPI.COMM_WORLD.bcast(self.epochs, root=0)
        self.epoch_probability = MPI.COMM_WORLD.bcast(self.epoch_probability, root=0)

    def get_n_highest_improvement(self, n:int) -> Tuple:
        current_epoch = self.epoch_probability[-1]

        if len(self.epoch_probability) > 1:
            last_epoch = self.epoch_probability[-2]

            chain = MarkovChain(self.goal_descriptions)
            chain.probabilities = current_epoch.probabilities - last_epoch.probabilities
        else:
            chain = current_epoch
        (x, y) = chain.get_n_highest_probabilities(n)
        
        probabilities = chain.probabilities[(x, y)]

        pairs = zip(x, y)

        class_pairs = [(
            chain.index_map[x],
            chain.index_map[y]
        ) for x, y in pairs]

        return probabilities, class_pairs