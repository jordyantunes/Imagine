from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from itertools import product


class MarkovChain:
    trials: np.array
    outcomes: np.array
    probabilities: np.array
    class_map : Dict[str, int]
    index_map : Dict[int, str]
    n_classes : int

    def __init__(self, classes:List[str]) -> None:
        self.n_classes = len(classes) + 1
        self.class_map = {
            c : i
            for i, c in
            enumerate(['ini'] + classes)
        }

        self.index_map = {v: k for k, v in self.class_map.items()}

        self.trials = np.zeros((self.n_classes, self.n_classes))
        self.outcomes = np.zeros((self.n_classes, self.n_classes))
        self.probabilities = np.random.rand(self.n_classes, self.n_classes)


        self.probabilities[list(range(self.n_classes)), list(range(self.n_classes))] = 0.0
        self.probabilities[list(range(self.n_classes)), [0 for i in range(self.n_classes)]] = 0.0


    def get_proba(self, class_name:str):
        return self.get_proba_given(class_name, 'ini')

    def get_proba_given(self, class_name0:str, given_class_name:str):
        return self.probabilities[self.class_map[given_class_name], self.class_map[class_name0]]

    def update_outcomes(self, outcomes:List[int], class_name:str, given_class_name:Optional[str]=None):
        given_class_name = given_class_name or 'ini'

        self.trials[self.class_map[given_class_name], self.class_map[class_name]] += len(outcomes)
        self.outcomes[self.class_map[given_class_name], self.class_map[class_name]] += sum(outcomes)
        self.probabilities[self.class_map[given_class_name], self.class_map[class_name]] = (
            self.outcomes[self.class_map[given_class_name], self.class_map[class_name]] /
            self.trials[self.class_map[given_class_name], self.class_map[class_name]]
        )

    def get_n_highest_probabilities(self, n:int, no_ini:bool=True):
        if no_ini:
            flattened = self.probabilities[1:,1:].flatten()
        else:
            flattened = self.probabilities.flatten()

        idx = np.argpartition(flattened, -n)[-n:]
        indices = idx[np.argsort((-flattened)[idx])]

        if no_ini:
            x, y = np.unravel_index(indices, (self.probabilities.shape[0] - 1, self.probabilities.shape[1] - 1))
            return (x + 1, y + 1)
        else:
            x, y = np.unravel_index(indices, self.probabilities.shape)
            return (x, y)


    def get_graph(self, output_file:str='probability_graph.png'):
        G = nx.DiGraph()
        for name, id in self.class_map.items():
            G.add_node(id, label=name)

        edges = product(self.class_map.values(), self.class_map.values())
        edges = np.array([(n1, n2) for n1, n2 in edges if n2 != 0 and n1 != n2])

        n1, n2 = edges[:,0], edges[:,1]
        weight = self.probabilities[n1, n2]

        edges = zip(n1, n2, weight)
        G.add_weighted_edges_from(edges)
        
        pos = nx.spring_layout(G)

        for edge in G.edges(data=True): edge[2]['label'] = "{:.2f}".format(edge[2]['weight'])

        p=nx.drawing.nx_pydot.to_pydot(G)
        p.write_png(output_file)


if __name__ == '__main__':
    chain = MarkovChain(['grow animal', 'grasp fruit', 'turn off light'])

    # tried growing animal 10 times, suceeded 5
    chain.update_outcomes([0,0,0,0,0,1,1,1,1,1], 'grow animal')
    chain.update_outcomes([0,0,1,1,1,1,1,1,1,1], 'grasp fruit')
    chain.update_outcomes([0,1,1,1,1,1,1,1,1,1], 'grow animal', 'grasp fruit')

    print("Probabilidades\n", chain.probabilities)
    highest_probs_indices = chain.get_n_highest_probabilities(3)
    print("Indices", highest_probs_indices)
    print("Highest probs", chain.probabilities[highest_probs_indices])

    pairs = zip(highest_probs_indices[0],highest_probs_indices[1])

    class_pairs = [(
        chain.index_map[x],
        chain.index_map[y]
    ) for x, y in pairs]

    print(class_pairs)
    chain.get_graph()
