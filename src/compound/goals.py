from typing import List, Dict
from .probability import MarkovChain


class CompoundGoalGenerator:
    known_goals: List[str]

    def __init__(self, known_goals: List[str]) -> None:
        self.known_goals = known_goals
        self.chain = MarkovChain(known_goals)

    