from entmatcher.algorithms.base import BaseMatcher

class RL(BaseMatcher):
    def __init__(self, args):
        super().__init__(args)
        self.match_strategy = "rl"



