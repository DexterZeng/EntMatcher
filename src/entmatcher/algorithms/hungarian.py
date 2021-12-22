from entmatcher.algorithms.base import BaseMatcher

class Hun(BaseMatcher):
    def __init__(self, args):
        super().__init__(args)
        self.match_strategy = "hun"



