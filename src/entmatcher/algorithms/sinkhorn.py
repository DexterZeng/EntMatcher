from entmatcher.algorithms.base import BaseMatcher

class Sinkhorn(BaseMatcher):
    def __init__(self, args):
        super().__init__(args)
        self.score_strategy = "sinkhorn"




