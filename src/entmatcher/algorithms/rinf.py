from entmatcher.algorithms.base import BaseMatcher

class RInf(BaseMatcher):
    def __init__(self, args):
        super().__init__(args)
        self.score_strategy = "rinf"




