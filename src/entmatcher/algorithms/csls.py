from entmatcher.algorithms.base import BaseMatcher

class CSLS(BaseMatcher):
    def __init__(self, args):
        super().__init__(args)
        self.score_strategy = "csls"




