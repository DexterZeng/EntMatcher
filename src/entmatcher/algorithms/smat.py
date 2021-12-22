from entmatcher.algorithms.base import BaseMatcher

class SMat(BaseMatcher):
    def __init__(self, args):
        super().__init__(args)
        self.match_strategy = "sm"



