import entmatcher.modules.similarity as sim
import entmatcher.modules.score as score
import entmatcher.modules.matching as matching

class BaseMatcher():
    def __init__(self, args):
        self.args = args
        self.sim_strategy = "cosine"
        self.score_strategy = "none"
        self.match_strategy = "greedy"
        # self.features = features
        # self.d = d

    def match(self, features, d):
        # first convert to the similarity matrix
        if len(features) == 1:
            aep_fuse = sim.get(features[0], d.test_lefts, d.test_rights, self.sim_strategy)
        else:
            aep = sim.get(features[0], d.test_lefts, d.test_rights, self.sim_strategy)
            aep_n = sim.get(features[1], d.test_lefts, d.test_rights, self.sim_strategy)
            aep_fuse = 0.5 * aep + 0.5 * aep_n
            del aep_n
            del aep
        aep_fuse = 1 - aep_fuse  # convert to similarity matrix
        aep_fuse = score.optimize(aep_fuse, self.score_strategy, self.args)
        matching.matching(aep_fuse, d, self.match_strategy, self.args)





