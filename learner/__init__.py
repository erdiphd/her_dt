from .hgg_DT import HGGLearner_DT
from .normal import NormalLearner
from .hgg import HGGLearner
from .hgg_DT import HGGLearner_DT

learner_collection = {
	'normal': NormalLearner,
	'dt-her': HGGLearner_DT,
	'hgg': HGGLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)