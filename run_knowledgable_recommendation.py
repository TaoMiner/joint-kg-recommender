import sys
import gflags
from jTransUP.models import knowledgable_recommendation
from jTransUP.models.base import get_flags, flag_defaults

FLAGS = gflags.FLAGS

if __name__ == '__main__':
    get_flags()
    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)
    knowledgable_recommendation.run(only_forward=FLAGS.eval_only_mode)