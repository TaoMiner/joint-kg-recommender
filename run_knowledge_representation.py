import sys
import gflags
from jTransUP.models import knowledge_representation
from jTransUP.models.base import get_flags, flag_defaults

FLAGS = gflags.FLAGS

if __name__ == '__main__':
    get_flags()
    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)
    knowledge_representation.run(only_forward=FLAGS.eval_only_mode)