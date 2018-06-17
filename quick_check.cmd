activate dlwin36
python -c "import tensorflow; print('tensorflow: %s, %s' % (tensorflow.__version__, tensorflow.__file__))"
python -c "import cntk; print('cntk: %s, %s' % (cntk.__version__, cntk.__file__))"
python -c "import mxnet; print('mxnet: %s, %s' % (mxnet.__version__, mxnet.__file__))"
python -c "import Keras; print('keras: %s, %s' % (keras.__version__, keras.__file__))"
python -c "import torch; print('pytorch: %s, %s' % (torch.__version__, torch.__file__))"

