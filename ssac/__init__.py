import logging as p_logging
import os
import warnings

if "LOG" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    p_logging.getLogger().setLevel("ERROR")
    warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
#THIS CODE SETS UP THE LOGGING FOR THE PYTHON RUN 
#(MAYBE MORE LOGGING SETTINGS ARE SET IN OTHER PARTS OF THE CODE)