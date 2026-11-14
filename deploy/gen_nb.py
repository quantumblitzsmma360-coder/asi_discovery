import nbformat
from nbformat.v4 import new_notebook
import ast

nb = new_notebook()

# Cell 0: Setup
cell0_source = [
    "# DeterministicASI - Mathematically Guaranteed ASI Kernel
",
    "# Seed=137, GPU Required, Z3 Verification
",
    "import os
",
    "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
",
    "os.environ['PYTHONHASHSEED'] = '137'
",
    "import torch
",
    "import torch.nn as nn
",
    "import torch.nn.functional as F
",
    "import numpy as np
",
    "import random
",
    "import math
",
    "import json
",
    "import subprocess
",
    "import sys
",
    "torch.manual_seed(137)
",
    "np.random.seed(137)
",
    "random.seed(137)
",
    "torch.backends.cudnn.deterministic = True
",
    "torch.backends.cudnn.benchmark = False
",
    "assert torch.cuda.is_available(), 'CUDA not available - GPU REQUIRED'
",
    "device = torch.device('cuda')
",
    "print('Device:', device)
",
    "print('GPU:', torch.cuda.get_device_name(0))
",
    "print('Torch:', torch.__version__)
"
]
nb.cells.append(nbformat.v4.new_code_cell(cell0_source))

# Cell 1: Z3
cell1_source = [
    "# Install Z3 for formal verification
",
    "import subprocess
",
    "import sys
",
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'z3-solver', '-q'])
",
    "from z3 import *
",
    "print('Z3 version:', get_version_string())
"
]
nb.cells.append(nbformat.v4.new_code_cell(cell1_source))

print('Cells 0-1 added')
