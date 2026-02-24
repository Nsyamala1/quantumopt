"""One-time script to update Cell 1 in train_gnn.ipynb with pre-built wheels install."""
import json

with open("train_gnn.ipynb") as f:
    nb = json.load(f)

new_source = [
    "# ╔══════════════════════════════════════════════════════════════════╗\n",
    "# ║  CELL 1 — Imports and Setup                                     ║\n",
    "# ╚══════════════════════════════════════════════════════════════════╝\n",
    "\n",
    "import torch\n",
    "import os\n",
    "\n",
    "# Get torch version for correct wheel URL\n",
    "torch_version = torch.__version__.split('+')[0]\n",
    "cuda_version = 'cu121'  # Kaggle T4 uses CUDA 12.1\n",
    "\n",
    "print(f'PyTorch: {torch_version}, CUDA: {cuda_version}')\n",
    "\n",
    "# Install using pre-built wheels — much faster than building from source\n",
    "os.system(f'pip install torch-geometric -q')\n",
    "os.system(f'pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html -q')\n",
    "os.system('pip install qiskit -q')\n",
    "\n",
    "print('Installation complete!')\n",
    "\n",
    "import json\n",
    "import time\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from torch_geometric.data import Data\n",
    "from torch_geometric.loader import DataLoader\n",
    "from torch_geometric.nn import GCNConv, GATConv, global_mean_pool\n",
    "from torch.nn import Linear, ReLU, Dropout, MSELoss, ELU\n",
    "from torch.optim import Adam\n",
    "from torch.optim.lr_scheduler import ReduceLROnPlateau\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# ── GPU Setup ──────────────────────────────────────────────────────\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "if torch.cuda.device_count() > 1:\n",
    "    print(f'🚀 Using {torch.cuda.device_count()} GPUs!')\n",
    "    multi_gpu = True\n",
    "else:\n",
    "    print(f'Using: {device}')\n",
    "    multi_gpu = False\n",
    "\n",
    "# Print GPU info\n",
    "if torch.cuda.is_available():\n",
    "    for i in range(torch.cuda.device_count()):\n",
    "        props = torch.cuda.get_device_properties(i)\n",
    "        print(f'  GPU {i}: {props.name} — {props.total_mem / 1024**3:.1f} GB')\n",
    "else:\n",
    "    print('  ⚠️  No GPU detected, training will be slow')\n",
    "\n",
    "print(f'\\nPyTorch: {torch.__version__}')\n",
    "print('✅ All imports successful')",
]

# Split into lines for notebook format
lines = []
for line in new_source:
    lines.append(line)

# Cell 1 is index 1 (index 0 is the markdown header)
nb["cells"][1]["source"] = lines

with open("train_gnn.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("✅ Cell 1 updated with pre-built wheels installation")
