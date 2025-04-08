# 2025-04-llm-collective
Finetuning a Llama model with PubMed data

# Installation

```
On della-gpu:
module load anaconda3/2024.10
conda create -n 202504_llm pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate 202504_llm
conda install sentencepiece -c conda-forge
pip install accelerate torchao torchtune transformers wandb
```

If you run into errors with the above, try:

```
conda install -c conda-forge libstdcxx-ng
```

# Data Gathering

nomic seemed to conflict with PyTorch, so I made a separate environment for it:

```
module load anaconda3/2024.10
conda create -n 202504_nomic nomic pandas -c conda-forge
conda activate nomic
```

Then this script was enough to grab a subset of the data:

```
from nomic import AtlasDataset
import pandas as pd

dataset = AtlasDataset('aaron/pubmed-full')
table = dataset.maps[0].data.tb
subset_table = table.slice(0, 50000)

subset_data = [
    {col: row[i] for i, col in enumerate(subset_table.column_names)}
     for row in zip(
        *[subset_table[col].to_pylist() for col in subset_table.column_names]
        )
]

df = pd.DataFrame(subset_data)
df.to_csv("/home/niznik/scratch/pubmed_subset_50k.csv", index=False)
```

# Data Cleaning

See format_data.py for conversion to jsonl format

# Finetuning

See finetune.yaml for the proper settings

# Manual Testing

See query.py for loading the model and sending "tests"