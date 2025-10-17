import sys
import modal

#Step 1: System Base (GPU ready foundation)
#Pulls NVIDIAs official CUDA 12.4 image with ubuntu 22.04 os
#CUDA allows you to utilize a gpu

evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
#System dependencies
    .apt_install(
        [
            "build-essential", #GCC/G++ toolchain
            "cmake",           #build system used by C/C++ projects
            "ninja-build",     # fast build backend used by PyTorch
            "libcudnn8",       # deep learning kernels (GPU ops)
            "libcudnn8-dev",   #also kernel for deep learning
            "git",             #needed for cloning Evo2 repo
            "gcc",             #compiler
            "g++",             #compiler
        ]
    )
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "PIP_EXTRA_INDEX_URL": "https://wheelnext.astral.sh/simple",  #fix
    })
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git "
        "&& cd evo2 && pip install . && pip install vortex"
    )
    .run_commands(
        "pip uninstall -y transformer-engine transformer_engine"
    )
    .run_commands(
        "pip install transformer_engine==2.0.0 --no-build-isolation"
    )
    
    # Remove manual transformer_engine installation
    .pip_install(
        "biopython",
        "torch==2.7.0",
        "fastapi",
        "uvicorn",
        "matplotlib",
        "pandas",
        "seaborn",
        "scikit-learn",
        "openpyxl",
        "modal",
    )
)

app = modal.App("variant-analysis-evo2", image=evo2_image)

volume = modal.Volume.from_name("hf_cache", create_if_missing=True) #here we defined volumes
mount_path = "/root/.cache/huggingface" 


@app.function(gpu="H100", volumes={mount_path: volume}, timeout = 1000) #volumes is a paramater, we want to use gpu and timeout is 1000 seconds aka 15 min until shutdown
def run_brca1_analysis(): #this will be more expensive to run

    import base64
    from io import BytesIO
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import seaborn as sns
    from sklearn.metrics import roc_auc_score

    from evo2 import Evo2

    WINDOW_SIZE = 8192
    #paste in model
    print("Loading evo2 model...")
    model = Evo2('evo2_1b')
    print("Evo2 model loaded")
    #paste in brca1 dataset
    brca1_df = pd.read_excel(
        '/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx', #corresponds to accessing it in folder
        header=2,
    )
    brca1_df = brca1_df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
    ]]
    #rename col names
    brca1_df.rename(columns={
        'chromosome': 'chrom',
        'position (hg19)': 'pos',
        'reference': 'ref',
        'alt': 'alt',
        'function.score.mean': 'score',
        'func.class': 'class',
    }, inplace=True)

    # Convert to two-class system
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')
    #load referende genome
    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break

    # Build mappings of unique reference sequences
    ref_seqs = []
    ref_seq_to_index = {}

# Parse sequences and store indexes
    ref_seq_indexes = []
    var_seqs = []

    brca1_subset = brca1_df.iloc[:500].copy()


    for _, row in brca1_subset.iterrows():
        p = row["pos"]-1 # Convert to 0-indexed position
        full_seq = seq_chr17

        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + \
              row["alt"] + ref_seq[snv_pos_in_ref+1:]

    
        # Get or create index for reference sequence
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)

    ref_seq_indexes = np.array(ref_seq_indexes)

    print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)

    print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)

    # Subtract score of corresponding reference sequences from scores of variant sequences
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]

    # Add delta scores to dataframe
    brca1_subset[f'evo2_delta_score'] = delta_scores

    # Calculate AUROC of zero-shot predictions
    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])
    print("AUROC: " + auroc)

    plt.figure(figsize=(4, 2))

    # Plot stripplot of distributions
    p = sns.stripplot(
        data=brca1_subset,
        x='evo2_delta_score',
        y='class',
        hue='class',
        order=['FUNC/INT', 'LOF'],
        palette=['#777777', 'C3'],
        size=2,
        jitter=0.3,
    )

    # Mark medians from each distribution
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x="evo2_delta_score",
                y="class",
                data=brca1_subset,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {'variants': brca1_subset.to_dict(orient="records"), "plot": plot_data, "auroc": auroc}



@app.function()
def brca1_example():
    print("Running BRCA1 variant anlysis with Evo2...")

    #Run inference
    returns = run_brca1_analysis.remote()

    #Show plot from returned data
    pass


@app.local_entrypoint()
def main():
    brca1_example.local()
