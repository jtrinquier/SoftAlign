# SoftAlign: End-to-End Structural Alignment for Protein Data

**UPDATE 12/01/2026: The colab notebooks have been updated to work with the new jax version.**

SoftAlign is an advanced alignment method designed to efficiently compare 3D protein structures. By leveraging structural information directly, SoftAlign provides an end-to-end alignment process, allowing for both highly accurate alignments and efficient computations. The method uses the 3D coordinates of protein pairs, transforming them into feature vectors through a retrained encoder of ProteinMPNN. This similarity matrix is then aligned using two strategies: a differentiable Smith-Waterman method and a novel softmax-based pseudo-alignment approach.

Our results demonstrate that SoftAlign is able to recapitulate TM-align results while being faster and more accurate than alternative tools like Foldseek. While not the fastest alignment method available, SoftAlign excels in precision and is well-suited for integration with other pre-filtering methods. Notably, the softmax-based alignment shows superior sensitivity for structure similarity detection compared to traditional methods.

SoftAlign also introduces a novel pseudo-alignment method based on softmax. This approach can be integrated into other models and architectures, even those not inherently focused on structural information. For a more detailed description of the method, please refer to the full paper [here](https://www.biorxiv.org/content/10.1101/2025.05.09.653096v1).

---

## Table of Contents

1.  [Introduction](#softalign-end-to-end-structural-alignment-for-protein-data)
2.  [Google Colab Notebooks](#-google-colab-notebooks)
3.  [Local Command-Line Scripts](#local-command-line-scripts)
    * [Environment Setup](#1-environment-setup)
    * [Script Usage](#2-script-usage)
        * [`align.py`](#a-alignpy-pairwise-protein-alignment)
        * [`structure_search.py`](#b-structure_searchpy-protein-structure-search)
    * [Troubleshooting and Notes](#3-troubleshooting-and-notes)


---

## 🔬 Google Colab Notebooks

To facilitate ease of use and reproducibility, we provide three Google Colab notebooks:

1.  **Inference Notebook**:
    * Loads AlphaFold or PDB structures by ID, or accepts custom .pdb files.
    * Runs SoftAlign to generate and visualize the alignment.
    * Computes TM-score and LDDT scores for quantitative comparison.
    * Lets you adjust the temperature to control alignment softness.
    * Includes an optional softmax mode to explore the pseudo-alignment method.

    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtrinquier/SoftAlign/blob/main/Colab/COLAB_SoftAlign.ipynb)
    [COLAB_SoftAlign.ipynb](https://colab.research.google.com/github/jtrinquier/SoftAlign/blob/main/Colab/COLAB_SoftAlign.ipynb)

2.  **Training Notebook**: Reproduces the training process with the same train-test split as described in our paper.
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtrinquier/SoftAlign/blob/main/Colab/SoftAlign_training.ipynb)
    [SoftAlign_training.ipynb](https://colab.research.google.com/github/jtrinquier/SoftAlign/blob/main/Colab/SoftAlign_training.ipynb)

3.  **All-vs-All Search Notebook**: Performs an all-vs-all search within the SCOPE 40 dataset. You can also input your own PDB folder and perform an all-vs-all search.
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtrinquier/SoftAlign/blob/main/Colab/Structure_Search_SoftAlign.ipynb)
    [Structure_Search_SoftAlign.ipynb](https://colab.research.google.com/github/jtrinquier/SoftAlign/blob/main/Colab/Structure_Search_SoftAlign.ipynb)

### Local Command-Line Scripts

For users who prefer local execution or require command-line automation, two dedicated Python scripts (`align.py` and `structure_search.py`) are provided. These scripts adapt functionalities from the original Colab notebooks, enabling pairwise protein alignment and structural database searches directly from your terminal.

#### 1. Environment Setup

To run these scripts locally, follow these steps:

* **Clone the SoftAlign Repository:**
    Navigate to your desired project directory and clone the official SoftAlign repository. This contains the core model code and pre-trained weights.
    ```bash
    git clone https://github.com/jtrinquier/SoftAlign.git
    cd SoftAlign
    ```

* **Create and Activate Conda Environment:**
    Create a new Conda environment with Python 3.10 (or a newer compatible version like 3.11/3.12).
    ```bash
    conda create -n softalign_env python=3.10
    conda activate softalign_env
    ```

* **Install Dependencies:**
    With your `softalign_env` active, install the necessary Python packages:
    * **Core Packages:**
        ```bash
        conda install numpy pandas matplotlib biopython
        ```
    * **JAX (Choose one based on your hardware):**
        * **For CPU:**
            ```bash
            pip install jax[cpu]
            ```
        * **For CUDA (NVIDIA GPU users - IMPORTANT: Check your CUDA Version with `nvidia-smi`):**
            Choose the `jaxlib` version compatible with your CUDA Toolkit. For example, if `nvidia-smi` shows CUDA 12.x:
            ```bash
            pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
            ```
    * **dm-haiku :**
        ```bash
        pip install git+https://github.com/deepmind/dm-haiku
        ```
    * **gdown (for downloading SCOPE database):**
        ```bash
        pip install gdown
        ```



#### 2. Script Usage

Activate your `softalign_env` (`conda activate softalign_env`) before running these commands.

##### a. `align.py` (Pairwise Protein Alignment)

This script performs soft alignment between two proteins, calculates LDDT scores, and visualizes/saves the alignment and similarity matrices.

* **Arguments:**
    * `--pdb_source`: (`af` (default) | `pdb` | `custom`) Source of PDB files.
    * `--pdb1_id`: Identifier for the first protein.
    * `--pdb2_id`: Identifier for the second protein.
    * `--temperature`: (Optional, float) Alignment temperature (default: `1e-4`).
    * `--output_dir`: (Optional, str) Directory to save plots and `.npy` alignment matrix (default: `./softalign_output`).

* **Example: Aligning two proteins from AlphaFold DB:**
    ```bash
    python align.py --pdb1_id Q5VSL9 --pdb2_id A0A7L4L2T3 --output_dir ./pairwise_alignments
    ```
    *(For custom PDB files, ensure they are in the current directory and use `--pdb_source custom --pdb1_id my_protein_A.pdb`)*

##### b. `structure_search.py` (Protein Structure Search)

This script allows for structural searches (one-vs-all or all-vs-all) against a database, and optionally plots a specific pairwise alignment from the loaded dataset.

* **Data Source Arguments:**
    * `--use_scope_database`: (Flag) Uses a precomputed SCOPE database.
    * `--pdb_folder_path`: (str) Path to your custom PDB folder (required if `--use_scope_database` is not set).
    * `--chain_ids_file`: (Optional, str) Path to a CSV file (e.g., `id,chain`) for custom PDBs.

* **Model Arguments:**
    * `--model_type`: (`Softmax` (default) | `Smith-Waterman`) Type of SoftAlign model to use.

* **Search Arguments:**
    * `--query_id`: (str) ID for a "one-vs-all" search (e.g., `d2dixa1` for SCOPE). Leave empty to skip.
    * `--run_all_vs_all`: (Flag) Performs a full all-vs-all search across the dataset.

* **Optional Pairwise Plotting Arguments (from loaded dataset):**
    * `--pdb1_to_plot_id`: (str) First PDB ID for plot.
    * `--pdb2_to_plot_id`: (str) Second PDB ID for plot.
    * `--temperature`: (Optional, float) Alignment temperature for this specific plot.

* **Output Arguments:**
    * `--output_dir`: (Optional, str) Directory to save score CSVs, plots, and `.npy` files (default: `./softalign_search_output`).

* **Example 1: One-vs-all search with SCOPE database and specific pairwise plot:**
    ```bash
    python structure_search.py \
        --use_scope_database \
        --model_type Softmax \
        --query_id d2dixa1 \
        --pdb1_to_plot_id d2dixa1 \
        --pdb2_to_plot_id d1qrjb1 \
        --output_dir ./search_and_plot_results
    ```
   

* **Example 2: All-vs-all search using a custom PDB folder:**
    ```bash
    # Assuming 'my_pdbs' folder exists with .pdb files
    python structure_search.py \
        --pdb_folder_path ./my_pdbs \
        --model_type Smith-Waterman \
        --run_all_vs_all \
        --output_dir ./my_custom_search_results
    ```
## 3. Troubleshooting and Notes

###  JAX Version Compatibility

The latest releases of **JAX** and **jaxlib** introduce changes that may cause errors when running SoftAlign with the default installation commands. If you encounter issues related to JAX, Haiku, it is likely due to using a JAX version that is too new. We will solve this compatibility issue in a future update.

SoftAlign has been tested and confirmed to work with the following versions:

```bash
jax==0.4.28
jaxlib==0.4.28+cuda12.cudnn89
dm-haiku==0.0.15.dev0


