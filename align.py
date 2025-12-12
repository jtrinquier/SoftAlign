import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import argparse
import matplotlib.pyplot as plt # Import for plotting

# Add SoftAlign directory to sys.path
softalign_path = os.path.join(os.getcwd(), '') # Assuming you are in the SoftAlign directory
if softalign_path not in sys.path:
    sys.path.append(softalign_path)

softalign_code_path = os.path.join(softalign_path, 'softalign')
if softalign_code_path not in sys.path:
    sys.path.append(softalign_code_path)

# Import SoftAlign modules
import Input_MPNN as input_
import END_TO_END_MODELS as ete
import Score_align as lddt # Score_align is imported as lddt for LDDT calculation

def main():
    parser = argparse.ArgumentParser(description="Run SoftAlign for protein structural alignment and LDDT scoring without TM-align.")
    parser.add_argument("--pdb_source", type=str, default="af", choices=["af", "pdb", "custom"],
                        help="Source of PDB files: 'af' for AlphaFold, 'pdb' for RCSB PDB, 'custom' for local files.")
    parser.add_argument("--pdb1_id", type=str, default="Q5VSL9",
                        help="Identifier for the first protein (UniProt ID for AlphaFold, PDB ID for RCSB PDB, or filename for custom).")
    parser.add_argument("--pdb2_id", type=str, default="A0A7L4L2T3",
                        help="Identifier for the second protein (UniProt ID for AlphaFold, PDB ID for RCSB PDB, or filename for custom).")
    parser.add_argument("--temperature", type=float, default=1e-4,
                        help="Temperature parameter for soft alignment.")
    parser.add_argument("--model_path", type=str, default="./models/CONT_SW_05_T_3_1",
                        help="Path to the pre-trained SoftAlign model parameters.")
    parser.add_argument("--output_dir", type=str, default="./softalign_output",
                        help="Directory to save output files (plots and alignment matrix).")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    pdb1_file = f"{args.pdb1_id}.pdb"
    pdb2_file = f"{args.pdb2_id}.pdb"

    # --- 1. Define Protein Structures ---
    if args.pdb_source == "af":
        print(f"Downloading {args.pdb1_id} and {args.pdb2_id} from AlphaFold...")
        os.system(f"wget -qO {pdb1_file} https://alphafold.ebi.ac.uk/files/AF-{args.pdb1_id}-F1-model_v6.pdb")
        os.system(f"wget -qO {pdb2_file} https://alphafold.ebi.ac.uk/files/AF-{args.pdb2_id}-F1-model_v6.pdb")
    elif args.pdb_source == "pdb":
        print(f"Downloading {args.pdb1_id} and {args.pdb2_id} from RCSB PDB...")
        os.system(f"wget -qO {pdb1_file} https://files.rcsb.org/view/{args.pdb1_id}.pdb")
        os.system(f"wget -qO {pdb2_file} https://files.rcsb.org/view/{args.pdb2_id}.pdb")
    elif args.pdb_source == "custom":
        print(f"Using custom files: {args.pdb1_id} and {args.pdb2_id}. Ensure they are in the current directory.")
        pdb1_file = args.pdb1_id # Use the ID directly as filename for custom
        pdb2_file = args.pdb2_id
        if not os.path.exists(pdb1_file) or not os.path.exists(pdb2_file):
            print(f"Error: One or both custom PDB files ({pdb1_file}, {pdb2_file}) not found.")
            sys.exit(1)
    else:
        print("Invalid PDB source. Please choose from 'af', 'pdb', or 'custom'.")
        sys.exit(1)

    # --- 2. Load Protein Coordinates ---
    try:
        X1, mask1, chain1, res1 = input_.get_inputs_mpnn(pdb1_file, chain='A')
        X2, mask2, chain2, res2 = input_.get_inputs_mpnn(pdb2_file, chain='A')
    except Exception as e:
        print(f"Error loading PDB files: {e}")
        print("Please ensure the PDB files are valid and correctly downloaded/provided.")
        sys.exit(1)

    # Prepare inputs for the model
    x1_coords = X1[:, :, 1] # Assuming 1 is the index for C-alpha coordinates
    x2_coords = X2[:, :, 1]
    lens = jnp.array([X1.shape[1], X2.shape[1]])[None, :]

    # --- 3. Load SoftAlign Model ---
    num_layers = 3
    num_neighbors = 64
    encoding_dim = 64
    affine = True
    soft_max = False # Set to False for standard model weights

    def model_end_to_end(x1, x2, lens, t):
        model = ete.END_TO_END(
            encoding_dim, encoding_dim, encoding_dim, num_layers,
            num_neighbors, affine=affine, soft_max=soft_max,
            dropout=0., augment_eps=0.0)
        return model(x1, x2, lens, t)

    MODEL_ETE = hk.transform(model_end_to_end)

    key = jax.random.PRNGKey(0)

    # Load model parameters
    try:
        with open(args.model_path, "rb") as f:
            params = pickle.load(f)
        print(f"✅ Loaded model params from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model parameters file not found at {args.model_path}.")
        print("Please ensure you have cloned the SoftAlign repository and downloaded the model weights into the 'models' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading model parameters: {e}")
        sys.exit(1)

    # --- 4. Compute Soft Alignment ---
    print(f"\nTemperature set to {args.temperature:.1e}")
    soft_aln, sim_matrix, score = MODEL_ETE.apply(params, key, (X1, mask1, chain1, res1), (X2, mask2, chain2, res2), lens, args.temperature)

    # --- 5. Calculate LDDT Scores ---
    # Create dummy masks for LDDT calculation (as seen in the Colab notebook)
    mask_1 = jnp.ones((1, X1.shape[1], X1.shape[1]))
    mask_2 = jnp.ones((1, X2.shape[1], X2.shape[1]))

    # LDDT scores for SoftAlign model
    lddt_soft_1 = lddt.get_LDDTloss(X1[:, :, 1], X2[:, :, 1], soft_aln, mask_1, mask_2, args.temperature)
    lddt_soft_2 = lddt.get_LDDTloss(X2[:, :, 1], X1[:, :, 1], soft_aln.transpose(0, 2, 1), mask_2, mask_1, args.temperature)

    print(f"\n📎 Comparing alignments between **{args.pdb1_id}** and **{args.pdb2_id}**\n")
    print(f"🧠 SoftAlign model:")
    print(f"  • LDDT {args.pdb1_id}→{args.pdb2_id}: {lddt_soft_1[0]:.4f}")
    print(f"  • LDDT {args.pdb2_id}→{args.pdb1_id}: {lddt_soft_2[0]:.4f}")


    # --- 6. Visualize and Save Outputs ---

    # Convert JAX arrays to NumPy for plotting and saving
    soft_aln_np = np.asarray(soft_aln[0]) # Assuming batch size 1
    sim_matrix_np = np.asarray(sim_matrix[0]) # Assuming batch size 1

    # Plotting Alignment Matrix
    plt.figure(figsize=(8, 7))
    plt.imshow(soft_aln_np, cmap='viridis', origin='lower',
               extent=[0, X2.shape[1], 0, X1.shape[1]])
    plt.colorbar(label='Alignment Score')
    plt.xlabel(f'{args.pdb2_id} Residue Index')
    plt.ylabel(f'{args.pdb1_id} Residue Index')
    plt.title(f'Soft Alignment Matrix ({args.pdb1_id} vs {args.pdb2_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'soft_alignment_matrix_{args.pdb1_id}_vs_{args.pdb2_id}.png'))
    print(f"✅ Saved alignment matrix plot to {os.path.join(args.output_dir, f'soft_alignment_matrix_{args.pdb1_id}_vs_{args.pdb2_id}.png')}")
    # plt.show() # Uncomment if you want to display the plot directly

    # Plotting Similarity Matrix
    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix_np, cmap='viridis', origin='lower',
               extent=[0, X2.shape[1], 0, X1.shape[1]])
    plt.colorbar(label='Similarity Score')
    plt.xlabel(f'{args.pdb2_id} Residue Index')
    plt.ylabel(f'{args.pdb1_id} Residue Index')
    plt.title(f'Similarity Matrix ({args.pdb1_id} vs {args.pdb2_id})')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'similarity_matrix_{args.pdb1_id}_vs_{args.pdb2_id}.png'))
    print(f"✅ Saved similarity matrix plot to {os.path.join(args.output_dir, f'similarity_matrix_{args.pdb1_id}_vs_{args.pdb2_id}.png')}")
    # plt.show() # Uncomment if you want to display the plot directly


    # Save Alignment Matrix (soft_aln) with PDB names
    alignment_filename = os.path.join(args.output_dir, f'soft_alignment_{args.pdb1_id}_vs_{args.pdb2_id}.npy')
    np.save(alignment_filename, soft_aln_np)
    print(f"✅ Saved soft alignment matrix to {alignment_filename}")

    print("\nProcessing complete. Check the output directory for results.")

if __name__ == "__main__":
    main()
