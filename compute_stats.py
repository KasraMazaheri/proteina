import pandas as pd
import argparse
import subprocess
import shutil
import os
import biotite.structure.io.pdb as pdb
import biotite.structure.sse as annotate
import numpy as np
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from tqdm import tqdm

def analyze_secondary_structure(pdb_path):
    # 1. Load the structure
    pdb_file = pdb.PDBFile.read(pdb_path)
    array = pdb_file.get_structure(model=1)

    # 2. Filter for CA atoms (P-SEA uses Carbon-alpha positions)
    ca_atoms = array[array.atom_name == "CA"]

    # 3. Run the P-SEA algorithm
    # Returns an array of characters: 'a' (alpha), 'b' (beta), 'c' (coil)
    sse = annotate.annotate_sse(ca_atoms)

    # 4. Count occurrences
    total = len(sse)
    alpha_count = np.count_nonzero(sse == 'a')
    beta_count = np.count_nonzero(sse == 'b')
    coil_count = np.count_nonzero(sse == 'c')

    return np.array([alpha_count, beta_count, coil_count]) / total, total

FOLDSEEK_DB_DIR = "/homes/kasram/broteina/SiD_Protein/additional_files/foldseek_databases"

def evaluate_novelty(ckpt_name, dataset, base):
    db_path = f"{FOLDSEEK_DB_DIR}/{dataset}"
    if not os.path.exists(f"{db_path}.dbtype"):
        raise FileNotFoundError(f"Foldseek database not found at {db_path} (expected files like {db_path}.dbtype). Set FOLDSEEK_DB_DIR to the directory containing the {dataset} database.")
    out_path = f"foldseek_tmp/{ckpt_name}/novelty_{dataset}"
    subprocess.run(
        f"foldseek easy-search {base}/{ckpt_name}/pdbs/designable {db_path} {out_path} foldseek_tmp/{ckpt_name}  --alignment-type 1 --exhaustive-search --tmscore-threshold 0.0 --max-seqs 10000000000 --format-output query,target,alntmscore,lddt",
        shell=True, check=True,
    )
    df = pd.read_csv(out_path, header=None, names=["protein","target","TM","lddt"], sep="\t")
    tot_tm = 0
    for protein in tqdm(df["protein"].unique()):
        max_tm = df[df["protein"] == protein]["TM"].max()
        tot_tm += max_tm
    return tot_tm / len(df["protein"].unique())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute stats for a given checkpoint")
    parser.add_argument('--ckpt_name', '-c', help='Name of the checkpoint to process', required=True)
    args = parser.parse_args()
    ckpt_name = args.ckpt_name
    base = "samples/neurips/"

    designable_list = os.listdir(f"{base}/{ckpt_name}/pdbs/designable")
    num_designable = len(designable_list)
    num_undesignable = len(os.listdir(f"{base}/{ckpt_name}/pdbs/undesignable"))
    print("Designability:", num_designable / (num_designable + num_undesignable))

    sec = np.zeros((3,))
    struc = {}
    nres_by_file = {}

    for f in designable_list:
        path = f"{base}/{ckpt_name}/pdbs/designable/{f}"
        sec_frac, nres = analyze_secondary_structure(path)
        sec += sec_frac
        struc[f] = get_structure(path)
        nres_by_file[f] = nres

    print("Secondary Structure Content:", sec / num_designable)

    os.makedirs("foldseek_tmp", exist_ok=True)
    if os.path.exists(f"foldseek_tmp/{ckpt_name}"):
        shutil.rmtree(f"foldseek_tmp/{ckpt_name}")
    os.makedirs(f"foldseek_tmp/{ckpt_name}")
    subprocess.run(f"foldseek easy-cluster {base}/{ckpt_name}/pdbs/designable foldseek_tmp/{ckpt_name}/res foldseek_tmp/{ckpt_name} --alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    df = pd.read_csv(f"foldseek_tmp/{ckpt_name}/res_cluster.tsv", sep="\t", header=None, names=["cluster", "protein"])
    print("Diversity:", len(df["cluster"].unique()) / len(df))

    tm_sum = {}
    tm_count = {}

    for idx, f1 in enumerate(tqdm(designable_list)):
        nres = nres_by_file[f1]
        coords1, seq1 = get_residue_data(next(struc[f1].get_chains()))
        for f2 in designable_list[idx+1:]:
            if nres_by_file[f2] == nres:
                coords2, seq2 = get_residue_data(next(struc[f2].get_chains()))
                result = tm_align(coords1, coords2, seq1, seq2)
                tm_sum[nres] = tm_sum.get(nres, 0) + result.tm_norm_chain1
                tm_count[nres] = tm_count.get(nres, 0) + 1

    tm_avg = []
    for nres in sorted(tm_count.keys()):
        tm_avg.append((nres, tm_sum[nres] / tm_count[nres]))
    print("TMScore by Length:", tm_avg)
    if tm_avg:
        print("Average TMScore:", sum(v for _, v in tm_avg) / len(tm_avg))

    print("PDB Novelty:", evaluate_novelty(ckpt_name, "pdb", base))
    print("AFDB Novelty:", evaluate_novelty(ckpt_name, "afdb", base))