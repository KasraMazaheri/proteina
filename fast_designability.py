import argparse
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess

from ProteinMPNN.protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from ProteinMPNN.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

from training.proteina.proteinfoundation.utils.align_utils.align_utils import kabsch_align_ind
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from training.proteina.evaluations.evaluation_utils import parse_pdb_file

from dotenv import load_dotenv

import click

import argparse
import random
import shutil
import loralib as lora

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from proteinfoundation.utils.lora_utils import replace_lora_layers


from proteinfoundation.metrics.designability import scRMSD
from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
)
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level, write_prot_to_pdb

def get_args():
    
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--suppress_print", type=int, default=0, help="0 for False, 1 for True")

  
    argparser.add_argument("--ca_only", action="store_true", default=True, help="Parse CA-only structures and use CA-only models (default: false)")   
    argparser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    argparser.add_argument("--use_soluble_model", action="store_true", default=False, help="Flag to load ProteinMPNN weights trained on soluble proteins only.")


    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")
 
    argparser.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")

    argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    argparser.add_argument("--path_to_fasta", type=str, default="", help="score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")


    argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    argparser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   
 
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=8, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=200, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
    
    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_chains", type=str, default='A', help="Define which chains need to be designed for a single PDB ")
    argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
   
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
    
    return argparser.parse_args(args = [])

class Designability:

    def __init__(self, device):

        self.device = device

        args = get_args()

        print_all = args.suppress_print == 0

        if args.path_to_model_weights:
            model_folder_path = args.path_to_model_weights
            if model_folder_path[-1] != '/':
                model_folder_path = model_folder_path + '/'
        else: 
            file_path = os.path.realpath(__file__)
            k = file_path.rfind("/")
            # file_path = file_path[:k-1]
            # k = file_path.rfind("/")
            # file_path = file_path[:k-1]
            # k = file_path.rfind("/")
            if args.ca_only:
                print("Using CA-ProteinMPNN!")
                model_folder_path = file_path[:k] + '/ProteinMPNN/ca_model_weights/'
                if args.use_soluble_model:
                    print("WARNING: CA-SolubleMPNN is not available yet")
                    sys.exit()
            else:
                if args.use_soluble_model:
                    print("Using ProteinMPNN trained on soluble proteins only!")
                    model_folder_path = file_path[:k] + '/soluble_model_weights/'
                else:
                    model_folder_path = file_path[:k] + '/vanilla_model_weights/'

        checkpoint_path = model_folder_path + f'{args.model_name}.pt'

        hidden_dim = 128
        num_layers = 3

        checkpoint = torch.load(checkpoint_path, map_location=device)
        noise_level_print = checkpoint['noise_level']
        model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        if print_all:
            print(40*'-')
            print('Number of edges:', checkpoint['num_edges'])
            print(f'Training noise level: {noise_level_print}A')

        self.args = args

        local_dir = "/home/gridsan/mmorsy/proteins_project/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a"
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir, local_files_only=True
        )
        self.esm_model = EsmForProteinFolding.from_pretrained(
            local_dir, local_files_only=True
        ).to(device)

        print("Loaded ESM-Fold model for structure prediction.")

    def proteinMPNN(self, proteins, return_grad = False):

        args = self.args

        if args.seed:
            seed=args.seed
        else:
            seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        args.batch_size = min(args.batch_size, proteins.shape[0])
        
        NUM_BATCHES = proteins.shape[0]//args.batch_size
        temperatures = [float(item) for item in args.sampling_temp.split()]
        omit_AAs_list = args.omit_AAs
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        alphabet_dict = dict(zip(alphabet, range(21)))    
        print_all = args.suppress_print == 0 
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
        
        chain_id_dict = None
        fixed_positions_dict = None
        pssm_dict = None
        omit_AA_dict = None
        bias_AA_dict = None
        tied_positions_dict = None
        bias_by_res_dict = None
    
        bias_AAs_np = np.zeros(len(alphabet))

        all_seqs = []
        grads = []
        
        for ix in range(NUM_BATCHES):

            start = ix*args.batch_size
            end = start + args.batch_size

            X = proteins[start:end]
            B,N = X.shape[:2]
            S = torch.zeros((B,N), device=X.device)
            mask = torch.ones((B,N), device=X.device)
            chain_M = torch.ones((B,N), device=X.device)
            chain_encoding_all = torch.zeros((B,N), device=X.device)
            residue_idx = torch.arange(N, device=X.device).unsqueeze(0).expand(B, N)
            chain_M_pos = torch.ones((B,N), device=X.device)
            omit_AA_mask = torch.zeros((B,N,21), device=X.device)
            pssm_coef = torch.zeros((B,N), device=X.device)
            pssm_bias = torch.zeros((B,N,21), device=X.device)
            pssm_log_odds_mask = torch.ones((B,N,21), device=X.device)
            bias_by_res_all = torch.zeros((B,N,21), device=X.device)

            for temp in temperatures:
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                sample_dict = self.model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                for i in range(B):
                    all_seqs.append(_S_to_seq(sample_dict['S'][i], chain_M[i]))
                
                if return_grad:
                    log_probs = sample_dict['log_probs']
                    # grad = torch.autograd.grad(log_probs, X, grad_outputs=torch.ones_like(log_probs, device='cuda'))[0]
                    # print(torch.autograd.grad(log_probs, X, grad_outputs=torch.ones_like(log_probs, device='cuda'))[0][0])
                    # grads.append(grad)

        return all_seqs, grads

    def scRMSD(self, proteins, return_grad = False):

        proteins.requires_grad_(return_grad)
        
        ns = self.args.num_seq_per_target
        proteins_copied = proteins.repeat_interleave(ns, dim=0)
        with torch.set_grad_enabled(return_grad):
            seqs, log_prob_grads = self.proteinMPNN(proteins_copied, return_grad)
        batch_size = min(20, len(seqs))
        rmsd_list = []
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                )
                inputs = {k: inputs[k].to(self.device) for k in inputs}

                outputs = self.esm_model(**inputs)
                atom37_outputs = atom14_to_atom37(outputs["positions"][-1], outputs)
                pred_positions = atom37_outputs[:, :, 1, :]

            pred_positions.requires_grad_(return_grad)

            for j in range(len(pred_positions)):
                coors_1, coors_2 = kabsch_align_ind(pred_positions[j], proteins[(i+j)//ns], ret_both=True)
                sq_err = (coors_1 - coors_2) ** 2
                rmsd_list.append(sq_err.sum(dim=-1).mean().sqrt())

        rmsd_list = torch.stack(rmsd_list)
        rmsd_list = rmsd_list.view(-1, ns)
        scores = rmsd_list.min(dim=-1).values

        if return_grad:

            log_prob_grads = torch.cat(log_prob_grads, dim=0)
            log_prob_grads = log_prob_grads.view(-1, ns, *log_prob_grads.shape[1:]).sum(dim=1)
            rmsd_grad = torch.autograd.grad(scores, proteins, grad_outputs=torch.ones_like(scores))[0]

            return scores, scores[:, None, None] * log_prob_grads + rmsd_grad

        return scores
    
class GenDataset(Dataset):
    """
    Dataset that indicates length of the proteins to generate,
    discretization step size, and number of samples per length,
    empirical (len, cath_code) joint distribution.
    """

    bucket_min_len = 50
    bucket_max_len = 274
    bucket_step_size = 25

    def __init__(self, nres=[110], dt=0.005, nsamples=10, len_cath_codes=None):
        # nres is a list of integers
        # len_cath_codes is a list of [len, List[cath_code]] pairs, representing (len, cath_code) joint distribution
        super(GenDataset, self).__init__()
        self.nres = [int(n) for n in nres]
        self.dt = dt
        if isinstance(nsamples, List):
            assert len(nsamples) == len(nres)
            self.nsamples = nsamples
        elif isinstance(nsamples, int):
            self.nsamples = [nsamples] * len(nres)
        else:
            raise ValueError(f"Unknown type of nsamples {type(nsamples)}")
        self.cath_codes_given_len_bucket = self.bucketize(len_cath_codes)

    def bucketize(self, len_cath_codes):
        """Build length buckets for cath_codes. Record the cath_code distribution given length bucket"""
        if len_cath_codes is None:
            return None
        bucket = list(
            range(self.bucket_min_len, self.bucket_max_len, self.bucket_step_size)
        )
        cath_codes_given_len_bucket = [[] for _ in range(len(bucket))]
        for _len, code in len_cath_codes:
            bucket_idx = (_len - self.bucket_min_len) // self.bucket_step_size
            cath_codes_given_len_bucket[bucket_idx].append(code)
        return cath_codes_given_len_bucket

    def __len__(self):
        return len(self.nres)

    def __getitem__(self, index):
        result = {
            "nres": self.nres[index],
            "dt": self.dt,
            "nsamples": self.nsamples[index],
        }
        if self.cath_codes_given_len_bucket is not None:
            if self.nres[index] <= self.bucket_max_len:
                bucket_idx = (
                    self.nres[index] - self.bucket_min_len
                ) // self.bucket_step_size
            else:
                bucket_idx = -1
            result["cath_code"] = random.choices(
                self.cath_codes_given_len_bucket[bucket_idx], k=self.nsamples[index]
            )
        return result


def split_nlens(nlens_dict, max_nsamples=16, n_replica=1):
    """
    Split nlens into data points (len, nsample) as val dataset and guarantee that
        1. len(val_dataset) should be a multiple of n_replica, to ensure that we don't introduce additional samples for multi-gpu validation
        2. nsample should be the same for all data points if n_replica > 1 (multi-gpu)

    Args:
        nlens_dict: Dict of nlens distribution.
            nlens_dict["length_ranges"] is a set of bin boundaries.
            nlens_dict["length_distribution"] is the numbers of samples in each bin
        max_nsamples: Maximum nsample in each data point
        n_replica: Number of GPUs

    Returns:
        lens_sample (List[int]): List of len for val data points
        nsamples (List[int]): List of nsample for val data points
    """
    lengths_range = nlens_dict["length_ranges"].tolist()
    length_distribution = nlens_dict["length_distribution"].tolist()
    lens_sample, nsamples = [], []
    for length, cnt in zip(lengths_range, length_distribution):
        for i in range(0, cnt, max_nsamples):
            lens_sample.append(length)
            if i + max_nsamples <= cnt:
                nsamples.append(max_nsamples)
            else:
                nsamples.append(cnt - i)

    max_nsamples = max(nsamples)
    for i in range(len(nsamples)):
        nsamples[i] += max_nsamples - nsamples[i]

    while len(lens_sample) % n_replica != 0:
        lens_sample.append(lens_sample[-1])
        nsamples.append(max_nsamples)

    return lens_sample, nsamples


def parse_nlens_cfg(cfg):
    """Parse lengths distribution. Either loading an empirical one or build with arguments in yaml file"""
    if cfg.get("nres_lens_distribution_path") is not None:
        # Sample according to length distribution
        nlens_dict = torch.load(cfg.nres_lens_distribution_path)
    else:
        # Sample with pre-specified lengths
        if cfg.nres_lens:
            _lens_sample = cfg.nres_lens
        else:
            _lens_sample = [
                int(v)
                for v in np.arange(cfg.min_len, cfg.max_len + 1, cfg.step_len).tolist()
            ]
        nlens_dict = {
            "length_ranges": torch.as_tensor(_lens_sample),
            "length_distribution": torch.as_tensor(
                [cfg.nsamples_per_len] * len(_lens_sample)
            ),
        }
    return nlens_dict


def parse_len_cath_code(cfg):
    """Load (len, cath_codes) joint distribution. Apply mask according to the guidance cath code level"""
    if cfg.get("len_cath_code_path") is not None:
        logger.info(
            f"Loading empirical (length, cath_code) distribution from {cfg.len_cath_code_path}"
        )
        _len_cath_codes = torch.load(cfg.len_cath_code_path)
        level = cfg.get("cath_code_level")
        len_cath_codes = []
        for i in range(len(_len_cath_codes)):
            _len, code = _len_cath_codes[i]
            code = mask_cath_code_by_level(code, level="H")
            if level == "A" or level == "C":
                code = mask_cath_code_by_level(code, level="T")
                if level == "C":
                    code = mask_cath_code_by_level(code, level="A")
            len_cath_codes.append((_len, code))
    else:
        logger.info(
            "No empirical (length, cath_code) distribution provided. Use unconditional training."
        )
        len_cath_codes = None
    return len_cath_codes

class ModelDesignability:

    def __init__(self):

        load_dotenv()

        parser = argparse.ArgumentParser(description="Job info")
        parser.add_argument(
            "--config_name",
            type=str,
            default="inference_base",
            help="Name of the config yaml file.",
        )
        args = parser.parse_args()
        logger.info(" ".join(sys.argv))

        assert (
            torch.cuda.is_available()
        ), "CUDA not available"  # Needed for ESMfold and designability
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        )  # Send to stdout

        # Inference config
        # If config_subdir is None then use base inference config
        # Otherwise use config_subdir/some_config
        if args.config_subdir is None:
            config_path = "../configs/experiment_config"
        else:
            config_path = f"../configs/experiment_config/{args.config_subdir}"

        with hydra.initialize(config_path, version_base=hydra.__version__):
            # If number provided use it, otherwise name
            if args.config_number != -1:
                config_name = f"inf_{args.config_number}"
            else:
                config_name = args.config_name
            cfg = hydra.compose(config_name=config_name)
            logger.info(f"Inference config {cfg}")
            run_name = cfg.run_name_

        assert (
            not cfg.compute_designability or not cfg.compute_fid
        ), "Designability cannot be computed together with FID"

        # Set root path for this inference run
        root_path = f"./inference/{config_name}"
        if os.path.exists(root_path):
            shutil.rmtree(root_path)
        os.makedirs(root_path, exist_ok=True)

        # Set seed
        logger.info(f"Seeding everything to seed {cfg.seed}")
        L.seed_everything(cfg.seed)

        # Create length dataset
        nlens_dict = parse_nlens_cfg(cfg)
        lens_sample, nsamples = split_nlens(
            nlens_dict, max_nsamples=cfg.max_nsamples, n_replica=1
        )  # Assume running on 1 GPU
        if cfg.fold_cond:
            len_cath_codes = parse_len_cath_code(cfg)
        else:
            len_cath_codes = None
        dataset = GenDataset(
            nres=lens_sample, nsamples=nsamples, dt=cfg.dt, len_cath_codes=len_cath_codes
        )
        self.dataloader = DataLoader(dataset, batch_size=1)
        # Note: Batch size should be left as 1, it is not the actual batch size.
        # Each sample returned by this loader is a 3-tuple (L, nsamples, dt) where
        #   - L (int) is the number of residues in the proteins to be samples
        #   - nsamples (int) is the number of proteins to generate (happens in parallel),
        #     so if nsamples=10 it means that it will produce 10 proteins of length L (all sampled in parallel)
        #   - dt (float) step-size used for the ODE integrator
        #   - cath_code (Optional[List[str]]) cath code for conditional generation

        # Flatten config and use it to initialize results dataframes columns
        flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
        flat_dict = {k: str(v) for k, v in flat_dict.items()}
        columns = list(flat_dict.keys())

        # Sample the model
        self.trainer = L.Trainer(accelerator="gpu", devices=torch.cuda.device_count())
        self.cfg = cfg

    def compute_designability(self, ckpt_file):

        # Load model from checkpoint
        logger.info(f"Using checkpoint {ckpt_file}")
        assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

        # Check if using lora and load model
        if not self.cfg.lora.use:
            model = Proteina.load_from_checkpoint(ckpt_file)
        
        else: # If using lora, create lora layers and reload the state_dict
            model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
            logger.info("Re-create LoRA layers and reload the weights now")
            replace_lora_layers(
                model,
                self.cfg["lora"]["r"],
                self.cfg["lora"]["lora_alpha"],
                self.cfg["lora"]["lora_dropout"],
            )
            lora.mark_only_lora_as_trainable(model, bias=self.cfg["lora"]["train_bias"])
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])

        # Set inference variables and potentially load autoguidance
        nn_ag = None
        if (
            self.cfg.get("autoguidance_ratio", 0.0) > 0
            and self.cfg.get("guidance_weight", 1.0) != 1.0
        ):
            assert self.cfg.autoguidance_ckpt_path is not None
            ckpt_ag_file = self.cfg.autoguidance_ckpt_path
            model_ag = Proteina.load_from_checkpoint(ckpt_ag_file)
            nn_ag = model_ag.nn

        model.configure_inference(self.cfg, nn_ag=nn_ag)

        self.compute_designability = True
        predictions = self.trainer.predict(model, self.dataloader)
        return (predictions < 2).mean().item()

if __name__ == "__main__":

    model_designability = ModelDesignability()
    print(model_designability.compute_designability())