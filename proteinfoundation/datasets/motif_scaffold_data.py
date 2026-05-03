from pathlib import Path
from typing import Callable, List, Literal, Optional

import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data, Dataset

from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.datasets.pdb_data import PDBDataSplitter
from proteinfoundation.utils.coors_utils import ang_to_nm


class MotifScaffoldDataset(Dataset):
    def __init__(
        self,
        file_names: List[str],
        input_paths: List[str],
        data_dir: str,
        transform: Optional[Callable] = None,
        in_memory: bool = False,
    ):
        self.file_names = file_names
        self.input_paths = input_paths
        self.data_dir = Path(data_dir)
        self.raw_root = self.data_dir / "raw"
        self.transform = transform
        self.in_memory = in_memory
        self.data = None
        self.bad_indices = set()

        if self.in_memory:
            logger.info("Reading motif scaffold data into memory")
            kept_file_names = []
            kept_input_paths = []
            data = []
            for i in range(len(self.file_names)):
                graph = self._load_example(i, allow_skip=True)
                if graph is None:
                    continue
                kept_file_names.append(self.file_names[i])
                kept_input_paths.append(self.input_paths[i])
                data.append(graph)
            self.file_names = kept_file_names
            self.input_paths = kept_input_paths
            self.data = data
            logger.warning(
                f"Loaded {len(self.data)} motif scaffold examples into memory; "
                f"skipped {len(self.bad_indices)} corrupt examples"
            )

    def __len__(self):
        return len(self.file_names)

    def _dict_to_graph(self, item: dict, file_id: str) -> Data:
        nres = int(item["nres"])
        ca_coords = item["ca_coords"].float()
        motif_seq_mask = item["motif_seq_mask"].bool()
        motif_ca_coords = item["motif_ca_coords"].float()

        coords = torch.zeros((nres, 37, 3), dtype=torch.float32)
        coords[:, 1, :] = ca_coords

        coord_mask = torch.zeros((nres, 37), dtype=torch.bool)
        coord_mask[:, 1] = True

        fixed_sequence_mask = motif_seq_mask
        graph = Data(
            id=file_id,
            source_id=str(item["source_id"]),
            coords=coords,
            coord_mask=coord_mask,
            residue_pdb_idx=torch.arange(1, nres + 1, dtype=torch.long),
            seq_pos=torch.arange(nres, dtype=torch.long).unsqueeze(-1),
            fixed_sequence_mask=fixed_sequence_mask,
            motif_mask=fixed_sequence_mask.clone(),
            x_motif=ang_to_nm(motif_ca_coords),
            nres=nres,
            database="motif_scaffold",
        )
        return graph

    def _load_example(self, idx: int, allow_skip: bool = False) -> Optional[Data]:
        file_id = self.file_names[idx]
        input_path = self.raw_root / self.input_paths[idx]
        try:
            item = torch.load(input_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            self.bad_indices.add(idx)
            size_bytes = input_path.stat().st_size if input_path.exists() else -1
            logger.error(
                f"Failed to load motif scaffold example idx={idx} "
                f"file_id={file_id} path={input_path} size_bytes={size_bytes}: {exc!r}"
            )
            if allow_skip:
                return None
            raise
        graph = self._dict_to_graph(item, file_id=file_id)
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __getitem__(self, idx: int) -> Data:
        if self.data is not None:
            return self.data[idx]

        nitems = len(self.file_names)
        if nitems == 0:
            raise IndexError("Motif scaffold dataset is empty")

        for offset in range(nitems):
            candidate_idx = (idx + offset) % nitems
            if candidate_idx in self.bad_indices:
                continue
            graph = self._load_example(candidate_idx, allow_skip=True)
            if graph is not None:
                return graph

        raise RuntimeError(
            "All motif scaffold dataset examples failed to load. "
            "Check the motif dataset files on disk."
        )


class MotifScaffoldLightningDataModule(BaseLightningDataModule):
    def __init__(
        self,
        data_dir: str,
        datasplitter: PDBDataSplitter,
        file_identifier: str = "motif_scaffold_pdb",
        in_memory: bool = False,
        batch_padding: bool = True,
        sampling_mode: Literal["random", "cluster-random", "cluster-reps"] = "random",
        transforms: Optional[List[Callable]] = None,
        pre_transforms: Optional[List[Callable]] = None,
        pre_filters: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(
            batch_padding=batch_padding,
            sampling_mode=sampling_mode,
            transforms=transforms,
            pre_transforms=pre_transforms,
            pre_filters=pre_filters,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.datasplitter = datasplitter
        self.file_identifier = file_identifier
        self.in_memory = in_memory
        self.df_data = None
        self.dfs_splits = None
        self.clusterid_to_seqid_mappings = None

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        if self.df_data is None:
            df_data_name = f"{self.file_identifier}.csv"
            logger.info(f"Loading motif scaffold csv from {df_data_name}")
            self.df_data = pd.read_csv(self.data_dir / df_data_name)

        (
            self.dfs_splits,
            self.clusterid_to_seqid_mappings,
        ) = self.datasplitter.split_data(self.df_data, self.file_identifier)

        if stage == "fit" or stage is None:
            self.train_ds = self.train_dataset()
            self.val_ds = self.val_dataset()
        elif stage == "test":
            self.test_ds = self.test_dataset()

    def _get_dataset(
        self, split: Literal["train", "val", "test"]
    ) -> MotifScaffoldDataset:
        df_split = self.dfs_splits[split]
        return MotifScaffoldDataset(
            file_names=df_split["pdb"].tolist(),
            input_paths=df_split["input_path"].tolist(),
            data_dir=str(self.data_dir),
            transform=self.transform,
            in_memory=self.in_memory,
        )
