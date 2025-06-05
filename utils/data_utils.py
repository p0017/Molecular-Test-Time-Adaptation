# Contains functions and classes needed for preprocessing and loading the data

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Fragments import (
    fr_Al_OH,
    fr_Ar_OH,
    fr_COO,
    fr_NH2,
    fr_amide,
    fr_ester,
    fr_ether,
    fr_halogen,
)
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from beartype import beartype


@beartype
def one_hot_encoding_unk(value, choices: list) -> list:
    """One hot encoding with unknown value handling.
    If the value is in choices, it puts a 1 at the corresponding index.
    Otherwise, it puts a 1 at the last index (unknown).
    Args:
        value: The value to encode
        choices: List of known/valid choices
    Returns:
        list: One-hot encoded vector with length len(choices) + 1
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom) -> list:
    """Returns a feature list for the atom by concatenating one-hot encodings.
    Args:
        atom: RDKit atom object
    Returns:
        list: Flattened list of atom features including symbol, degree, charge,
              hydrogen count, hybridization, aromaticity, and mass
    """
    atom_features = [
        one_hot_encoding_unk(
            atom.GetSymbol(),
            [
                "B",
                "Be",
                "Br",
                "C",
                "Cl",
                "F",
                "I",
                "N",
                "Nb",
                "O",
                "P",
                "S",
                "Se",
                "Si",
                "V",
                "W",
            ],
        ),
        one_hot_encoding_unk(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]),
        one_hot_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]),
        one_hot_encoding_unk(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4]),
        one_hot_encoding_unk(
            int(atom.GetHybridization()),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        ),
        [1 if atom.GetIsAromatic() else 0],
        [atom.GetMass() * 0.01],
    ]
    return sum(atom_features, [])  # Flatten the list into a single list


def get_bond_features(bond) -> list:
    """Returns a one-hot encoded feature list for the bond.
    Args:
        bond: RDKit bond object or None
    Returns:
        list: A 7-dimensional feature vector where:
            - Index 0: 1 if bond is None, 0 otherwise
            - Index 1: 1 if single bond, 0 otherwise
            - Index 2: 1 if double bond, 0 otherwise
            - Index 3: 1 if triple bond, 0 otherwise
            - Index 4: 1 if aromatic bond, 0 otherwise
            - Index 5: 1 if bond is conjugated, 0 otherwise
            - Index 6: 1 if bond is in ring, 0 otherwise
    """
    bond_fdim = 7

    if bond is None:
        bond_features = [1] + [0] * (bond_fdim - 1)
    else:
        bt = bond.GetBondType()
        bond_features = [
            0,  # Zeroth index indicates if bond is None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
    return bond_features


@beartype
class MolGraph:
    """Custom molecular graph for a given SMILES string.
    Creates a graph of a molecule containing atom features,
    bond features, and node connectivity information.
    Args:
        smiles (str): SMILES string representation of the molecule
    Attributes:
        smiles (str): The input SMILES string
        atom_features (list): List of atom feature vectors
        bond_features (list): List of bond feature vectors (stored twice for each bond)
        edge_index (list): List of tuples representing connected atom pairs
    """

    def __init__(self, smiles: str):
        self.smiles = smiles
        self.atom_features = []
        self.bond_features = []
        self.edge_index = []

        molecule = Chem.MolFromSmiles(self.smiles)
        n_atoms = molecule.GetNumAtoms()

        for atom_1 in range(n_atoms):
            self.atom_features.append(
                get_atom_features(molecule.GetAtomWithIdx(atom_1))
            )

            for atom_2 in range(atom_1 + 1, n_atoms):
                bond = molecule.GetBondBetweenAtoms(atom_1, atom_2)
                if bond is None:
                    continue
                bond_features = get_bond_features(bond)
                self.bond_features.append(bond_features)
                self.bond_features.append(
                    bond_features
                )  # Bond features are added twice for both directions
                self.edge_index.extend(
                    [(atom_1, atom_2), (atom_2, atom_1)]
                )  # Edge index list with tuples of connected nodes instead of adjacency matrix


class ChemDataset(Dataset):
    @beartype
    def __init__(
        self,
        smiles: np.ndarray,
        labels,
        flip_prob: float = 0.5,
        noise_std: float = 1.0,
        node_mask_prob: float = 0.5,
        edge_drop_prob: float = 0.5,
        precompute: bool = False,
    ):
        """Initialize ChemDataset with SMILES strings and labels.
        Args:
            smiles (np.ndarray): SMILES molecular representations
            labels: Target labels for the molecules
            flip_prob (float, optional): Probability of flipping bits for denoising task.
                                        Reference: denoising autoencoders use ~10%, BERT uses 15%.
            noise_std (float, optional): Standard deviation for noise addition. Defaults to 0.4.
            precompute (bool, optional): Whether to precompute dataset for faster GPU training.
        """
        super(ChemDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels
        self.cache = {}
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.node_mask_prob = node_mask_prob
        self.edge_drop_prob  = edge_drop_prob
        self.precompute = precompute

        # Precomputing the dataset such that the get method is faster, and the GPU doesn't have to wait for the CPU
        if precompute:
            print(f"Precomputing data...")
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self.process_key, idx)
                    for idx in range(len(self.smiles))
                ]

                for future in as_completed(futures):
                    future.result()

            print(f"Precomputation finished. {len(self.cache)} molecules cached.")

    def process_key(self, key):
        """Process a key to retrieve the corresponding molecule graph data.
        Returns cached molecule if available, otherwise creates a new MolGraph
        from SMILES string and caches the result.
        Args:
            key: Index or identifier for the molecule

        Returns:
            Processed molecule data object
        """
        smiles = self.smiles[key]
        if self.precompute:
            # only cache the fully processed Data when precompute=True
            if smiles not in self.cache:
                molgraph = MolGraph(smiles)
                data = self.molgraph2data(molgraph, key)
                self.cache[smiles] = data
            return self.cache[smiles]
        else:
            # rebuild every time for fresh noise
            molgraph = MolGraph(smiles)
            return self.molgraph2data(molgraph, key)

    def molgraph2data(self, molgraph, key):
        """Convert a molecular graph to PyTorch Geometric Data object with optional noise augmentation.
        Args:
            molgraph: Molecular graph object containing atom_features, edge_index, and bond_features
            key: Index key to retrieve corresponding labels and SMILES from stored data
        Returns:
            tg.data.Data: PyTorch Geometric data object with node features, edge indices,
                          edge attributes, labels, SMILES, and optionally noisy versions
                          of features if augmentation is enabled
        """
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.atom_features, dtype=torch.float)
        data.edge_index = (
            torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        )
        data.edge_attr = torch.tensor(molgraph.bond_features, dtype=torch.float)
        data.y = torch.tensor([self.labels[key]], dtype=torch.float)
        data.smiles = self.smiles[key]

        x_noisy = data.x.clone()
        edge_attr_noisy = data.edge_attr.clone()
        edge_index_noisy = data.edge_index.clone()

        # node‐level masking (zero out entire feature vectors)
        if self.node_mask_prob > 0:
            mask = (
                torch.rand(x_noisy.size(0), device=x_noisy.device) < self.node_mask_prob
            )
            x_noisy[mask] = 0.0

        # random edge‐drop on the noisy graph
        if self.edge_drop_prob > 0:
            num_edges = edge_attr_noisy.size(0)
            if num_edges % 2 != 0:
                raise ValueError(f"Expected even number of edges, got {num_edges}")

            num_bonds = num_edges // 2
            # sample which bonds to KEEP
            keep_bond = (
                torch.rand(num_bonds, device=edge_attr_noisy.device)
                > self.edge_drop_prob
            )
            # never drop all edges
            if keep_bond.sum() == 0:
                idx = torch.randint(num_bonds, (1,), device=keep_bond.device)
                keep_bond[idx] = True

            # repeat each decision for both directions
            keep = torch.empty(num_edges, dtype=torch.bool, device=keep_bond.device)
            keep[0::2] = keep_bond
            keep[1::2] = keep_bond

            edge_attr_noisy = edge_attr_noisy[keep]
            edge_index_noisy = edge_index_noisy[:, keep]

        # bit‐flipping on remaining binary features
        if self.flip_prob > 0:
            # flip node‐features except last (mass) dim
            bf = x_noisy[:, :-1]
            flip_mask = torch.rand_like(bf) < self.flip_prob
            bf[flip_mask] = 1.0 - bf[flip_mask]
            x_noisy[:, :-1] = bf
            # flip edge attributes
            ef = edge_attr_noisy
            flip_mask_e = torch.rand_like(ef) < self.flip_prob
            ef[flip_mask_e] = 1.0 - ef[flip_mask_e]
            edge_attr_noisy = ef

        # Gaussian noise on continuous mass feature (last dim of x)
        if self.noise_std > 0:
            m = x_noisy[:, -1:].clone()
            m = m + m * torch.randn_like(m) * self.noise_std
            x_noisy[:, -1:] = m

        data.x_noisy = x_noisy
        data.edge_index_noisy = edge_index_noisy
        data.edge_attr_noisy = edge_attr_noisy
        return data

    def get(self, key):
        return self.process_key(key)

    def __getitem__(self, key):
        # Standard get method for PyTorch Dataset
        return self.process_key(key)

    def len(self):
        return len(self.smiles)

    def __len__(self):
        # Standard len method for PyTorch Dataset
        return len(self.smiles)


@beartype
def construct_loader(
    data_df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    shuffle: bool = True,
    batch_size: int = 512,
):
    """Constructs a PyTorch Geometric DataLoader from a DataFrame containing SMILES and target data.
    Args:
        data_df (pd.DataFrame): Input DataFrame containing molecular data
        smiles_column (str): Name of the column containing SMILES strings
        target_column (str): Name of the column containing target values
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.
    Returns:
        DataLoader: PyTorch DataLoader for the chemical dataset
    Raises:
        AssertionError: If the input DataFrame is empty
    """
    assert len(data_df) > 0, "DataFrame is empty"

    smiles = data_df[smiles_column].values
    labels = data_df[target_column].values.astype(np.float32)

    dataset = ChemDataset(smiles, labels)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )
    return loader


@beartype
def get_mol_infos(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Add molecular information to a DataFrame including atom counts and functional group descriptors.
    Args:
        df: DataFrame containing molecular data
        smiles_col: Name of the column containing SMILES strings

    Returns:
        DataFrame with additional molecular descriptor columns
    """
    df_updated = deepcopy(df)
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    df_updated["num_atoms"] = [mol.GetNumAtoms() for mol in mols]

    def get_descriptors(mol) -> dict:
        """Compute specific functional group descriptors for a given molecule.
        Args:
            mol: RDKit molecule object or None

        Returns:
            dict: Dictionary with functional group counts
        """
        if mol is None:
            return {
                "fr_Al_OH": 0,
                "fr_Ar_OH": 0,
                "fr_COO": 0,
                "fr_NH2": 0,
                "fr_amide": 0,
                "fr_ester": 0,
                "fr_ether": 0,
                "fr_halogen": 0,
            }
        else:
            return {
                "fr_Al_OH": fr_Al_OH(mol),
                "fr_Ar_OH": fr_Ar_OH(mol),
                "fr_COO": fr_COO(mol),
                "fr_NH2": fr_NH2(mol),
                "fr_amide": fr_amide(mol),
                "fr_ester": fr_ester(mol),
                "fr_ether": fr_ether(mol),
                "fr_halogen": fr_halogen(mol),
            }

    with ThreadPoolExecutor(max_workers=8) as executor:
        functional_groups = list(executor.map(get_descriptors, mols))

    fg_df = pd.DataFrame(functional_groups)
    df_updated = pd.concat([df_updated, fg_df], axis=1)
    return df_updated
