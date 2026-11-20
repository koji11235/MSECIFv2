"""Regression tests that run msECIF on the bundled SampleStructures.

Each structure is run through GetECIF, and the features are checked against the
reference CSVs written by Ex1_generate_feature.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import msecif_v2

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "SampleStructures"
PDB_IDS = sorted(p.name for p in SAMPLES.iterdir() if p.is_dir())
LIGAND_SUFFIXES = ["_ligand_H_added.sdf", "_ligand.mol2"]
SHELLS = [2.5, 4.5, 6.5, 8.5, 10.0]  # as in Ex1_generate_feature.ipynb


def _protein(pdb):
    return str(SAMPLES / pdb / f"{pdb}_protein.pdb")


def _ligand(pdb, suffix):
    return str(SAMPLES / pdb / f"{pdb}{suffix}")


@pytest.fixture(scope="module")
def multi_shelled_reference():
    return pd.read_csv(ROOT / "Ex1_multi-shelledECIF_feature.csv", index_col="PDB")


@pytest.fixture(scope="module")
def weighted_reference():
    return pd.read_csv(ROOT / "Ex1_weightedECIF_feature.csv", index_col="PDB")


def test_sample_structures_found():
    assert PDB_IDS, f"no sample structures found under {SAMPLES}"


def test_rdkit_explicit_valence():
    atom = Chem.MolFromSmiles("CO").GetAtomWithIdx(0)
    assert atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT) == 1


@pytest.mark.parametrize("suffix", LIGAND_SUFFIXES)
@pytest.mark.parametrize("pdb", PDB_IDS)
def test_getecif_runs_on_sample(pdb, suffix):
    ecif = msecif_v2.GetECIF(_protein(pdb), _ligand(pdb, suffix))
    assert len(ecif) == len(msecif_v2.PossibleECIF)
    assert sum(ecif) > 0


@pytest.mark.parametrize("suffix", LIGAND_SUFFIXES)
@pytest.mark.parametrize("pdb", PDB_IDS)
def test_multi_shelled_matches_reference(pdb, suffix, multi_shelled_reference):
    assert list(multi_shelled_reference.columns) == msecif_v2.make_feature_names_multi_shelled_ecif(SHELLS)
    feature = msecif_v2.get_multi_shelled_ecif(_protein(pdb), _ligand(pdb, suffix), SHELLS)
    np.testing.assert_array_equal(np.asarray(feature), multi_shelled_reference.loc[pdb].to_numpy())


@pytest.mark.parametrize("pdb", PDB_IDS)
def test_weighted_matches_reference(pdb, weighted_reference):
    assert list(weighted_reference.columns) == list(msecif_v2.PossibleECIF)
    feature = msecif_v2.get_weighted_ecif(
        _protein(pdb), _ligand(pdb, "_ligand_H_added.sdf"), distance_cutoff=10.0, squared=True
    )
    np.testing.assert_allclose(np.asarray(feature), weighted_reference.loc[pdb].to_numpy(), rtol=1e-9)
