from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit import RDLogger
from tqdm.auto import tqdm
RDLogger.DisableLog('rdApp.*')

tqdm.pandas()


def extract_molecule_features(df, smiles):
    merged = df.merge(smiles, on='DRUG_NAME', how='inner')
    merged = merged[~merged.Smiles.isna()]

    merged['Mol'] = merged['Smiles'].progress_apply(Chem.MolFromSmiles)
    merged = merged[~merged.Mol.isna()]

    merged['MolWt'] = merged['Mol'].progress_apply(Descriptors.ExactMolWt)
    merged['TPSA'] = merged['Mol'].progress_apply(Descriptors.TPSA)
    merged['LogP'] = merged['Mol'].progress_apply(Descriptors.MolLogP)
    merged['HAcceptors'] = merged['Mol'].progress_apply(Lipinski.NumHAcceptors)
    merged['HDonors'] = merged['Mol'].progress_apply(Lipinski.NumHDonors)
    merged['RotatableBonds'] = merged['Mol'].progress_apply(Lipinski.NumRotatableBonds)
    merged['RingCount'] = merged['Mol'].progress_apply(Lipinski.RingCount)
    merged['AromaticRings'] = merged['Mol'].progress_apply(Lipinski.NumAromaticRings)
    merged['RingCount'] = merged['Mol'].progress_apply(Lipinski.RingCount)
    merged['ALERTS'] = merged['Mol'].progress_apply(lambda mol: QED.properties(mol).ALERTS)

    merged.drop(columns=['Mol'], inplace=True)

    return merged
