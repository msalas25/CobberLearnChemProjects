import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Find theobromine
compounds = pcp.get_compounds("theobromine", "name")

theobromine = compounds[0]

# Get SMILES from PubChem
smiles = theobromine.isomeric_smiles

print("Theobromine SMILES:", smiles)

# Create RDKit molecule
mol = Chem.MolFromSmiles(smiles)

# Calculate descriptors
exact_weight = Descriptors.ExactMolWt(mol)
h_bond_donors = Lipinski.NumHDonors(mol)
tpsa = Descriptors.TPSA(mol)
rotatable_bonds = Lipinski.NumRotatableBonds(mol)
aromatic_rings = Lipinski.NumAromaticRings(mol)

# Print results
print("Exact Molecular Weight:", exact_weight)
print("Hydrogen Bond Donors:", h_bond_donors)
print("TPSA:", tpsa)
print("Rotatable Bonds:", rotatable_bonds)
print("Aromatic Rings:", aromatic_rings)
import random

# List of drugs from your assignment
drugs = [
    "Ibuprofen",
    "Acetaminophen",
    "Morphine",
    "Aspirin",
    "Caffeine",
    "Theobromine",
    "Naproxen",
    "Ciprofloxacin",
    "Amoxicillin",
    "Lidocaine"
]

# Drugs that cannot be chosen
excluded_drugs = [
    "Ibuprofen",
    "Acetaminophen",
    "Morphine",
    "Aspirin"
]

# Remove excluded drugs
available_drugs = [
    drug for drug in drugs
    if drug not in excluded_drugs
]

# Randomly choose one drug
chosen_drug = random.choice(available_drugs)

print("Your randomly selected drug is:", chosen_drug)
from rdkit import Chem
from rdkit.Chem import Descriptors

# SMILES string for ethanol
smiles = "CCO"

# Create molecule object
mol = Chem.MolFromSmiles(smiles)

# Calculate descriptors
exact_weight = Descriptors.ExactMolWt(mol)
h_bond_donors = Descriptors.NumHDonors(mol)
tpsa = Descriptors.TPSA(mol)

# Print results
print("Exact Molecular Weight:", exact_weight)
print("Hydrogen Bond Donors:", h_bond_donors)
print("TPSA:", tpsa)