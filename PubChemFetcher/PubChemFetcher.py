import pubchempy as pcp

# Search for theobromine
theobromine = pcp.get_compounds("theobromine", "name")[0]

# Display information
print("Name:", theobromine.iupac_name)
print("Molecular Formula:", theobromine.molecular_formula)
print("Molecular Weight:", theobromine.molecular_weight)
print("Canonical SMILES:", theobromine.canonical_smiles)
print("Isomeric SMILES:", theobromine.isomeric_smiles)
print("CID:", theobromine.cid)