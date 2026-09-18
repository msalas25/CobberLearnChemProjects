import pubchempy as pcp # Search for theobromine by name compounds = pcp.get_compounds("theobromine", "name") # Check if the compound was found if compounds: theobromine = compounds[0] print("\nTheobromine Information") print("=" * import pubchempy as pcp

# Search for theobromine in PubChem
compounds = pcp.get_compounds("theobromine", "name")

# Get the first result
theobromine = compounds[0]

# Print the information
print("Molecular Weight:", theobromine.molecular_weight)
print("Molecular Formula:", theobromine.molecular_formula)
print("SMILES:", theobromine.smiles)
30) print(f"Molecular Weight: {theobromine.molecular_weight} g/mol") print(f"Molecular Formula: {theobromine.molecular_formula}") print(f"SMILES: {theobromine.isomeric_smiles}") print("=" * 30) else: print("Theobromine could not be found.")