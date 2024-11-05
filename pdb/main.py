import os
from Bio import PDB
import pandas as pd
from collections import defaultdict

class SimpleStructureProcessor:
    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)
        # First we define valid residues
        self.protein_residues = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                               'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                               'THR', 'TRP', 'TYR', 'VAL'}
        self.rna_residues = {'A', 'C', 'G', 'U'}

    def extract_data(self, pdb_file):
        """Extract sequences and coordinates from PDB file."""
        structure = self.parser.get_structure('structure', pdb_file)
        
        # Storing data for each chain
        chain_data = defaultdict(list)
        sequences = defaultdict(str)
        
        # We process first model only (typical for X-ray structures apparently + for simplicity)
        model = structure[0]
        
        for chain in model:
            chain_id = chain.id
            
            for residue in chain:
                res_name = residue.get_resname()
                
                # Process protein residues
                if res_name in self.protein_residues:
                    chain_type = 'protein'
                    try:
                        # Get CA coordinates for proteins
                        ca_atom = residue['CA']
                        coords = ca_atom.get_coord()
                        sequences[chain_id] += res_name
                        
                        chain_data[chain_id].append({
                            'chain_id': chain_id,
                            'chain_type': chain_type,
                            'residue': res_name,
                            'position': residue.id[1],
                            'x': coords[0],
                            'y': coords[1],
                            'z': coords[2]
                        })
                    except Exception as e:
                        print(e,ca_atom)
                        continue
                
                # Process RNA residues
                elif res_name in self.rna_residues:
                    chain_type = 'RNA'
                    try:
                        # Get C1' coordinates for RNA (a consistent reference point as we discuessed)
                        c1_atom = residue["C1'"]
                        coords = c1_atom.get_coord()
                        sequences[chain_id] += res_name
                        
                        chain_data[chain_id].append({
                            'chain_id': chain_id,
                            'chain_type': chain_type,
                            'residue': res_name,
                            'position': residue.id[1],
                            'x': coords[0],
                            'y': coords[1],
                            'z': coords[2]
                        })
                    except KeyError:
                        continue
                else:
                    print(f"missing protein residue : {res_name}")

        
        return chain_data, sequences

    def process_pdb_files(self, pdb_files, output_dir='processed_data3'):
        """Process multiple PDB files and create ML dataset."""
        os.makedirs(output_dir, exist_ok=True)
        all_data = []
        sequences_dict = {}
        
        for pdb_file in pdb_files:
            if not os.path.exists(pdb_file):
                print(f"File not found: {pdb_file}")
                continue
                
            print(f"Processing {pdb_file}...")
            pdb_id = os.path.basename(pdb_file).split('.')[0]
            
            try:
                chain_data, sequences = self.extract_data(pdb_file)
                
                # Adding data from all chains
                for chain_id, data in chain_data.items():
                    for entry in data:
                        entry['pdb_id'] = pdb_id
                        all_data.append(entry)
                
                # Store sequences
                sequences_dict[pdb_id] = sequences
                
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
                continue

        # Create DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Save coordinates dataset
            coords_path = os.path.join(output_dir, 'coordinates.csv')
            df.to_csv(coords_path, index=False)
            print(f"Coordinates saved to {coords_path}")
            
            # Save sequences
            sequences_path = os.path.join(output_dir, 'sequences.txt')
            with open(sequences_path, 'w') as f:
                for pdb_id, chains in sequences_dict.items():
                    f.write(f">{pdb_id}\n")
                    for chain_id, sequence in chains.items():
                        f.write(f"Chain {chain_id}: {sequence}\n")
            print(f"Sequences saved to {sequences_path}")
            
            return df
        
        return None


if __name__ == "__main__":
    processor = SimpleStructureProcessor()
    pdb_files = ['4ola.pdb']  # We can add our PDB file paths here
    dataset = processor.process_pdb_files(pdb_files)