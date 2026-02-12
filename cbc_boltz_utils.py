import os
import yaml
import time
import subprocess
import csv
import shutil
import json
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser,PDBIO
from Bio.PDB.mmcifio import MMCIFIO



def save_pdb_with_seqres(pdb_path, output_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", str(pdb_path))
    
    seqres_lines = []
    
    for chain in structure.get_chains():
        chain_id = chain.id
        # Extract residues (ignoring HETATMs)
        residues = [res.get_resname() for res in chain.get_residues() if res.id[0] == " "]
        num_res = len(residues)
        
        # PDB SEQRES format: 13 residues per line
        # Format: SEQRES [Serial] [Chain] [NumRes] [Res1] [Res2]...
        for i in range(0, num_res, 13):
            line_res = residues[i:i+13]
            serial = (i // 13) + 1
            res_string = " ".join(f"{res:>3}" for res in line_res)
            # Line format according to PDB columns
            line = f"SEQRES {serial:>3} {chain_id} {num_res:>4}  {res_string:<52}\n"
            seqres_lines.append(line)

    # 1. Get the ATOM records using Biopython
    import io
    io_buffer = io.StringIO()
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(io_buffer)
    atom_data = io_buffer.getvalue()

    # 2. Combine SEQRES and ATOM data
    with open(output_path, "w") as f:
        f.writelines(seqres_lines)
        f.write(atom_data)

def create_boltz_yaml(yaml_path, target_seq, binder_seq, auto_disulfide, is_cyclic,use_template,input_struct,template_dir,force,threshold):
    """Generates Boltz-1 YAML with cyclic/covalent constraints for BindCraft."""



    disulf_res = []
    if auto_disulfide:
        for index, aa in enumerate(binder_seq):
            if aa == "C":
                disulf_res.append(index + 1)
    
    manifest = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": target_seq, "msa": "empty"}},
            {"protein": {"id": "B", "sequence": binder_seq, "msa": "empty", "cyclic": is_cyclic}}
        ]
    }
        
    if len(disulf_res) == 2:
        manifest["constraints"] = [{
            "bond": {
                "atom1": ["B", int(disulf_res[0]), "SG"],
                "atom2": ["B", int(disulf_res[1]), "SG"]
            }
        }]

    if use_template:
        
        template_dir.mkdir(exist_ok=True)
        target_pdb = template_dir / input_struct.name.replace('.pdb', '_template.pdb')

        save_pdb_with_seqres(input_struct, target_pdb)
        if force: 
            manifest["templates"] = [
                {
                "pdb": str(target_pdb.resolve()),
                "force": True,      # 
                "threshold": threshold    # Distance in Angstroms the bakcbone can deviate from the template
                }
            ]
        else:
            manifest["templates"] = [
                {
                "pdb": str(target_pdb.resolve()),
                }

                ]
    # Ensure 
    with open(yaml_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def run_boltz(yaml_dir, output_dir, conda_env="boltz"):
    """
    Runs Boltz via 'conda run' to ensure it uses the correct environment 
    without requiring the user to manually switch shells.
    """
    start_time = time.time()
    
    # The 'conda run -n' command executes the tool inside the boltz env
    # and returns to the current env automatically.
    cmd = [
        'conda', 'run', '-n', conda_env, 
        'boltz', 'predict', str(yaml_dir), 
        "--out_dir", str(output_dir), 
        "--use_potentials", 
        "--output_format", "pdb"
    ]
    
   
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: Boltz failed. Check if the conda env {conda_env} exists. Error: {e}")
        raise

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f}s")

def parse_results(boltz_out_dir, csv_path, structures_dir, get_sequence_func):
    """Parses Boltz outputs and extracts pLDDT for target and binder."""
    boltz_out_dir = Path(boltz_out_dir)
    structures_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_dirs = list(boltz_out_dir.glob("predictions/*"))
    all_data = []
    
    headers = [
        "Design",'binder_sequence' ,"boltz_confidence_score", "boltz_complex_pDE", "boltz_complex_iPDE", "boltz_pTM_complex", "boltz_ipTM_complex", 
        "boltz_pLDDT_complex", "boltz_ipLDDT_complex", "boltz_pLDDT_target", "boltz_pLDDT_binder"
    ]

    for pred_dir in prediction_dirs:
        struct_files = list(pred_dir.glob("*.pdb"))
        if not struct_files: continue
            
        src_struct = struct_files[0]
        # Copy and rename (remove model_0 suffix)
        dest_name = src_struct.name.replace('_model_0', '')
        shutil.copy2(src_struct, structures_dir / dest_name)
        
        # Load pLDDT scores
        plddt_files = list(pred_dir.glob("plddt*"))
        if not plddt_files: continue
        data = np.load(plddt_files[0].resolve())
        arr = data[data.files[0]]

        # Calc lengths for slicing - NOTE: Chain A=Target, B=Binder
        target_len = len(get_sequence_func(src_struct, "A"))
        binder_len = len(get_sequence_func(src_struct, "B"))

        target_plddt = np.mean(arr[0:target_len]) 
        binder_plddt = np.mean(arr[target_len:target_len + binder_len])

        json_files = list(pred_dir.glob("*.json"))
        if not json_files: continue

        with open(json_files[0], 'r') as f:
            m = json.load(f)
            all_data.append({
                "Design": dest_name.replace('.pdb', ''),
                "binder_sequence": get_sequence_func(src_struct, "B"),
                "boltz_confidence_score": m.get("confidence_score"),
                "boltz_pTM_complex": m.get("ptm"),
                "boltz_ipTM_complex": m.get("iptm"),
                "boltz_complex_pDE": m.get("complex_pde"),
                "boltz_complex_iPDE": m.get("complex_ipde"),
                "boltz_pLDDT_complex": m.get("complex_plddt"),
                "boltz_ipLDDT_complex": m.get("complex_iplddt"),
                "boltz_pLDDT_target": target_plddt,
                "boltz_pLDDT_binder": binder_plddt
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_data)