#!/usr/bin/env python3
"""Design pipeline: Includes the Alphafold2 design part, prefiltering for clashes, followed by ProteinMPNN sequence design. 
All sequences of a design trajectory are predicted as one Batch using Boltz2 with optional covalent constraints. 
The Repredicted Structures are relaxed and scored using PyRosetta and finally filtered based on the Boltz2 and Rosetta scores. 
The design loop continues until the specified number of accepted designs or maximum number of trajectories is reached.
"""
import os
import shutil
import sys
import argparse
import time
import copy
import tempfile
import pandas as pd
import csv
from Bio import PDB
import numpy as np
from functions import *
from pathlib import Path
import yaml

def _safe_remove(path: str, debug: bool = False):
    """Remove a file if it exists; never crash the pipeline due to cleanup."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            if debug:
                print(f"[DEBUG scaffold_pdb] removed tmp_pdb: {path}")
    except Exception as exc:
        if debug:
            print(f"[DEBUG scaffold_pdb] WARNING: could not remove tmp_pdb {path}: {exc}")


def _parse_chain_ids(chain_spec: str):
    if chain_spec is None:
        return []
    return [c.strip() for c in str(chain_spec).split(",") if c.strip()]


def _shift_position_csv(pos_csv: str, offset: int, length: int) -> str:
    """Shift a comma-separated list of 1-based positions by offset, keeping only 1..length."""
    if pos_csv is None:
        return ""
    s = str(pos_csv).strip()
    if not s:
        return ""
    out = []
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            p = int(tok)
        except ValueError:
            continue
        if p <= 0:
            continue
        ps = p + int(offset)
        if 1 <= ps <= int(length):
            out.append(ps)
    out = sorted(set(out))
    return ",".join(str(x) for x in out)


def make_temp_pdb_with_shifted_scaffold(
    *,
    starting_pdb: str,
    target_chains: str,
    binder_chain: str,
    binder_len: int,
    scaffold_indices_1b,
    base_positions_min_1b: int,
    ds_pairs_0b=None,
    seed: int,
    out_dir: str,
    debug: bool = False,
):
    """Create a temporary PDB with a binder chain of length binder_len.

    - The binder is initialized as GLY residues with random-ish backbone coordinates.
    - Scaffold residues are copied (all atoms) from the original binder chain at indices in scaffold_indices_1b
      and inserted into the new binder by applying a constant integer shift.

    Offset logic (per your spec):
      sample u ~ Uniform{1, ..., binder_len - len(scaffold_indices) - 1}
      offset = (u + 1) - base_positions_min_1b
    The +1 keeps 1-based residue indices valid.

    Returns (temp_pdb_path, offset).
    """
    scaffold = sorted({int(x) for x in (scaffold_indices_1b or []) if int(x) > 0})
    if len(scaffold) == 0:
        raise ValueError("scaffold_indices_1b is empty")
    if int(binder_len) <= 0:
        raise ValueError("binder_len must be > 0")
    if int(binder_len) < len(scaffold):
        raise ValueError(
            f"binder_len ({binder_len}) < len(scaffold_indices) ({len(scaffold)})"
        )

    rng = np.random.default_rng(int(seed))
    u_max = int(binder_len) - int(len(scaffold))    
    u = int(rng.integers(1, u_max)) 
    offset = int(u + 1) - int(base_positions_min_1b)    

    # Parse original structure
    parser = PDB.PDBParser(QUIET=True)
    src_structure = parser.get_structure("src", starting_pdb)
    src_model = src_structure[0]

    # Copy target chains as-is
    new_structure = PDB.Structure.Structure("tmp")
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)

    for cid in _parse_chain_ids(target_chains):
        if cid not in src_model:
            raise KeyError(f"Target chain '{cid}' not found in {starting_pdb}")
        new_model.add(copy.deepcopy(src_model[cid]))

    if binder_chain not in src_model:
        raise KeyError(f"Binder chain '{binder_chain}' not found in {starting_pdb}")
    src_binder_chain = src_model[binder_chain]
    src_binder_res = [r for r in src_binder_chain.get_residues() if PDB.is_aa(r, standard=True)]

    # Placeholder GLY chain
    new_binder_chain = PDB.Chain.Chain(binder_chain)

    # Place placeholder backbone near the target centroid to avoid insane distances
    try:
        target_atoms = []
        for cid in _parse_chain_ids(target_chains):
            for atom in src_model[cid].get_atoms():
                if atom.get_id() == "CA":
                    target_atoms.append(atom.get_coord())
        if target_atoms:
            base_xyz = np.mean(np.asarray(target_atoms, dtype=float), axis=0)
        else:
            base_xyz = np.zeros(3, dtype=float)
    except Exception:
        base_xyz = np.zeros(3, dtype=float)

    def _make_placeholder_res(resname: str, resseq_1b: int) -> PDB.Residue.Residue:
        res_id = (" ", int(resseq_1b), " ")
        res = PDB.Residue.Residue(res_id, str(resname), " ")
        # crude, random-ish backbone coordinates
        ca = base_xyz + rng.normal(scale=8.0, size=3) + np.array([resseq_1b * 0.5, 0.0, 0.0])
        n = ca + np.array([-1.2, 0.2, 0.1])
        c = ca + np.array([1.3, -0.2, -0.1])
        o = c + np.array([0.6, -0.6, 0.0])
        res.add(PDB.Atom.Atom("N", n.astype(float), 1.0, 20.0, " ", "N", int(resseq_1b), element="N"))
        res.add(PDB.Atom.Atom("CA", ca.astype(float), 1.0, 20.0, " ", "CA", int(resseq_1b), element="C"))
        res.add(PDB.Atom.Atom("C", c.astype(float), 1.0, 20.0, " ", "C", int(resseq_1b), element="C"))
        res.add(PDB.Atom.Atom("O", o.astype(float), 1.0, 20.0, " ", "O", int(resseq_1b), element="O"))
        return res

    for i in range(1, int(binder_len) + 1):
        new_binder_chain.add(_make_placeholder_res("GLY", i))

    # Insert scaffold residues by shifting indices
    for src_pos_1b in scaffold:
        if src_pos_1b < 1 or src_pos_1b > len(src_binder_res):
            continue
        dst_pos_1b = int(src_pos_1b) + int(offset)
        if not (1 <= dst_pos_1b <= int(binder_len)):
            continue

        src_res = src_binder_res[int(src_pos_1b) - 1]
        new_res = copy.deepcopy(src_res)
        new_res.id = (" ", int(dst_pos_1b), " ")

        # Replace placeholder at dst position
        if new_res.id in new_binder_chain:
            new_binder_chain.detach_child(new_res.id)
        new_binder_chain.add(new_res)

    # Enforce disulfide cysteine identity at specified indices (0-based binder indices)
    # This intentionally overrides any scaffold insertion at those positions.
    ds_positions_1b = set()
    if ds_pairs_0b:
        try:
            for (i0, j0) in ds_pairs_0b:
                for idx0 in (i0, j0):
                    idx0i = int(idx0)
                    if idx0i < 0:
                        continue
                    ds_positions_1b.add(idx0i + 1)
        except Exception:
            ds_positions_1b = set()
    for p1 in sorted(ds_positions_1b):
        if 1 <= int(p1) <= int(binder_len):
            rid = (" ", int(p1), " ")
            if rid in new_binder_chain:
                new_binder_chain.detach_child(rid)
            new_binder_chain.add(_make_placeholder_res("CYS", int(p1)))

    # Ensure binder residues are ordered by resseq
    try:
        new_binder_chain.child_list.sort(key=lambda r: int(r.id[1]))
    except Exception:
        pass

    new_model.add(new_binder_chain)

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f"tmp_scaffold_{int(seed)}_l{int(binder_len)}_off{int(offset)}.pdb")
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(tmp_path)

    if debug:
        print("[DEBUG scaffold_pdb] tmp_path:", tmp_path)
        print("[DEBUG scaffold_pdb] scaffold_len:", len(scaffold))
        print("[DEBUG scaffold_pdb] base_positions_min_1b:", int(base_positions_min_1b))
        print("[DEBUG scaffold_pdb] sampled_u:", int(u))
        print("[DEBUG scaffold_pdb] offset:", int(offset))
        if ds_positions_1b:
            print("[DEBUG scaffold_pdb] disulfide CYS positions (1b):", sorted(ds_positions_1b))

    return tmp_path, offset


def extract_chain_sequence_from_pdb(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in {pdb_path}")
    chain = model[chain_id]
    seq = []
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        seq.append(three_to_one_map.get(residue.get_resname(), 'X'))
    return ''.join(seq)


def main():
    # Ensure GPU is available (halts otherwise)
    check_jax_gpu()

    parser = argparse.ArgumentParser(description="BindCraft design stage (AF2 + optional MPNN)")
    parser.add_argument("--settings", "-s", required=True, help="Path to basic settings.json")
    parser.add_argument("--filters", "-f", default="./settings_filters/default_filters.json", help="Path to filters.json")
    parser.add_argument("--advanced", "-a", default="./settings_advanced/default_4stage_multimer.json", help="Path to advanced settings json")
    parser.add_argument("--prefilters", "-p", default=None, help="Optional prefilters json for trajectory pre-filtering")
    args = parser.parse_args()

    settings_path, filters_path, advanced_path, prefilters_path = perform_input_check(args)
    target_settings, advanced_settings, filters, prefilters = load_json_settings(settings_path, filters_path, advanced_path, prefilters_path)

    debug = target_settings.get("debug_mode", False)

    # Preserve unshifted base specs for binder_advanced so runtime shifting doesn't compound across trajectories.
    # Prefer specs provided in settings.json; fall back to advanced.json for backward compatibility.
    if target_settings.get("protocol", "binder") == "binder_advanced":
        scaffold_indices = []
        for key in ("fixed_positions", "template_positions", "sequence_positions"):
            base_key = f"{key}_base"
            # Always prefer the value from settings.json when present; else fall back to advanced.json
            base_val = target_settings.get(key, advanced_settings.get(key))
            advanced_settings[base_key] = base_val
            advanced_settings[key] = copy.deepcopy(base_val)
            # Track global min/max across all *_base position specs 
            def _to_int_list(v):
                if v is None:
                    return []
                if isinstance(v, str):
                    v = v.strip()
                    if not v:
                        return []
                    return [int(x) for x in v.replace(" ", "").split(",") if x]
                if isinstance(v, (list, tuple, set)):
                    return [int(x) for x in v]
                return [int(v)]

            _vals = _to_int_list(base_val)
            scaffold_indices.extend(_vals)
            if _vals:
                cur_min = advanced_settings.get("base_positions_min")
                cur_max = advanced_settings.get("base_positions_max")
                advanced_settings["base_positions_min"] = min(_vals) if cur_min is None else min(cur_min, min(_vals))
                advanced_settings["base_positions_max"] = max(_vals) if cur_max is None else max(cur_max, max(_vals))
        advanced_settings["scaffold_indices"] = sorted(set(scaffold_indices))
        if debug:
            print("Overall base_positions_min:", advanced_settings["base_positions_min"])
            print("Overall base_positions_max:", advanced_settings["base_positions_max"])
        # Prefer disulfide_num from settings.json; fallback to advanced.json, default 1
        advanced_settings["disulfide_num"] = int(target_settings.get("disulfide_num", advanced_settings.get("disulfide_num", 1)))


    settings_file = os.path.basename(settings_path).split(".")[0]
    filters_file = os.path.basename(filters_path).split(".")[0]
    advanced_file = os.path.basename(advanced_path).split(".")[0]
    prefilters_file = os.path.basename(prefilters_path).split(".")[0] if prefilters_path else None

    design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])
    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindcraft_folder)

    design_paths = generate_directories_isolated(target_settings["design_path"])
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(target_settings["design_path"], "trajectory_stats.csv")
    # mpnn_csv = os.path.join(target_settings["design_path"], "mpnn_design_stats.csv")
    # AF2_design_csv = os.path.join(target_settings["design_path"], "AF2_design_stats.csv")
    # final_csv = os.path.join(target_settings["design_path"], "final_design_stats.csv")
    failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

    create_dataframe(trajectory_csv, trajectory_labels)
    # create_dataframe(mpnn_csv, design_labels)
    # create_dataframe(AF2_design_csv, design_labels)
    # create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    print(f"Running binder design for target {settings_file}")
    print(f"Design settings used: {settings_file}")
    # print(f"Filtering designs based on {filters_file}")
    # if prefilters_file is not None and advanced_settings.get("pre_filter_trajectory", False):
    #     print(f"Pre-filtering trajectories based on {prefilters_file}")
    # elif prefilters_file is None and advanced_settings.get("pre_filter_trajectory", False):
    #     print("-------------------------------------------------------------------------------------")
    #     print("Warning: pre_filter_trajectory is enabled but no prefilters file provided, exiting...")
    #     print("-------------------------------------------------------------------------------------")
    #     sys.exit(1)

    script_start_time = time.time()
    trajectory_n = 1


    design_path = Path(target_settings["design_path"])
    

    # Optional starting PDB helix check
    if advanced_settings.get("initial_helix_check_cutoff", 100.0) < 100.0:
        starting_pdb = target_settings["starting_pdb"]
        binder_chain_id = "B"
        helix_pct, sheet_pct, loop_pct, *_ = calc_ss_percentage(starting_pdb, advanced_settings, chain_id=binder_chain_id, sec_struct_only=True)
        if helix_pct > advanced_settings["initial_helix_check_cutoff"]:
            print("-------------------------------------------------------------------------------------")
            print(f"ERROR: Helical secondary structure detected in binder chain {binder_chain_id} ({helix_pct}% helix)")
            print("The binder_advanced protocol does not support helical starting structures.")
            print(f"Starting PDB: {starting_pdb}")
            print("Program terminated.")
            print("-------------------------------------------------------------------------------------")
            sys.exit(1)
        print(f"Starting binder secondary structure check passed: {sheet_pct}% sheet, {loop_pct}% loop")




# Begin of the Design Loop


    while True:

        #Check if number of accepted designs or number of trajectories are reached

        path = os.path.join(design_path, 'Accepted')
        
        if os.path.exists(path):
            n_accepted_designs = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
            
            if n_accepted_designs >= target_settings.get('number_of_final_designs', 100):
                print(f"Target number of accepted designs reached! Found {n_accepted_designs} accepted designs. Stopping...")
                break
        #Check if the maximum number of trajectories is reached

        if check_n_trajectories(design_paths, advanced_settings):
            break

        trajectory_start_time = time.time()
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        def _normalize_disulfide_pairs_spec(spec):
            """Normalize disulfide_pairs to a list of (i, j) pairs (0-based, binder-local).

            Accepts:
            - [[0, -1]] / [(0, -1)]  -> one pair
            - [0, -1]                -> one pair
            - [0, -1, 3, 7]          -> paired sequentially
            """
            if spec is None:
                return []

            if isinstance(spec, (list, tuple)) and len(spec) == 0:
                return []

            # Single pair like [0, -1]
            if (
                isinstance(spec, (list, tuple))
                and len(spec) == 2
                and not any(isinstance(x, (list, tuple)) for x in spec)
            ):
                return [(spec[0], spec[1])]

            # List of pairs
            if (
                isinstance(spec, (list, tuple))
                and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in spec)
            ):
                return [(x[0], x[1]) for x in spec]

            # Flat list -> pair it
            if isinstance(spec, (list, tuple)) and all(not isinstance(x, (list, tuple)) for x in spec):
                if len(spec) % 2 != 0:
                    return []
                out = []
                for k in range(0, len(spec), 2):
                    out.append((spec[k], spec[k + 1]))
                return out

            return []

        if target_settings.get("protocol", "binder") == "binder_advanced":
            required_min_length = len(advanced_settings["scaffold_indices"])
            if debug:
                print("Required minimum binder length based on base positions:", required_min_length)   
            if "lengths" in target_settings and target_settings["lengths"]:
                min_length = min(target_settings["lengths"]) if min(target_settings["lengths"]) > required_min_length else required_min_length
                max_length = max(target_settings["lengths"]) if max(target_settings["lengths"]) > min_length else min_length
                samples = np.arange(min_length, max_length + 1)
                length = int(np.random.choice(samples))
            else:
                # Fallback to the starting binder chain length if no sampling range was provided
                length = get_chain_length(target_settings["starting_pdb"], "B")
            if length < required_min_length:
                length = int(required_min_length)
        elif target_settings.get("protocol", "binder") == "binder":
            min_length = min(target_settings["lengths"])
            max_length = max(target_settings["lengths"]) if max(target_settings["lengths"]) > min_length else min_length
            samples = np.arange(min_length, max_length + 1)
            length = np.random.choice(samples)
        else:
            print(f"Invalid protocol specified in settings.json, {target_settings.get('protocol', 'None')}, exiting...")
            sys.exit(1)
        if debug:
            print("Selected binder length for this trajectory:", length)

        # if target settings contains max_trajectories, override advanced setting
        if "max_trajectories" in target_settings:
            advanced_settings["max_trajectories"] = int(target_settings["max_trajectories"])

        advanced_settings["binder_len_runtime"] = int(length)

        ds_pairs = []
        # Prefer disulfide specs from settings.json; fall back to advanced.json for compatibility
        ds_pairs_pre_raw = target_settings.get("disulfide_pairs", advanced_settings.get("disulfide_pairs"))
        ds_pairs_pre = _normalize_disulfide_pairs_spec(ds_pairs_pre_raw)
        if ds_pairs_pre_raw and not ds_pairs_pre:
            print(f"Warning: invalid disulfide_pairs format in settings; ignoring: {ds_pairs_pre_raw}")
        if ds_pairs_pre:
            valid = True
            seen = set()
            min_sep = int(advanced_settings.get("disulfide_min_sep", 5))

            def _norm_ds_idx(idx, L):
                idx_int = int(idx)
                if idx_int < 0:
                    idx_int = L + idx_int
                return idx_int

            try:
                for (i_raw, j_raw) in ds_pairs_pre:
                    i = _norm_ds_idx(i_raw, length)
                    j = _norm_ds_idx(j_raw, length)
                    if i == j or i < 0 or j < 0 or i >= length or j >= length:
                        valid = False
                        break
                    if abs(i - j) < min_sep:
                        valid = False
                        break
                    if i in seen or j in seen:
                        valid = False
                        break
                    seen.add(i)
                    seen.add(j)
                    ds_pairs.append((i, j))
            except Exception:
                valid = False

            if not valid:
                ds_pairs = []

        # === binder_advanced: build a temporary PDB with shifted scaffold coordinates ===
        # This implements scaffold shifting by actually moving the scaffold residues in the template binder chain,
        # rather than only shifting the index bookkeeping.
        starting_pdb_runtime = target_settings["starting_pdb"]
         # Track per-trajectory temp PDB (cleanup later if debug is False)
        tmp_pdb_path = ""
        if target_settings.get("protocol", "binder") == "binder_advanced":
            shuffle_scaffold = bool(
                target_settings.get(
                    "shuffle_scaffold_positions",
                    target_settings.get("shift_motif_positions", False),
                )
            )
            use_template_binder = bool(target_settings.get("use_template_binder", False))
            binder_chain_id = target_settings.get("binder_chain", "B")

            scaffold_indices = advanced_settings.get("scaffold_indices", [])
            base_min_1b = advanced_settings.get("base_positions_min")

            if shuffle_scaffold and use_template_binder and scaffold_indices and base_min_1b is not None:
                tmp_dir = os.path.join(target_settings["design_path"], "tmp_pdbs")
                tmp_pdb, offset = make_temp_pdb_with_shifted_scaffold(
                    starting_pdb=target_settings["starting_pdb"],
                    target_chains=target_settings.get("chains", "A"),
                    binder_chain=binder_chain_id,
                    binder_len=int(length),
                    scaffold_indices_1b=scaffold_indices,
                    base_positions_min_1b=int(base_min_1b),
                    ds_pairs_0b=ds_pairs,
                    seed=int(seed),
                    out_dir=tmp_dir,
                    debug=bool(debug),
                )
                starting_pdb_runtime = tmp_pdb
                tmp_pdb_path = tmp_pdb

                # Shift position specs to match the moved scaffold in the temp PDB.
                advanced_settings["fixed_positions"] = _shift_position_csv(
                    advanced_settings.get("fixed_positions_base"), offset, int(length)
                )
                advanced_settings["template_positions"] = _shift_position_csv(
                    advanced_settings.get("template_positions_base"), offset, int(length)
                )
                advanced_settings["sequence_positions"] = _shift_position_csv(
                    advanced_settings.get("sequence_positions_base"), offset, int(length)
                )
                advanced_settings["positions_shift_offset_runtime"] = int(offset)
                if debug:
                    print("[DEBUG scaffold_pdb] shifted fixed_positions:", advanced_settings.get("fixed_positions"))
                    print("[DEBUG scaffold_pdb] shifted template_positions:", advanced_settings.get("template_positions"))
                    print("[DEBUG scaffold_pdb] shifted sequence_positions:", advanced_settings.get("sequence_positions"))
        helicity_value = load_helicity(advanced_settings)

        design_name = f"{target_settings['binder_name']}_l{length}_s{seed}"
        trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
        trajectory_exists = any(os.path.exists(os.path.join(design_paths[td], design_name + ".pdb")) for td in trajectory_dirs)

        if trajectory_exists:
            if not debug:
                _safe_remove(tmp_pdb_path, debug=False)
            trajectory_n += 1
            continue

        # AF2 design trajectory

        print(f"Starting trajectory: {design_name}")
        trajectory = binder_hallucination(design_name, starting_pdb_runtime, target_settings.get("protocol", "binder"), target_settings["chains"],
                                           target_settings.get("target_hotspot_residues", None), target_settings.get("pos", None), length, seed, ds_pairs, helicity_value,
                                           design_models, advanced_settings, design_paths, failure_csv)
        trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}
            # Delete temp PDB right after AF design prep/run, unless debugging
        if not debug:
            _safe_remove(tmp_pdb_path, debug=False)

        # Save the (target+binder) trajectory sequence as FASTA alongside MPNN FASTAs
        try:
            trajectory_fastas_dir = os.path.join(target_settings["design_path"], "trajectory_fastas")
            os.makedirs(trajectory_fastas_dir, exist_ok=True)
            target_chain_ids = [c.strip() for c in str(target_settings.get("chains", "A")).split(',') if c.strip()]
            binder_chain_id = target_settings.get("binder_chain", "B")
            chain_ids_in_order = target_chain_ids + [binder_chain_id]
            full_seq = "/".join(extract_chain_sequence_from_pdb(trajectory_pdb, c) for c in chain_ids_in_order)
            fasta_name = os.path.join(trajectory_fastas_dir, f"{design_name}_complete.fasta")
            with open(fasta_name, "w") as fh:
                fh.write(f">{design_name}\n{full_seq}\n")
        except Exception as exc:
            print(f"Warning: could not write trajectory FASTA for {design_name}: {exc}")

        trajectory_time = time.time() - trajectory_start_time
        trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
        print(f"Starting trajectory took: {trajectory_time_text}\n")

        if trajectory.aux["log"].get("terminate", "") == "":
            trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
            pr_relax(trajectory_pdb, trajectory_relaxed)
            binder_chain = "B"

            num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
            num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)
            (trajectory_alpha, trajectory_beta, trajectory_loops,
             trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface,
             trajectory_i_plddt, trajectory_ss_plddt) = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

            trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(trajectory_relaxed, binder_chain)
            trajectory_sequence = trajectory.get_seq(get_best=True)[0]
            traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)
            trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

            trajectory_data = [design_name, advanced_settings["design_algorithm"], length, seed, helicity_value, target_settings.get("target_hotspot_residues", ""), trajectory_sequence, trajectory_interface_residues,
                               trajectory_metrics.get('plddt'), trajectory_metrics.get('ptm'), trajectory_metrics.get('i_ptm'), trajectory_metrics.get('pae'), trajectory_metrics.get('i_pae'),
                               trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed, trajectory_interface_scores['binder_score'],
                               trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'], trajectory_interface_scores['interface_packstat'],
                               trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'], trajectory_interface_scores['interface_dG_SASA_ratio'],
                               trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'], trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
                               trajectory_interface_scores['interface_hbond_percentage'], trajectory_interface_scores['interface_delta_unsat_hbonds'], trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                               trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd,
                               trajectory_time_text, traj_seq_notes, settings_file, filters_file, advanced_file]
            insert_data(trajectory_csv, trajectory_data)

            if debug:
                print(f"fixed_positions (runtime): {advanced_settings.get('fixed_positions', None)}")
                print(f"sequence_positions (runtime): {advanced_settings.get('sequence_positions', None)}")
                print(f"trajectory_interface_residues: {trajectory_interface_residues}")
                print(f"disulfide_pairs (0-based): {ds_pairs}")

            # Build fixed MPNN scaffold positions in the same "B{resnum}" format as trajectory_interface_residues
            def _parse_pos_list(v):
                if v is None:
                    return set()
                if isinstance(v, (list, tuple, set)):
                    return {int(x) for x in v}
                if isinstance(v, str):
                    v = v.strip()
                    if not v:
                        return set()
                    return {int(x) for x in v.replace(" ", "").split(",") if x}
                return {int(v)}

            fixed_positions_set = _parse_pos_list(advanced_settings.get("fixed_positions"))
            sequence_positions_set = _parse_pos_list(advanced_settings.get("sequence_positions"))
            ds_positions_set = {idx + 1 for (i, j) in ds_pairs for idx in (i, j)}  # ds_pairs are 0-based -> convert to 1-based

            fixed_mpnn_scaffold = {
                f"{binder_chain}{pos}"
                for pos in (fixed_positions_set | sequence_positions_set | ds_positions_set)
            }

            if debug:
                print(f"fixed_mpnn_scaffold: {fixed_mpnn_scaffold}")

            if not trajectory_interface_residues:
                print(f"No interface residues found for {design_name}, skipping MPNN optimization")

            elif advanced_settings["enable_mpnn"]:
                mpnn_n = 1
                accepted_mpnn = 0
                design_start_time = time.time()

                mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, fixed_mpnn_scaffold, advanced_settings)

                # Save complete (target+binder) sequences for all sampled MPNN outputs
                mpnn_complete_dir = os.path.join(target_settings["design_path"], "trajectory_fastas")
                os.makedirs(mpnn_complete_dir, exist_ok=True)
                for n in range(advanced_settings["num_seqs"]):
                    full_seq = mpnn_trajectories['seq'][n]
                    fasta_name = os.path.join(mpnn_complete_dir, f"{design_name}_mpnn{n+1}_complete.fasta")
                    with open(fasta_name, "w") as fh:
                        fh.write(f">{design_name}_mpnn{n+1}\n{full_seq}\n")

                # existing_mpnn_sequences = set(pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values)
                restricted_AAs = set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(',')) if advanced_settings.get("force_reject_AA") else set()

                mpnn_sequences = sorted({
                    mpnn_trajectories['seq'][n][-length:]: {
                        'seq': mpnn_trajectories['seq'][n][-length:],
                        'score': mpnn_trajectories['score'][n],
                        'seqid': mpnn_trajectories['seqid'][n]
                    } for n in range(advanced_settings["num_seqs"])
                    if (not restricted_AAs or not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
                    and mpnn_trajectories['seq'][n][-length:] #not in existing_mpnn_sequences
                }.values(), key=lambda x: x['score'])
                # del existing_mpnn_sequences

                # if mpnn_sequences:
                #     if advanced_settings["save_mpnn_fasta"] is True:
                #         save_fasta(design_name, mpnn_trajectories['seq'][n][-length:], design_paths)
                #     if advanced_settings.get("optimise_beta") and float(trajectory_beta) > 15:
                #         advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]

                #     clear_mem()
                #     complex_prediction_model, binder_prediction_model = init_prediction_models(
                #         trajectory_pdb=trajectory_pdb,
                #         length=length,
                #         mk_afdesign_model=mk_afdesign_model,
                #         multimer_validation=multimer_validation,
                #         target_settings=target_settings,
                #         advanced_settings=advanced_settings
                #     )

                #     for mpnn_sequence in mpnn_sequences:
                #         filter_conditions, mpnn_csv_out, failure_csv_out, final_csv_out = filter_design(
                #             sequence=mpnn_sequence,
                #             basis_design_name=design_name,
                #             design_paths=design_paths,
                #             trajectory_pdb=trajectory_pdb,
                #             length=length,
                #             helicity_value=helicity_value,
                #             seed=seed,
                #             prediction_models=prediction_models,
                #             binder_chain=binder_chain,
                #             filters=filters,
                #             design_labels=design_labels,
                #             target_settings=target_settings,
                #             advanced_settings=advanced_settings,
                #             advanced_file=advanced_file,
                #             settings_file=settings_file,
                #             filters_file=filters_file,
                #             stats_csv=mpnn_csv,
                #             failure_csv=failure_csv,
                #             final_csv=final_csv,
                #             complex_prediction_model=complex_prediction_model,
                #             binder_prediction_model=binder_prediction_model,
                #             is_mpnn_model=True,
                #             mpnn_n=mpnn_n
                #         )
                #         mpnn_n += 1
                #         if filter_conditions is True:
                #             accepted_mpnn += 1
                #         if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                #             break

                #     design_time = time.time() - design_start_time
                #     design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                #     print(f"MPNN redesign for trajectory {design_name} took: {design_time_text}\n")
                # else:
                #     print('No unique MPNN sequences after filtering omit_AAs/duplicates')

            # acceptance monitoring
            # if trajectory_n >= advanced_settings.get("start_monitoring", 1000) and advanced_settings.get("enable_rejection_check", False):
            #     acceptance = accepted_designs / trajectory_n if trajectory_n else 0
            #     if acceptance < advanced_settings.get("acceptance_rate", 0):
            #         print("Acceptance rate below threshold, stopping early")
            #         break



    #check if a trajectory has a succesful design, if yes run boltz reprediction on that design
        if not trajectory.aux["log"].get("terminate", "") == "":
            print(f"Starting new Trajectory")
            continue


   #  Setup paths correctly using Pathlib --> one dir for each trajectory so boltz is loaded only one per tracetory
        
        fasta_dir = design_path / "trajectory_fastas"
        boltz_out_dir = design_path / "boltz_reprediction" / design_name
        boltz_templates_dir = design_path / "boltz_reprediction" / "templates"
        boltz_out_dir.mkdir(parents=True, exist_ok=True)
        boltz_templates_dir.mkdir(parents=True, exist_ok=True)


        # Initialize variables
        auto_disulfide = (target_settings.get('disulfide_num') == 1)
        fasta_found_for_design = False

        design_struct = design_path / "Trajectory" / f"{design_name}.pdb"
        # 2. Check for FASTAs
        if fasta_dir.exists():
            fasta_files = list(fasta_dir.glob('*.fasta'))
            for fasta_path in fasta_files:
                if design_name in fasta_path.name:
                    fasta_found_for_design = True
                    
                    # Read the file content to get sequences
                    with open(fasta_path, 'r') as f:
                        # Skip header, join lines, split by '/'
                        content = "".join([line.strip() for line in f if not line.startswith(">")])
                        
                        if '/' in content:
                            target_seq, binder_seq = content.split('/')
                            
                            yaml_name = fasta_path.stem + ".yaml"
                            create_boltz_yaml(boltz_out_dir / yaml_name, 
                                                            target_seq, 
                                                            binder_seq, 
                                                            auto_disulfide, 
                                                            False, 
                                                            target_settings.get("use_template", False), 
                                                            design_struct, 
                                                            boltz_templates_dir,
                                                            target_settings.get("force", False),
                                                            target_settings.get("threshold", 4.0)
                                            )                        
        # 3. Fallback to PDB if no FASTAs were matched for this specific design
        if not fasta_found_for_design:
            design_struct = design_path / "Trajectory" / f"{design_name}.pdb"
            
            if design_struct.exists():
                target_seq = extract_chain_sequence_from_pdb(str(design_struct), "A")
                binder_seq = extract_chain_sequence_from_pdb(str(design_struct), "B")
                
                yaml_name = f"{design_name}.yaml"

                create_boltz_yaml(boltz_out_dir / yaml_name, 
                                  target_seq, 
                                  binder_seq, 
                                  auto_disulfide, 
                                  False, 
                                  target_settings.get("use_template", False), 
                                  design_struct, 
                                  boltz_templates_dir,
                                  target_settings.get("force", False),
                                  target_settings.get("threshold", 4.0)
                )
        # Run Boltz reprediction for this design and all mpnn sequences in one batch
        conda_env = target_settings.get("boltz_conda_env", "boltz")
        print(f'Running Boltz repredictions for design: {design_name}')
        run_boltz(design_path / "boltz_reprediction" / design_name, design_path / "boltz_reprediction" / 'predictions', conda_env)
        
       
        print(f'Reprediction finished. Parsing results for: {design_name}')
        

        parse_results( 
            design_path / "boltz_reprediction" / "predictions" / f"boltz_results_{design_name}", 
            design_path / "boltz_repredicted_stats.csv", 
            design_path / "Repredicted",
            extract_chain_sequence_from_pdb
        )
        
        #Cleanup Boltz templates to save space (optional)
        clean_up = True
        dir_to_remove = design_path / "boltz_reprediction" 
        if clean_up :
            shutil.rmtree(dir_to_remove)


        #Boltz Reprediction finished --> PyRosetta Relaxation and Scoring of Boltz outputs

        # --- PyRosetta Initialization ---
        _has_is_init = hasattr(pr, "is_initialized")
        try:
            if not _has_is_init or not pr.is_initialized():
                pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
        except AttributeError:
            pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

        # --- Relaxation Phase ---
        # Collecting all unrelaxed PDBs for this design's Boltz reprediction outputs based on naming convention, to relax them in batch. 
        unrelaxed_pdbs = list((design_path / "Repredicted").glob(f"*{design_name}*.pdb"))
        print(f"Found {len(unrelaxed_pdbs)} unrelaxed complex PDB(s) for Trajectory {design_name} to relax")
        
        disulfide_enabled = (target_settings.get('disulfide_num') == 1)
        relaxed_dir = design_path / 'Repredicted' / "Relaxed"
        relaxed_dir.mkdir(parents=True, exist_ok=True)

        for unrelaxed_pdb in unrelaxed_pdbs:
            design_base, model_num = parse_design_and_model(unrelaxed_pdb)
            out_relaxed = relaxed_dir / f"{design_base}_model{model_num}.pdb"
            
            if out_relaxed.exists():
                continue
                
            print(f"Relaxing {unrelaxed_pdb.name} -> {out_relaxed.name}")
            
            pairs = None
            disulfide_for_this = False
            if disulfide_enabled:
                try:
                    binder_seq = extract_chain_sequence_from_pdb(str(unrelaxed_pdb), binder_chain)
                    pairs = infer_single_disulfide_pair_from_binder_sequence(binder_seq)
                    if pairs:
                        disulfide_for_this = True
                except Exception as exc:
                    print(f"Warning: skipping relaxation for {unrelaxed_pdb.name} due to disulfide error: {exc}")
                    continue

            try:
                pr_relax(
                    str(unrelaxed_pdb),
                    str(out_relaxed),
                    disulfide=disulfide_for_this,
                    binder_chain=binder_chain,
                    binder_local_pairs=pairs,
                )
            except Exception as exc:
                print(f"Warning: skipping relaxation for {unrelaxed_pdb.name}: {exc}")
                continue

        # --- Scoring Phase ---
        #Collect all relaxed PDBs for this trajectory 

        relaxed_pdbs = list(relaxed_dir.glob(f"*{design_name}*.pdb"))
        if not relaxed_pdbs:
            print(f"No relaxed PDBs found for {design_name}; skipping scoring.")
            continue

        # Map models to paths for iteration





        model_map = {}
        for rp in relaxed_pdbs:
            _, m_num = parse_design_and_model(rp)
            model_map[m_num] = rp
        
        model_nums = sorted(model_map.keys())
        print(f"Scoring {design_name} from {len(model_nums)} model PDB(s)")

        helicity_value = load_helicity(advanced_settings)
        
        try:
            # Use the first available model to get the primary sequence
            primary_pdb = model_map[model_nums[0]]
            binder_seq = extract_chain_sequence_from_pdb(str(primary_pdb), binder_chain)
        except Exception as exc:
            print(f"Skipping {design_name} sequence extraction: {exc}")
            continue

        length = len(binder_seq)
        trajectory_pdb = design_path / "Trajectory" / f"{design_name}.pdb"
        if not trajectory_pdb.exists(): trajectory_pdb = None

        complex_statistics = {}
        interface_residues_for_row = ""
        

        
        df = pd.read_csv(str(design_path/"boltz_repredicted_stats.csv"))

        #Iterate over the just relaxed PDBs to calculate scores and compile statistics for each model, then merge with the original design metrics and write to final CSV
        for pdb_path  in relaxed_pdbs:

           
            print ('the for loop begins here!')
            print('An the relexed pdbs are:', relaxed_pdbs)
            try:
                
                unrelaxed_candidate = design_path / "Repredicted" / pdb_path.name
                
                num_clashes_unrelaxed = calculate_clash_score(str(unrelaxed_candidate)) if unrelaxed_candidate.exists() else None
                num_clashes_relaxed = calculate_clash_score(str(pdb_path))
                
                interface_scores, interface_AA, interface_residues = score_interface(str(pdb_path), binder_chain)
                alpha, beta, loops, alpha_interface, beta_interface, loops_interface, i_plddt, ss_plddt = calc_ss_percentage(str(pdb_path), advanced_settings, binder_chain)
                
                target_rmsd = target_pdb_rmsd(str(pdb_path), target_settings["starting_pdb"], target_settings["chains"])
                hotspot_rmsd = unaligned_rmsd(str(trajectory_pdb), str(pdb_path), binder_chain, binder_chain) if trajectory_pdb else None

                if not interface_residues_for_row:
                    interface_residues_for_row = interface_residues

                
                #Extract the Boltz metrics from the currently scored design
                row_dict = df[df['Design'].str.contains(pdb_path.stem.replace('_model1',''))].iloc[0]


                print (f"boltzvalues for this design {str(df['Design'].iloc[0])} are: {row_dict}")
                
                complex_statistics = {
                    
                    
                    'Unrelaxed_Clashes': num_clashes_unrelaxed,
                    'Relaxed_Clashes': num_clashes_relaxed,
                    'Binder_Energy_Score': interface_scores['binder_score'],
                    'Surface_Hydrophobicity': interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': interface_scores['interface_sc'],
                    'PackStat': interface_scores['interface_packstat'],
                    'dG': interface_scores['interface_dG'],
                    'dSASA': interface_scores['interface_dSASA'],
                    'dG/dSASA': interface_scores['interface_dG_SASA_ratio'],
                    'Interface_SASA_%': interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': interface_scores['interface_hydrophobicity'],
                    'n_InterfaceResidues': interface_scores['interface_nres'],
                    'n_InterfaceHbonds': interface_scores['interface_interface_hbonds'],
                    'InterfaceHbondsPercentage': interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': interface_scores['interface_delta_unsat_hbonds'],
                    'InterfaceUnsatHbondsPercentage': interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': interface_AA,
                    'Interface_Helix%': alpha_interface,
                    'Interface_BetaSheet%': beta_interface,
                    'Interface_Loop%': loops_interface,
                    'Binder_Helix%': alpha,
                    'Binder_BetaSheet%': beta,
                    'Binder_Loop%': loops,
                    'Hotspot_RMSD': hotspot_rmsd,
                    'Target_RMSD': target_rmsd,
                    'Algorithm': advanced_settings.get("design_algorithm"),
                    'Lenght':len(row_dict['binder_sequence']),
                    'Target_Hotspots':target_settings.get("target_hotspot_residues", "")

                }
                complex_statistics

               

                combined_row = {**row_dict, **complex_statistics}
                print (f'This is the combined metric for {pdb_path.name}: {combined_row}')
               




                
                # Convert the interface dict  to a string so it doesn't break the formatting.
                if 'InterfaceAAs' in combined_row:
                    combined_row['InterfaceAAs'] = str(combined_row['InterfaceAAs'])

                # Write resutls
                output_path = f"{design_path}/final_results.csv"
                file_exists = os.path.isfile(output_path)

                with open(output_path, 'a', newline='') as f:
                    # We use the keys of our combined dictionary as the header
                    writer = csv.DictWriter(f, fieldnames=combined_row.keys())
                    
                    # Write the header only if we are creating the file for the first time
                    if not file_exists:
                        writer.writeheader()
                        
                    writer.writerow(combined_row)


                print(f"Successfully merged and appended stats for {design_name} to {output_path}")
                print (f"This is the complex statistics for {pdb_path}: ", complex_statistics)

            except Exception as exc:
                print(f"Error scoring {design_name} model {model_num}: {exc}")
                continue




            #Starting the filtering and folder organization for this model's results
            metric_filter = list(filters.keys())

             #Filter this exact row:
            passed = True
            failed_filters_list = []
            try:
                for metric in metric_filter:
                    if metric not in combined_row:
                        continue
                        
                    threshold_val = filters[metric].get("threshold")
                    higher_is_better = filters[metric].get("higher")

                    if threshold_val is None or threshold_val == 'null':
                        continue

                    current_val = combined_row[metric]
                    
                    # Logic: If higher is better, but current is lower than threshold -> FAIL
                    if higher_is_better:
                        if current_val < threshold_val:
                            passed = False
                            failed_filters_list.append(f"{metric}({current_val}<{threshold_val})")
                    # Logic: If lower is better, but current is higher than threshold -> FAIL
                    else:
                        if current_val > threshold_val:
                            passed = False
                            failed_filters_list.append(f"{metric}({current_val}>{threshold_val})")

            except Exception as e:
                print(f"Filter Error: {e}")

            # Add the failure reasons to the row data for the folders
            combined_row['Failed_Filters'] = ", ".join(failed_filters_list) if not passed else ""

            # Write the results in the Accepted or failed dir for later analysis
            status_folder = 'Accepted' if passed else 'Failed'
            target_dir = design_path / status_folder
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy the PDB
            shutil.copy(pdb_path, target_dir)

            # Write the variant-specific CSV line into the folder
            # (e.g., peptide_l14_s306443/Accepted/accepted_metrics.csv)
            status_csv_path = target_dir / f"{status_folder.lower()}_metrics.csv"
            status_file_exists = status_csv_path.exists() and status_csv_path.stat().st_size > 0

            with open(status_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=combined_row.keys())
                if not status_file_exists:
                    writer.writeheader()
                writer.writerow(combined_row)

            if passed:
                print(f"Design {pdb_path.name} PASSED and saved to {status_folder}")
            else:
                print(f" Design {pdb_path.name} FAILED: {failed_filters_list}")

                
        # --- Cleanup ---
        trajectory_n += 1
        gc.collect()

        # --> next design trajectory starts. 

    

if __name__ == "__main__":
    main()
