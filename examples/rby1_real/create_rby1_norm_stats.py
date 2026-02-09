"""Generate quantile normalization stats for RBY1 (16-dim state/action).

Creates norm_stats for OpenPI policy server from H5 trajectory data.

Usage:
    python create_rby1_norm_stats.py --data-dir /path/to/h5_files --output-dir assets/rby1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import h5py
import numpy as np

try:
    from openpi.shared import normalize
except ImportError:
    print("Error: openpi package not found. Activate openpi conda environment.")
    sys.exit(1)


def load_h5_data(h5_files: List[Path], include_gripper: bool = True) -> Dict[str, np.ndarray]:
    """Load state and action data from H5 files."""
    states = []
    actions = []
    
    print(f"Loading {len(h5_files)} H5 files...")
    
    for h5_file in h5_files:
        print(f"  {h5_file.name}...", end=" ")
        try:
            with h5py.File(h5_file, 'r') as f:
                # Load robot position (14 joints)
                if 'robot_position' not in f:
                    print("missing 'robot_position', skipped")
                    continue
                robot_pos = f['robot_position'][:]
                
                # Append gripper state if requested (2 values)
                if include_gripper and 'gripper_state' in f:
                    gripper_state = f['gripper_state'][:]
                    state = np.concatenate([robot_pos, gripper_state], axis=-1)
                else:
                    state = robot_pos
                    if include_gripper:
                        print("(no gripper)", end=" ")
                
                states.append(state)
                
                # Load actions
                action_key = 'target_position' if 'target_position' in f else 'action'
                if action_key in f:
                    robot_action = f[action_key][:]
                    
                    if include_gripper and 'gripper_target' in f:
                        gripper_action = f['gripper_target'][:]
                        action = np.concatenate([robot_action, gripper_action], axis=-1)
                    else:
                        action = robot_action
                    
                    actions.append(action)
                    print(f"✓ ({state.shape[0]} steps)")
                else:
                    print(f"no actions")
                    
        except Exception as e:
            print(f"error: {e}")
            continue
    
    if not states:
        raise ValueError("No valid data found in H5 files")
    
    all_states = np.concatenate(states, axis=0)
    all_actions = np.concatenate(actions, axis=0) if actions else None
    
    print(f"\nTotal: {all_states.shape[0]} samples")
    print(f"  State dim: {all_states.shape[1]}")
    if all_actions is not None:
        print(f"  Action dim: {all_actions.shape[1]}")
    
    return {
        "states": all_states,
        "actions": all_actions,
    }


def compute_normalization_stats(data: np.ndarray, name: str) -> Any:
    """Compute quantile normalization statistics."""
    print(f"\nComputing {name} stats (dim={data.shape[-1]})...")
    
    accumulator = normalize.Accumulator(vector_length=data.shape[-1])
    for sample in data:
        accumulator.add(sample)
    
    stats = accumulator.get_statistics()
    print(f"  ✓ Mean: {stats.mean[:3]}...")
    print(f"  ✓ Q01:  {stats.q01[:3]}...")
    print(f"  ✓ Q99:  {stats.q99[:3]}...")
    
    return stats


def save_normalization_stats(output_dir: Path, state_stats: Any, action_stats: Any):
    """Save normalization stats to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    norm_stats = {"state": state_stats}
    if action_stats is not None:
        norm_stats["actions"] = action_stats
    
    normalize.save(str(output_dir), norm_stats)
    
    print(f"\n✓ Saved to {output_dir}/")
    print(f"  - state.npz (dim={state_stats.q01.shape[0]})")
    if action_stats is not None:
        print(f"  - actions.npz (dim={action_stats.q01.shape[0]})")
    
    # Verify
    loaded = normalize.load(str(output_dir))
    assert loaded['state'].q01.shape == state_stats.q01.shape
    print(f"\n✓ Verified: stats loaded correctly")


def main():
    parser = argparse.ArgumentParser(description="Create RBY1 normalization stats for OpenPI")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with H5 files")
    parser.add_argument("--output-dir", type=str, default="assets/rby1", help="Output directory")
    parser.add_argument("--pattern", type=str, default="*.h5", help="File pattern")
    parser.add_argument("--include-gripper", action="store_true", default=True, help="Include gripper")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        sys.exit(1)
    
    h5_files = sorted(data_dir.glob(args.pattern))
    if not h5_files:
        print(f"Error: No H5 files in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} files in {data_dir}")
    
    # Load, compute, save
    data = load_h5_data(h5_files, include_gripper=args.include_gripper)
    state_stats = compute_normalization_stats(data["states"], "state")
    action_stats = compute_normalization_stats(data["actions"], "actions") if data["actions"] is not None else None
    
    output_dir = Path(args.output_dir)
    save_normalization_stats(output_dir, state_stats, action_stats)
    
    print(f"\n{'='*60}")
    print("Next: Copy to checkpoint assets/")
    print(f"  cp -r {output_dir} /path/to/checkpoint/assets/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()