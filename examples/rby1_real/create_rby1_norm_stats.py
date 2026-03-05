"""Generate quantile normalization stats for RBY1 (16-dim state/action).

Creates norm_stats for OpenPI policy server from H5 trajectory data.
Expects H5 files with structure: samples/{robot_position, gripper_state,
robot_target_joints, gripper_target}. Supports nested episode directories.

Note: Uses robot_position[8:22] (14 arm joints) + gripper_state (2) = 16-dim
to match the data conversion pipeline (data_conversion.ipynb).

Usage:
    uv run python create_rby1_norm_stats.py --data-dir /path/to/episodes --output-dir assets/rby1
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
    """Load state and action data from H5 files.
    
    Supports two H5 layouts:
      - Flat: robot_position, gripper_state, target_position/action, gripper_target at root
      - Grouped (RBY1 raw): samples/{robot_position, gripper_state,
                             robot_target_joints, gripper_target}
    """
    states = []
    actions = []
    
    print(f"Loading {len(h5_files)} H5 files...")
    
    for h5_file in h5_files:
        print(f"  {'/'.join(h5_file.parts[-2:])}...", end=" ")
        try:
            with h5py.File(h5_file, 'r') as f:
                # Detect layout: grouped (samples/) or flat
                if 'samples' in f:
                    grp = f['samples']
                else:
                    grp = f

                # Load robot position (24 joints for RBY1)
                if 'robot_position' not in grp:
                    print("missing 'robot_position', skipped")
                    continue
                robot_pos = grp['robot_position'][:]
                # Slice to arm joints [8:22] (14-dim) to match data conversion
                robot_pos = robot_pos[:, 8:22]
                
                # Append gripper state if requested (2 values)
                if include_gripper and 'gripper_state' in grp:
                    gripper_state = grp['gripper_state'][:]
                    state = np.concatenate([robot_pos, gripper_state], axis=-1)
                else:
                    state = robot_pos
                    if include_gripper:
                        print("(no gripper)", end=" ")
                
                states.append(state)
                
                # Load actions: robot_target_joints > target_position > action
                for action_key in ('robot_target_joints', 'target_position', 'action'):
                    if action_key in grp:
                        break
                else:
                    action_key = None

                if action_key is not None:
                    robot_action = grp[action_key][:]
                    # Slice to arm joints [8:22] (14-dim) to match data conversion
                    robot_action = robot_action[:, 8:22]
                    
                    if include_gripper and 'gripper_target' in grp:
                        gripper_action = grp['gripper_target'][:]
                        action = np.concatenate([robot_action, gripper_action], axis=-1)
                    else:
                        action = robot_action
                    
                    actions.append(action)
                    print(f"✓ ({state.shape[0]} steps, state={state.shape[1]}d, action={action.shape[1]}d)")
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
    
    running_stats = normalize.RunningStats()
    running_stats.update(data)
    
    stats = running_stats.get_statistics()
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
    parser.add_argument("--pattern", type=str, default="**/*.h5", help="File glob pattern (supports ** for recursion)")
    parser.add_argument("--include-gripper", action="store_true", default=True, help="Include gripper")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        sys.exit(1)
    
    h5_files = sorted(data_dir.glob(args.pattern) if '**' not in args.pattern else data_dir.rglob(args.pattern.replace('**/', '').replace('**', '*.h5')))
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