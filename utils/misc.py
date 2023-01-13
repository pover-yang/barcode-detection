from pathlib import Path


def next_version(exp_root):
    exp_dirs = [d for d in Path(exp_root).iterdir() if d.is_dir()]
    version_nums = []
    for d in exp_dirs:
        if d.name.startswith('v') and d.name[1:].isdigit():
            version_nums.append(int(d.name[1:]))

    return max(version_nums) + 1 if version_nums else 1
