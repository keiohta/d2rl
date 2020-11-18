import os
import argparse
import logging
import subprocess

template = r"""#!/bin/bash

#SBATCH --array=1-{concurrent}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:{gpu}:1
#SBATCH -J CURL_{env}
#SBATCH -o CURL_{env}.log
#SBATCH -e CURL_{env}_error.log

python train.py --env {env} --batch_size {batch_size}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=5)
    parser.add_argument("--out", default="sruns")
    parser.add_argument("--run", default=False, action="store_true")
    parser.add_argument("--dir-root", default="output", type=str)
    args = parser.parse_args()

    dir_out = os.path.join("output", args.out)
    os.makedirs(dir_out, exist_ok=True)

    trial = args.trial
    is_run = args.run
    dir_root = args.dir_root

    envs = ['finger', 'cartpole', 'reacher', 'cheetah', 'walker', 'ball']

    for idx, env in enumerate(envs):
        gpu_type = "TitanV" if idx % 2 == 0 else "QuadroRTX6000"
        batch_size = 512 if env == 'cheetah' else 128
        cur_batch = template.format(concurrent=trial, gpu=gpu_type, dir_root=dir_root, batch_size=batch_size, env=env)

        cur_out = os.path.join(dir_out, f"CURL_{env}.srun")
        with open(cur_out, "w") as f:
            f.write(cur_batch)

        print(f"run {cur_out}")
        if is_run:
            args = ["sbatch", cur_out]
            subprocess.run(args)


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s')
    main()
