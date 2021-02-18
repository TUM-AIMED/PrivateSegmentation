import sys, os.path
import argparse
from random import uniform
from warnings import warn
import pandas as pd
from numpy import generic as np_generic
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.split(sys.path[0])[0])

from train import main


def ablation(args, noise_multiplier, max_grad_norm):
    # lr = 1e-3
    # epochs = 25
    # args = Namespace(
    #     bin_seg=True,
    #     config=f"ablationutilityprivacy",
    #     resume_checkpoint=None,
    #     train_federated=False,
    #     data_dir="./Task03_Liver",
    #     visdom=False,
    #     encrypted_inference=False,
    #     cuda=True,
    #     websockets=False,
    #     batch_size=32,
    #     train_resolution=256,
    #     inference_resolution=256,
    #     test_batch_size=32,
    #     test_interval=1,
    #     epochs=epochs,
    #     lr=lr,
    #     end_lr=1e-4,
    #     restarts=1,
    #     beta1=0.9,
    #     beta2=0.99,
    #     weight_decay=0,
    #     seed=1,
    #     log_interval=10,
    #     deterministic=False,
    #     optimizer="Adam",
    #     model="MoNet",
    #     pretrained=True,
    #     weight_classes=False,
    #     pooling_type="max",
    #     rotation=90,
    #     translate=0.5,
    #     scale=0.5,
    #     noise_std=0.01,
    #     noise_prob=0,
    #     mixup=False,
    #     num_threads=0,
    #     save_file=f"model_weights/completed_trainings_DPUTILITY.csv",
    #     name=f"DPUTILITY",
    # )
    # args.individual_albu_probs = 0.75
    # args.clahe = False
    # args.randomgamma = False
    # args.randombrightness = False
    # args.blur = False
    # args.elastic = False
    # args.optical_distortion = False
    # args.grid_distortion = False
    # args.grid_shuffle = False
    # args.hsv = False
    # args.invert = False
    # args.cutout = False
    # args.shadow = False
    # args.fog = False
    # args.sun_flare = False
    # args.solarize = False
    # args.equalize = False
    # args.grid_dropout = False
    # args.dump_gradients_every = None
    args.print_gradient_norm_every = -1

    args.config = (f"ablationutilityprivacy",)
    args.save_file = f"model_weights/completed_trainings_DPUTILITY.csv"
    args.name = f"DPUTILITY"

    args.deterministic = True
    args.data_dir = "./data/MSD/Task03_Liver/"
    # args.batch_size = 1
    args.differentially_private = True
    args.microbatch_size = 1
    args.target_delta = 1.0 / 4000.0
    args.noise_multiplier = noise_multiplier
    args.max_grad_norm = max_grad_norm

    # if cmdln_args.federated:
    #     args.unencrypted_aggregation = cmdln_args.unencrypted_aggregation
    #     args.sync_every_n_batch = trial.suggest_int("sigma", 1, 5)
    #     args.wait_interval = 0.1
    #     args.keep_optim_dict = False
    #     if not cmdln_args.unencrypted_aggregation:
    #         args.precision_fractional = 16
    #     # trial.suggest_categorical(
    #     #     "keep_optim_dict", [True, False]
    #     # )
    #     args.weighted_averaging = trial.suggest_categorical(
    #         "weighted_averaging", [True, False]
    #     )
    #     args.DPSSE = False
    #     args.dpsse_eps = 1.0
    #     args.microbatch_size = args.batch_size
    objectives, epsila = main(args, verbose=True, return_all_perfomances=True)
    if epsila[-1] < 1:
        warn(f"Epsilon is only {epsila[-1]:.2f}. Seems very low.")
    return objectives, epsila


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_file", required=True, type=str, help="Where to save results"
    )
    parser.add_argument(
        "--args_file", required=True, type=str, help="Where to get hyperparams from"
    )
    cmd_args = parser.parse_args()
    hyperparam_runs = pd.read_csv(cmd_args.args_file)
    args = argparse.Namespace()
    for key, value in dict(
        hyperparam_runs.iloc[hyperparam_runs["best_validation_score"].argmax()]
    ).items():
        if isinstance(value, np_generic):
            value = value.item()
        args.__setattr__(key, value)

    if not Path(cmd_args.save_file).is_file():
        pd.DataFrame(
            {"noise_multiplier": [], "max_grad_norm": [], "utility": [], "epsilon": []}
        ).to_csv(cmd_args.save_file)
    # noises = [0.5, 1.0, 1.5, 2.0, 2.5]
    # clips = [0.05, 0.1, 0.2, 0.5]
    # noise_clips = [(0.01, 1.0)] + [*product(noises, clips)]
    noise_clips = [(0.8, 1.0), (1.0, 0.5), (1.5, 0.1)]
    for i, (noise_multiplier, max_grad_norm) in enumerate(noise_clips):
        # noise_multiplier = 1e-6
        # max_grad_norm = 10.0
        utilities, epsila = ablation(args, noise_multiplier, max_grad_norm)
        assert len(utilities) == len(epsila)
        results = {
            "noise_multiplier": [noise_multiplier] * len(utilities),
            "max_grad_norm": [max_grad_norm] * len(utilities),
            "utility": utilities,
            "epsilon": epsila,
        }
        print(f"Results: {results}")
        pd.DataFrame(results).to_csv(cmd_args.save_file, mode="a", header=False)
