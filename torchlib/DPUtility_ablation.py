import sys, os.path
from argparse import Namespace
from random import uniform
from warnings import warn
from pandas import DataFrame

sys.path.insert(0, os.path.split(sys.path[0])[0])

from train import main


def ablation(noise_multiplier, max_grad_norm):
    lr = 1e-3
    epochs = 25
    args = Namespace(
        bin_seg=True,
        config=f"ablationutilityprivacy",
        resume_checkpoint=None,
        train_federated=False,
        data_dir="./Task03_Liver",
        visdom=False,
        encrypted_inference=False,
        cuda=True,
        websockets=False,
        batch_size=32,
        train_resolution=256,
        inference_resolution=256,
        test_batch_size=32,
        test_interval=1,
        epochs=epochs,
        lr=lr,
        end_lr=1e-4,
        restarts=1,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0,
        seed=1,
        log_interval=10,
        deterministic=False,
        optimizer="Adam",
        model="MoNet",
        pretrained=True,
        weight_classes=False,
        pooling_type="max",
        rotation=90,
        translate=0.5,
        scale=0.5,
        noise_std=0.01,
        noise_prob=0,
        mixup=False,
        num_threads=0,
        save_file=f"model_weights/completed_trainings_DPUTILITY.csv",
        name=f"DPUTILITY",
    )
    args.individual_albu_probs = 0.75
    args.clahe = False
    args.randomgamma = False
    args.randombrightness = False
    args.blur = False
    args.elastic = False
    args.optical_distortion = False
    args.grid_distortion = False
    args.grid_shuffle = False
    args.hsv = False
    args.invert = False
    args.cutout = False
    args.shadow = False
    args.fog = False
    args.sun_flare = False
    args.solarize = False
    args.equalize = False
    args.grid_dropout = False
    args.dump_gradients_every = None

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
    best_val_acc, epsilon = main(args, verbose=False)
    if epsilon < 1:
        warn(f"Epsilon is only {epsilon:.2f}. Seems very low.")
    return best_val_acc, epsilon


if __name__ == "__main__":
    DataFrame(
        {"noise_multiplier": [], "max_grad_norm": [], "utility": [], "epsilon": []}
    ).to_csv("ablation.csv")
    for i in range(40):
        noise_multiplier = uniform(0.1, 1.0)
        max_grad_norm = uniform(0.1, 2.0)
        utility, epsilon = ablation(noise_multiplier, max_grad_norm)
        results = {
            "noise_multiplier": [noise_multiplier],
            "max_grad_norm": [max_grad_norm],
            "utility": [utility],
            "epsilon": [epsilon],
        }
        print(f"Results: {results}")
        DataFrame(results).to_csv("ablation.csv", mode="a", header=False)
