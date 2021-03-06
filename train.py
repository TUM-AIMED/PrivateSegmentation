from os import path, remove, environ, getcwd

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import configparser
import random
import shutil
from datetime import datetime
import time
from warnings import warn
from copy import deepcopy

import numpy as np
import syft as sy
import torch
from cv2 import INTER_NEAREST

# torch.set_num_threads(36)

import torch.nn as nn
import torch.optim as optim
import tqdm
import visdom
import albumentations as a
from tabulate import tabulate
from torchvision import datasets, transforms
from optuna import TrialPruned
from math import ceil, floor
from torchlib.dataloader import (
    calc_mean_std,
    AlbumentationsTorchTransform,
    random_split,
    create_albu_transform,
    CombinedLoader,
    SegmentationData,  # Segmentation
    MSD_data,
    MSD_data_images,
)  # pylint:disable=import-error
from torchlib.models import (
    conv_at_resolution,  # pylint:disable=import-error
    resnet18,
    vgg16,
    SimpleSegNet,  # Segmentation
    MoNet,
    getMoNet,
)
from torchlib.utils import (
    Arguments,
    Cross_entropy_one_hot,
    LearningRateScheduler,
    MixUp,
    save_config_results,
    save_model,
    test,
    train,
    train_federated,
    setup_pysyft,
    calc_class_weights,
)
import segmentation_models_pytorch as smp
from revision_scripts.module_modification import (
    convert_batchnorm_modules,
    _batchnorm_to_bn_without_stats,
)
from opacus import PrivacyEngine
from joblib import load as joblibload

from syft.frameworks.torch.fl.dataloader import PoissonBatchSampler
from torch.utils.data import SequentialSampler


def main(
    args, verbose=True, optuna_trial=None, cmd_args=None, return_all_perfomances=False
):

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.deterministic and args.websockets:
        warn(
            "Training with GridNodes is not compatible with deterministic training.\n"
            "Switching deterministic flag to False"
        )
        args.deterministic = False
    if args.deterministic:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    kwargs = {"num_workers": args.num_threads, "pin_memory": True,} if use_cuda else {}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "{:s}_{:s}_{:s}".format(
        "federated" if args.train_federated else "vanilla",
        args.data_dir.replace("/", ""),
        timestamp,
    )
    num_classes = 10 if args.data_dir == "mnist" else 3
    class_names = None
    # Dataset creation and definition
    if args.train_federated:
        if hasattr(torch, "torch_hooked"):
            hook = sy.hook
        else:
            hook = sy.TorchHook(torch)

        (
            train_loader,
            val_loader,
            total_L,
            workers,
            worker_names,
            crypto_provider,
            val_mean_std,
        ) = setup_pysyft(args, hook, verbose=verbose,)
    else:
        if args.data_dir == "mnist":
            val_mean_std = torch.tensor(  # pylint:disable=not-callable
                [[0.1307], [0.3081]]
            )
            mean, std = val_mean_std
            if args.pretrained:
                mean, std = mean[None, None, :], std[None, None, :]
            train_tf = [
                transforms.Resize(args.train_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if args.pretrained:
                repeat = transforms.Lambda(
                    lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                        x, 3, dim=0
                    )
                )
                train_tf.append(repeat)
            dataset = datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(train_tf),
            )
            total_L = len(dataset)
            fraction = 1.0 / args.validation_split
            dataset, valset = random_split(
                dataset,
                [int(ceil(total_L * (1.0 - fraction))), int(floor(total_L * fraction))],
            )
        elif args.bin_seg:
            # NOTE: the different other segmentation datasets were left commented out

            ## MSRC dataset ##
            # dataset = SegmentationData(image_paths_file='data/segmentation_data/train.txt')
            # valset = SegmentationData(image_paths_file='data/segmentation_data/val.txt')

            ## MSD dataset ##
            # RES = 256
            # RES_Z = 64
            # CROP_HEIGHT = 16

            # sample_limit = 2
            # dataset = MSD_data(
            #     path_string=PATH,
            #     res=RES,
            #     res_z=RES_Z,
            #     crop_height=CROP_HEIGHT,
            #     sample_limit=sample_limit,
            # )

            # # split into val and train set
            # train_size = int(0.8 * len(dataset))
            # val_size = len(dataset) - train_size
            # dataset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

            ####              MSD dataset preprocessed version              ####
            # NOTE: Albumentations need the input (width, height, channel)     #
            #       --> otherwise: instead (1, 256, 256) -> (256, 256, 256)    #
            # NOTE: Torch transforms need the input (channel, width, height)   #
            #       --> otherwise: type not subscriptable error                #
            # NOTE: The AlbumentationsTorchTransform wrapper auto. changes     #
            #       (256, 256, 1) back to (1, 256, 256) after the transforms.  #
            #       Putting a torch transform in the wrapper leads to          #
            #       'force_apply...' error                                     #
            # NOTE: Easiest way to do transforms: keep as PIL images.          #
            ####################################################################

            # transforms applied to get the stats: mean and val
            basic_tfs = [
                a.Resize(args.inference_resolution, args.inference_resolution,),
                a.RandomCrop(args.train_resolution, args.train_resolution),
                a.ToFloat(max_value=255.0),
            ]
            stats_tf_imgs = AlbumentationsTorchTransform(a.Compose(basic_tfs))
            # dataset to calculate stats
            dataset = MSD_data_images(
                args.data_dir + "/train", transform=stats_tf_imgs,
            )
            # get stats
            val_mean_std = calc_mean_std(dataset)
            mean, std = val_mean_std

            # change transforms based on stats
            train_tf = create_albu_transform(args, mean, std)

            # mask is a special keyword in albumentations
            dataset.transform = train_tf
            val_trans = a.Compose(
                [
                    *basic_tfs,
                    a.Normalize(mean, std, max_pixel_value=1.0),
                    a.Lambda(
                        image=lambda x, **kwargs: x.reshape(
                            # add extra channel to be compatible with nn.Conv2D
                            -1,
                            args.train_resolution,
                            args.train_resolution,
                        ),
                        mask=lambda x, **kwargs: np.where(
                            # binarize masks
                            x.reshape(-1, args.train_resolution, args.train_resolution)
                            / 255.0
                            > 0.5,
                            np.ones_like(x),
                            np.zeros_like(x),
                        ).astype(np.float32),
                    ),
                ]
            )
            valset = MSD_data_images(
                args.data_dir + "/val",
                transform=AlbumentationsTorchTransform(val_trans),
            )

        else:
            # Different train and inference resolution only works with adaptive
            # pooling in model activated
            stats_tf = AlbumentationsTorchTransform(
                a.Compose(
                    [
                        a.Resize(args.inference_resolution, args.inference_resolution),
                        a.RandomCrop(args.train_resolution, args.train_resolution),
                        a.ToFloat(max_value=255.0),
                    ]
                )
            )
            # dataset = PPPP(
            #     "data/Labels.csv",
            loader = CombinedLoader()
            if not args.pretrained:
                loader.change_channels(1)
            dataset = datasets.ImageFolder(
                args.data_dir, transform=stats_tf, loader=loader,
            )
            # TODO: issue #1 - this only creates two 2 new dimensions in case of three channels
            assert (
                len(dataset.classes) == 3
            ), "Dataset must have exactly 3 classes: normal, bacterial and viral"
            val_mean_std = calc_mean_std(dataset)
            mean, std = val_mean_std
            # TODO: issue #1 - this only creates two 2 new dimensions in case of three channels
            if args.pretrained:
                mean, std = mean[None, None, :], std[None, None, :]
            dataset.transform = create_albu_transform(args, mean, std)
            class_names = dataset.classes
            stats_tf.transform.transforms.transforms.append(
                a.Normalize(mean, std, max_pixel_value=1.0)
            )
            valset = datasets.ImageFolder(  # TODO hardcoded path
                "data/test", transform=stats_tf, loader=loader
            )
            # occurances = dataset.get_class_occurances()

        # total_L = total_L if args.train_federated else len(dataset)
        # fraction = 1.0 / args.validation_split
        # dataset, valset = random_split(
        #     dataset,
        #     [int(ceil(total_L * (1.0 - fraction))), int(floor(total_L * fraction))],
        # )
        if args.differentially_private:
            sampler = SequentialSampler(range(len(dataset)))
            batch_sampler = PoissonBatchSampler(sampler, args.batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                # sampler=sampler,
                **kwargs,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, **kwargs,
            )

        # val_tf = [
        #     a.Resize(args.inference_resolution, args.inference_resolution),
        #     a.CenterCrop(args.inference_resolution, args.inference_resolution),
        #     a.ToFloat(max_value=255.0),
        #     a.Normalize(mean, std, max_pixel_value=1.0),
        # ]
        # if not args.pretrained:
        #     val_tf.append(a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis]))
        # valset.dataset.transform = AlbumentationsTorchTransform(a.Compose(val_tf))

        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.test_batch_size, shuffle=True, **kwargs,
        )
        # del total_L, fraction

    cw = None
    if args.weight_classes:
        cw = calc_class_weights(args, train_loader, num_classes)
        cw = cw.to(device)

    scheduler = LearningRateScheduler(
        args.epochs, np.log10(args.lr), np.log10(args.end_lr), restarts=args.restarts
    )

    ## visdom
    vis_params = None
    if args.visdom:
        vis = visdom.Visdom()
        assert vis.check_connection(
            timeout_seconds=3
        ), "Connection to the visdom server could not be established!"
        vis_env = path.join(
            "federated" if args.train_federated else "vanilla", timestamp
        )
        plt_dict = dict(
            name="training loss",
            ytickmax=10,
            xlabel="epoch",
            ylabel="loss",
            legend=["train_loss"],
        )
        vis.line(
            X=np.zeros((1, 2)),
            Y=np.zeros((1, 2)),
            win="loss_win",
            opts={
                "legend": ["train_loss", "val_loss"],
                "xlabel": "epochs",
                "ylabel": "loss",
            },
            env=vis_env,
        )
        if not args.bin_seg:
            vis.line(
                X=np.zeros((1, 2)),
                Y=np.zeros((1, 2)),
                win="metrics_win",
                opts={
                    "legend": ["matthews coeff", "ROC AUC"],
                    "xlabel": "epochs",
                    "ylabel": "m coeff [%] / ROC AUC",
                },
                env=vis_env,
            )
        vis.line(
            X=np.zeros((1, 1)),
            Y=np.zeros((1, 1)),
            win="lr_win",
            opts={"legend": ["learning_rate"], "xlabel": "epochs", "ylabel": "lr"},
            env=vis_env,
        )
        vis_params = {"vis": vis, "vis_env": vis_env}
    # for the models that are loaded in directly (e.g. U-Net)
    if args.model == "unet":
        warn(
            "Pure UNet is deprecated. Please specify backbone (unet_resnet18, unet_mobilenet_v2, unet_vgg11_bn)",
            category=DeprecationWarning,
        )
        exit()
    if args.model == "vgg16":
        model_type = vgg16
        model_args = {
            "pretrained": args.pretrained,
            "num_classes": num_classes,
            "in_channels": 1 if args.data_dir == "mnist" or not args.pretrained else 3,
            "adptpool": False,
            "input_size": args.inference_resolution,
            "pooling": args.pooling_type,
        }
    elif args.model == "simpleconv":
        if args.pretrained:
            warn("No pretrained version available")

        model_type = conv_at_resolution[args.train_resolution]
        model_args = {
            "num_classes": num_classes,
            "in_channels": 1 if args.data_dir == "mnist" or not args.pretrained else 3,
            "pooling": args.pooling_type,
        }
    elif args.model == "resnet-18":
        model_type = resnet18
        model_args = {
            "pretrained": args.pretrained,
            "num_classes": num_classes,
            "in_channels": 1 if args.data_dir == "mnist" or not args.pretrained else 3,
            "adptpool": False,
            "input_size": args.inference_resolution,
            "pooling": args.pooling_type,
        }
    # Segmentation
    elif args.model == "SimpleSegNet":
        model_type = SimpleSegNet
        # no params for now
        model_args = {}
    elif "unet" in args.model:
        backbone = "_".join(
            args.model.split("_")[1:]
        )  # remove the unet from the model str
        # because we don't call any function but directly create the model
        # preprocessing step due to version problem (model was saved from torch 1.7.1)
        # resnet18 can be directly replaced by vgg11 and mobilenet
        # PRETRAINED_PATH =  "/pretrained_models/unet_resnet18_weights.dat"
        model_args = {
            "encoder_name": backbone,
            "classes": 1,
            "in_channels": 1,
            "activation": "sigmoid",
            "encoder_weights": None,
        }
        model_type = smp.Unet
        # model.encoder.conv1 = nn.Sequential(nn.Conv2d(1, 3, 1), model.encoder.conv1)

    elif args.model == "MoNet":
        model_type = getMoNet
        model_args = {
            "activation": "sigmoid",
        }
    else:
        raise ValueError(
            "Model name not understood. Please choose one of 'vgg16, 'simpleconv', resnet-18'."
        )

    model = model_type(**model_args)

    if args.pretrained and (not args.pretrained_path is None):
        # with open(getcwd() + args.pretrained_path, "rb") as handle:
        #     state_dict = pickle.load(handle)
        #     model.load_state_dict(state_dict)
        state_dict = joblibload(args.pretrained_path)
        model.load_state_dict(state_dict)
    model.to(device)

    if args.train_federated:
        model = {
            key: model.copy()
            for key in [w.id for w in workers.values()] + ["local_model"]
        }

    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "SGD":
        opt = optim.SGD
    elif args.optimizer == "Adam":
        opt = optim.Adam
        opt_kwargs["betas"] = (args.beta1, args.beta2)
    else:
        raise ValueError(
            "Optimizer name not understood. Please use one of 'SGD' or 'Adam'."
        )
        # if args.train_federated and not args.secure_aggregation:
        #     from syft.federated.floptimizer import Optims

        # optimizer = Optims(worker_names, optimizer)

    optimizer = (
        {
            idt: opt(m.parameters(), **opt_kwargs)
            for idt, m in model.items()
            if idt not in ["local_model", "crypto_provider"]
        }
        if args.train_federated
        else opt(model.parameters(), **opt_kwargs)
    )
    ALPHAS = None
    if args.differentially_private:
        if args.mixup:
            warn("Mixup and DP do not like each other.")
            exit()
        if args.weight_decay > 0:
            warn("We would recommend setting weight decay to 0 when using DP")
        if (
            args.clahe
            or args.randomgamma
            or args.randombrightness
            or args.blur
            or args.elastic
            or args.optical_distortion
            or args.grid_distortion
            or args.grid_shuffle
            or args.grid_shuffle
            or args.hsv
            or args.invert
            or args.cutout
            or args.shadow
            or args.sun_flare
            or args.fog
            or args.solarize
            or args.equalize
        ):
            warn("We would recommend not using augmentations with DP")
        ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        if args.train_federated:
            for key, m in model.items():
                model[key] = convert_batchnorm_modules(
                    m, converter=_batchnorm_to_bn_without_stats
                )
            for w in train_loader.keys():
                if not hasattr(w, "total_dp_steps"):
                    setattr(w, "total_dp_steps", 0)
        else:
            if not hasattr(model, "total_dp_steps"):
                setattr(model, "total_dp_steps", 0)
            # privacy_engine = PrivacyEngine(
            #     model,
            #     batch_size=args.batch_size,  # recommended in opacus tutorial
            #     sample_size=len(train_loader.dataset),
            #     alphas=ALPHAS,
            #     noise_multiplier=args.noise_multiplier,
            #     max_grad_norm=args.max_grad_norm,
            # )
            # privacy_engine.attach(optimizer)
            model = convert_batchnorm_modules(
                model, converter=_batchnorm_to_bn_without_stats
            )

    loss_args = {"weight": cw, "reduction": "mean"}
    if args.mixup or (args.weight_classes and args.train_federated):
        loss_fn = Cross_entropy_one_hot
    else:
        loss_fn = nn.CrossEntropyLoss

    # Segmentation - we have to ignore the classes with label -1, they represent unlabeled data
    # Only for MSRC dataset
    if args.bin_seg:
        # loss_args = {"ignore_index":-1, "reduction":"mean"}
        # reduction mean is set by defaut
        # stats for weighting from asmple 0.jpg in /train
        # white_pixels/ all_pixels = 0.0602 -> % of pos. classes
        # (256*256-a_np.sum())/a_np.sum() -> 15.598 times more negative classes
        # pos_weight = torch.tensor([15]).to(device)
        # loss_args = {"pos_weight" : pos_weight}

        # loss_fn = nn.BCEWithLogitsLoss
        loss_fn = smp.utils.losses.DiceLoss
        loss_args = {}

    loss_fn = loss_fn(**loss_args)

    if args.train_federated:
        loss_fn = {w: loss_fn.copy() for w in [*workers, "local_model"]}

    gradient_dump = {} if args.dump_gradients_every else None
    if args.dump_gradients_every and args.train_federated:
        print("[Warning] Dump gradients only supported for local training")
        exit()

    start_at_epoch = 1
    if cmd_args and cmd_args.resume_checkpoint:
        print("Resume training from a given checkpoint.")
        state = torch.load(cmd_args.resume_checkpoint, map_location=device)
        start_at_epoch = state["epoch"]
        # args = state["args"]
        checkpoint_args = state["args"]
        if cmd_args.train_federated and checkpoint_args.train_federated:
            opt_state_dict = state["optim_state_dict"]
            for w in worker_names:
                if w not in opt_state_dict:
                    warn(
                        (
                            "The worker names of the checkpoint and the current "
                            "configuration cannot be matched."
                        )
                    )
                    exit()
                optimizer[w].load_state_dict(opt_state_dict[w])
            for w in model.keys():
                model[w].load_state_dict(state["model_state_dict"])
        elif cmd_args.train_federated and not checkpoint_args.train_federated:
            assert (
                len(state["optim_state_dict"]) == 2
                and "param_groups" in state["optim_state_dict"]
                and "state" in state["optim_state_dict"]
            )  # model checkpoint was no federated training
            for w in worker_names:
                optimizer[w].load_state_dict(state["optim_state_dict"])
            for key in model.keys():
                model[key].load_state_dict(state["model_state_dict"])

        elif not cmd_args.train_federated and checkpoint_args.train_federated:
            # no optimizer is loaded
            model.load_state_dict(state["model_state_dict"]["local_model"])
        elif not cmd_args.train_federated and not checkpoint_args.train_federated:
            optimizer.load_state_dict(state["optim_state_dict"])
            model.load_state_dict(state["model_state_dict"])
        else:
            warn(
                (
                    "Checkpoint was not loaded as the combination of the "
                    "checkpoint and the current configuration is not handled yet."
                )
            )  # not possible to load previous optimizer if setting changed
        # args.incorporate_cmd_args(cmd_args)
    if args.train_federated:
        for m in model.values():
            m.to(device)
    else:
        model.to(device)
    test(
        args,
        model["local_model"] if args.train_federated else model,
        device,
        val_loader,
        start_at_epoch - 1,
        loss_fn["local_model"] if args.train_federated else loss_fn,
        num_classes,
        vis_params=vis_params,
        class_names=class_names,
        verbose=verbose,
    )
    objectives = []
    epsila = []
    model_paths = []
    times = []
    """if args.train_federated:
        test_params = {
            "device": device,
            "val_loader": val_loader,
            "loss_fn": loss_fn,
            "num_classes": num_classes,
            "class_names": class_names,
            "exp_name": exp_name,
            "optimizer": optimizer,
            "matthews_scores": matthews_scores,
            "model_paths": model_paths,
        }"""
    for epoch in (
        range(start_at_epoch, args.epochs + 1)
        if verbose
        else tqdm.tqdm(
            range(start_at_epoch, args.epochs + 1),
            leave=False,
            desc="training",
            total=args.epochs + 1,
            initial=start_at_epoch,
        )
    ):
        if args.train_federated:
            for w in worker_names:
                new_lr = scheduler.adjust_learning_rate(
                    optimizer[
                        w
                    ],  # if args.secure_aggregation else optimizer.get_optim(w),
                    epoch - 1,
                )
        else:
            new_lr = scheduler.adjust_learning_rate(optimizer, epoch - 1)
        if args.visdom:
            vis.line(
                X=np.asarray([epoch - 1]),
                Y=np.asarray([new_lr]),
                win="lr_win",
                name="learning_rate",
                update="append",
                env=vis_env,
            )

        epoch_start_time = time.time()
        if args.train_federated:
            model, epsilon = train_federated(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                loss_fn,
                crypto_provider,
                # In future test_params could be changed if testing
                # during epoch should be enabled
                test_params=None,
                vis_params=vis_params,
                verbose=verbose,
                alphas=ALPHAS,
            )

        else:
            model, epsilon, gradient_dump = train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                loss_fn,
                num_classes,
                vis_params=vis_params,
                verbose=verbose,
                gradient_dump=gradient_dump,
                alphas=ALPHAS,
            )
        # except Exception as e:
        times.append(time.time() - epoch_start_time)

        if (epoch % args.test_interval) == 0:
            _, objective = test(
                args,
                model["local_model"] if args.train_federated else model,
                device,
                val_loader,
                epoch,
                loss_fn["local_model"] if args.train_federated else loss_fn,
                num_classes=num_classes,
                vis_params=vis_params,
                class_names=class_names,
                verbose=verbose,
            )
            model_path = "model_weights/{:s}_epoch_{:03d}.pt".format(
                exp_name,
                epoch
                * (
                    args.repetitions_dataset
                    if "repetitions_dataset" in vars(args)
                    else 1
                ),
            )
            if optuna_trial:
                optuna_trial.report(
                    objective,
                    epoch
                    * (args.repetitions_dataset if args.repetitions_dataset else 1),
                )
                if optuna_trial.should_prune():
                    raise TrialPruned()

            save_model(model, optimizer, model_path, args, epoch, val_mean_std)
            objectives.append(objective)
            epsila.append(epsilon)
            model_paths.append(model_path)
    if return_all_perfomances:
        return_value = deepcopy(objectives), deepcopy(epsila)
    # reversal and formula because we want last occurance of highest value
    objectives = np.array(objectives[::-1])
    best_score_idx = np.argmax(objectives)
    highest_score = len(objectives) - best_score_idx - 1
    best_epoch = (
        highest_score + 1
    ) * args.test_interval  # actually -1 but we're switching to 1 indexed here
    best_model_file = model_paths[highest_score]
    print(
        "Highest objective-score was {:.2f} in epoch {:d}. \
        \nTime per epoch - mean: {:.2f}s, std: {:.2f}s".format(
            objectives[best_score_idx],
            best_epoch * (args.repetitions_dataset if args.train_federated else 1),
            np.mean(times), 
            np.std(times),
        )
    )
    # load best model on val set
    state = torch.load(best_model_file, map_location=device)
    if args.train_federated:
        model = model["local_model"]
    model.load_state_dict(state["model_state_dict"])

    shutil.copyfile(
        best_model_file, "model_weights/final_{:s}.pt".format(exp_name),
    )
    if args.save_file:
        save_config_results(
            args, objectives[best_score_idx], timestamp, args.save_file,
        )

    # delete old model weights
    if not return_all_perfomances:
        for model_file in model_paths:
            remove(model_file)

    if args.dump_gradients_every:
        torch.save(gradient_dump, f"model_weights/gradient_dump_{exp_name}.pt")
    if return_all_perfomances:
        return return_value
    else:
        return objectives[best_score_idx], epsilon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (.ini).",
    )
    parser.add_argument(
        "--train_federated", action="store_true", help="Train with federated learning."
    )
    parser.add_argument(
        "--unencrypted_aggregation",
        action="store_true",
        help="Turns off secure aggregation."
        "Slight advantages in terms of model performance and training speed.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        # required=True,
        default="data/train",
        help='Select a data folder [if "mnist" is passed, the torchvision MNIST dataset will be downloaded and used].',
    )
    parser.add_argument(
        "--visdom", action="store_true", help="Use Visdom for monitoring training."
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration.")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Start training from older model checkpoint",
    )
    parser.add_argument(
        "--websockets", action="store_true", help="Train using WebSockets."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Sets Syft workers to verbose mode"
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="model_weights/completed_trainings.csv",
        help="Store args and result in csv file.",
    )
    parser.add_argument(
        "--training_name",
        default=None,
        type=str,
        help="Optional name to be stored in csv file to later identify training.",
    )
    parser.add_argument(
        "--dump_gradients_every",
        default=None,
        type=int,
        help="Dump gradients during training every n steps",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Path to pretrained model weights that shall be used.",
    )
    cmd_args = parser.parse_args()

    config = configparser.ConfigParser()
    assert path.isfile(cmd_args.config), "Configuration file not found"
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode="train")
    if args.websockets:
        if not args.train_federated:
            raise RuntimeError("WebSockets can only be used when in federated mode.")
    ## CUDA in FL ##
    # if args.cuda and args.train_federated:
    #    warn(
    #        "CUDA is currently not supported by the backend. This option will be available at a later release",
    #        category=FutureWarning,
    #    )
    #    exit(0)
    if args.train_federated and (args.mixup or args.weight_classes):
        if args.mixup and args.mixup_lambda == 0.5:
            warn(
                "Class weighting and a lambda value of 0.5 are incompatible, setting lambda to 0.499",
                category=RuntimeWarning,
            )
            args.mixup_lambda = 0.499
    print(str(args))
    main(args, cmd_args=cmd_args)
