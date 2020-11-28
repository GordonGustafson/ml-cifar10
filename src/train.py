from src.model_utils import *

def search_models(model_results_cache):
    color_channels = 3
    num_classes = 10

    models_dict = collections.OrderedDict()

    NUM_VAL = 8000
    NUM_TRAIN = 42000


    l2_head = math.exp(-0.5)
    l2_backbone = math.exp(-1.0)
    num_epochs = 100
    p_dropout = math.exp(-2.5)
    for i in range(2):
        for p_perspective in np.exp(np.array([-1])):
            for distortion_scale in np.exp(np.array([-1])):
                crop_aug_args = {
                    "crop_weights": [0.33, 0.0, 0.33, 0.34],
                    "crop_sizes":   [(28, 28), (32, 24), (24, 32), (32, 32)],
                    "p_perspective": p_perspective,
                    "distortion_scale": distortion_scale,
                    "interpolation": 2,
                    "fill": MEAN_PIXEL_0_255,
                }
                for clipped_relu_mean in np.exp(np.array([-3])):
                    for clipped_relu_std in np.exp(np.array([-3])):
                        models_dict[f"1_ReLU_7_RandomClippedReLU({clipped_relu_mean},{clipped_relu_std})_3x3_64_64_max2_128_128_max2_256_256_max2_512__glbmax_BN_relu_dropout_fc_ALLBN(bias-only)_batch64_dup{i}",
                                    TrainParams(transform=get_transform_hflip_weighted_perspective_crop(**crop_aug_args),
                                                use_hflip_in_accuracy_eval=USE_HFLIP_IN_ACCURACY_EVAL,
                                                num_train=NUM_TRAIN,
                                                num_val=NUM_VAL,
                                                random_seed=0,
                                                num_epochs=num_epochs,
                                                distortion_scale=distortion_scale,
                                                p_dropout=p_dropout,
                                                head_params=ComponentParams(
                                                    weight_decay=l2_head,
                                                    optimizer="AdamW",
                                                    optimizer_params={"lr": 0.001 / 25, "betas": (0.85, 0.99)},
                                                    lr_scheduler="OneCycleLR",
                                                    lr_scheduler_params={
                                                        "max_lr": 0.001,
                                                        "total_steps": num_epochs * math.ceil(NUM_TRAIN / BATCH_SIZE),
                                                        "pct_start": 0.3,
                                                        "anneal_strategy": 'cos',
                                                        "cycle_momentum": True,
                                                        "base_momentum": 0.85,
                                                        "max_momentum": 0.95,
                                                        "div_factor": 25.0,
                                                        "final_div_factor": 10000.0,
                                                    }),
                                                backbone_params=ComponentParams(
                                                    weight_decay=l2_backbone,
                                                    optimizer="AdamW",
                                                    optimizer_params={"lr": 0.001 / 25, "betas": (0.85, 0.99)},
                                                    lr_scheduler="OneCycleLR",
                                                    lr_scheduler_params={
                                                        "max_lr": 0.001,
                                                        "total_steps": num_epochs * math.ceil(NUM_TRAIN / BATCH_SIZE),
                                                        "pct_start": 0.3,
                                                        "anneal_strategy": 'cos',
                                                        "cycle_momentum": True,
                                                        "base_momentum": 0.85,
                                                        "max_momentum": 0.95,
                                                        "div_factor": 25.0,
                                                        "final_div_factor": 10000.0,
                                                    }))

                            ] = TwoStageModel(
                                backbone=nn.Sequential(
                                    nn.Conv2d(3,   64,  kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(64, affine=False), FilterBias(64), nn.ReLU(),
                                    nn.Conv2d(64,  64,  kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(64, affine=False), FilterBias(64), RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                    nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
                                    nn.Conv2d(64,  128, kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(128, affine=False), FilterBias(128), RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(128, affine=False), FilterBias(128), RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                    nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
                                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(256, affine=False), FilterBias(256), RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(256, affine=False), FilterBias(256), RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                    nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
                                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.BatchNorm2d(512, affine=False), FilterBias(512), RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                ),
                                head=nn.Sequential(
                                    nn.MaxPool2d(kernel_size=4, stride=1, padding=0), FlattenKeepDims(),
                                    nn.BatchNorm2d(512, affine=False),
                                    FilterBias(512),
                                    RandomClippedReLU(clipped_relu_mean, clipped_relu_std),
                                    nn.Dropout(p_dropout),
                                    Flatten(),
                                    nn.Linear(512, num_classes)
                                )
                            )

    fill_results_cache(models_dict, model_results_cache)
    return models_dict.keys()

