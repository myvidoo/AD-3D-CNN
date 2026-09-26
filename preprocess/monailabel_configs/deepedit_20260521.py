# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, Optional, Union

from monai.networks.nets import SwinUNETR, DynUNet, UNETR, SegResNet
import torch

import lib.infers
import lib.trainers

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class DeepEdit(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.epistemic_enabled = None
        self.epistemic_samples = None

        # Single label
        self.labels = {
            "brain": 1,
            "background": 0,
        }

        # Number of input channels
        self.number_intensity_ch = 1

        # Change the default network to SegResNet
        network = self.conf.get("network", "segresnet")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{self.name}_{network}.pt"),
            os.path.join(self.model_dir, f"{self.name}_{network}.pt"),
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "false")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_deepedit_{network}_multilabel.pt"
            download_file(url, self.path[0])

        # Optimise the spatial size and batch handling for the 5090D 32GB of VRAM
        self.target_spacing = (1.0, 1.0, 1.0)
        # Increase the spatial size for better segmentation quality (the 5090D 32GB of VRAM can support a larger input)
        self.spatial_size = (192, 192, 192)  # raised from 160 to 192

        # Network - make SegResNet the default and optimise the parameters
        if network == "segresnet":
            print("Using SegResNet network for DeepEdit (Optimized for 5090D 32GB)")
            # SegResNet parameters optimised for 32GB of VRAM
            self.network = SegResNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                init_filters=32,  # increase the initial filter count to improve representational capacity
                blocks_down=(1, 2, 2, 4, 4),  # a deeper network structure
                blocks_up=(1, 1, 1, 1),
                dropout_prob=0.1,  # mild dropout to prevent overfitting
                norm="INSTANCE",  # Instance Norm works better for 3D images
                use_conv_final=True,  # use a final convolution layer to improve accuracy
            )
        elif network == "segresnet_large":
            print("Using SegResNet Large network for DeepEdit (Maximum quality for 5090D 32GB)")
            # SegResNet configuration maximising quality
            self.network = SegResNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                init_filters=48,  # a larger filter count
                blocks_down=(2, 3, 3, 5, 5),  # a deeper network
                blocks_up=(1, 1, 1, 1),
                dropout_prob=0.1,
                norm="INSTANCE",
                use_conv_final=True,
            )
        elif network == "swinunetr":
            print("Using SwinUNETR network for DeepEdit")
            feature_size = int(self.conf.get("feature_size", "48"))
            use_checkpoint = strtobool(self.conf.get("use_checkpoint", "false"))
            
            self.network = SwinUNETR(
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
                img_size=self.spatial_size,  # add the img_size argument
            )
        elif network == "unetr":
            print("Using UNETR network for DeepEdit")
            self.network = UNETR(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                img_size=self.spatial_size,
                feature_size=72,
                hidden_size=1536,
                mlp_dim=3072,
                num_heads=32,
                norm_name="instance",
                res_block=True,
            )
        else:
            print("Using DynUNet network for DeepEdit")
            self.network = DynUNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
                act_name=("LEAKYRELU", {"inplace": True, "negative_slope": 0.01}),
            )

        # Network with dropout for epistemic uncertainty
        if network in ["segresnet", "segresnet_large"]:
            # the epistemic version of SegResNet
            init_filters = 48 if network == "segresnet_large" else 32
            blocks_down = (2, 3, 3, 5, 5) if network == "segresnet_large" else (1, 2, 2, 4, 4)
            
            self.network_with_dropout = SegResNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                init_filters=init_filters,
                blocks_down=blocks_down,
                blocks_up=(1, 1, 1, 1),
                dropout_prob=0.2,  # increase dropout for uncertainty estimation
                norm="INSTANCE",
                use_conv_final=True,
            )
        elif network == "swinunetr":
            feature_size = int(self.conf.get("feature_size", "48"))
            use_checkpoint = strtobool(self.conf.get("use_checkpoint", "false"))
            
            self.network_with_dropout = SwinUNETR(
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
                drop_rate=0.2,
                img_size=self.spatial_size,
            )
        elif network == "unetr":
            self.network_with_dropout = UNETR(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                img_size=self.spatial_size,
                feature_size=72,
                hidden_size=1536,
                mlp_dim=3072,
                num_heads=32,
                norm_name="instance",
                res_block=True,
                dropout_rate=0.2,
            )
        else:
            self.network_with_dropout = DynUNet(
                spatial_dims=3,
                in_channels=len(self.labels) + self.number_intensity_ch,
                out_channels=len(self.labels),
                kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                norm_name="instance",
                deep_supervision=False,
                res_block=True,
                dropout=0.2,
                act_name=("LEAKYRELU", {"inplace": True, "negative_slope": 0.01}),
            )

        # Compute and print the number of model parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info(f"Network: {network}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Estimated memory usage: {total_params * 4 / (1024**3):.2f} GB (FP32)")

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")
        
        # Training configuration optimised for the 5090D
        # Note: BasicTrainTask uses train_batch_size / val_batch_size, not batch_size
        self.train_config = {
            "amp": False,  # enable automatic mixed precision
            "train_batch_size": 1,  # training batch size; starting small is advisable to avoid numerical instability
            "val_batch_size": 1,
            "pretrained": strtobool(self.conf.get("use_pretrained_model", "true")),
        }

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                config=self.train_config,
            ),
            f"{self.name}_seg": lib.infers.DeepEdit(
                path=self.path,
                network=self.network,
                labels=self.labels,
                preload=strtobool(self.conf.get("preload", "false")),
                spatial_size=self.spatial_size,
                target_spacing=self.target_spacing,
                number_intensity_ch=self.number_intensity_ch,
                type=InferType.SEGMENTATION,
                config=self.train_config,
            ),
        }

    def trainer(self) -> Optional[TrainTask]:
        network = self.conf.get("network", "segresnet")
        output_dir = os.path.join(self.model_dir, f"{self.name}_" + network)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.DeepEdit(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            number_intensity_ch=self.number_intensity_ch,
            config={
                "pretrained": strtobool(self.conf.get("use_pretrained_model", "true")),
                **self.train_config,
            },
            labels=self.labels,
            debug_mode=False,
            find_unused_parameters=True,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=self.network_with_dropout,
                transforms=lib.infers.DeepEdit(
                    type=InferType.DEEPEDIT,
                    path=self.path,
                    network=self.network,
                    labels=self.labels,
                    preload=strtobool(self.conf.get("preload", "false")),
                    spatial_size=self.spatial_size,
                ).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        return methods