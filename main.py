import os
import warnings

warnings.filterwarnings("ignore")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))
from itertools import chain

import hydra
from hydra.utils import instantiate

from typing import Optional
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torchmetrics.functional.image import image_gradients
from typing import Any, Callable, Dict, Optional, Tuple, List
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer, LightningModule

from omegaconf import OmegaConf
from PIL import Image

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from datamodule import UnpairedDataModule


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class NVLightningModule(LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        # @ Diffusion
        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            with_conditioning=False,
            # cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        )
        # init_weights(self.unet2d_model, init_type="normal")

        self.ddpmsch = DDPMScheduler(
            num_train_timesteps=self.model_cfg.timesteps,
            prediction_type=self.model_cfg.prediction_type,
            schedule="scaled_linear_beta",
            beta_start=0.0005,
            beta_end=0.0195,
        )

        self.ddimsch = DDIMScheduler(
            num_train_timesteps=self.model_cfg.timesteps,
            prediction_type=self.model_cfg.prediction_type,
            schedule="scaled_linear_beta",
            beta_start=0.0005,
            beta_end=0.0195,
        )

        self.inferer = DiffusionInferer(scheduler=self.ddimsch)

        if self.model_cfg.phase == "finetune":
            pass

        if self.train_cfg.ckpt:
            print("Loading.. ", self.train_cfg.ckpt)
            checkpoint = torch.load(
                self.train_cfg.ckpt, map_location=torch.device("cpu")
            )["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)

        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def forward_timing(self, image2d, noise=None, timesteps=None):
        _device = image2d.device
        B = image2d.shape[0]
        timesteps = (
            torch.zeros((B,), device=_device).long() if timesteps is None else timesteps
        )

        results = self.inferer(
            inputs=image2d,
            noise=noise,
            diffusion_model=self.unet2d_model,
            timesteps=timesteps,
        )
        return results

    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image2d = batch["image2d"]
        _device = batch["image2d"].device
        B = image2d.shape[0]

        timesteps = torch.randint(
            0, self.ddpmsch.num_train_timesteps, (B,), device=_device
        ).long()

        noise2d = torch.randn_like(image2d)

        output = self.forward_timing(
            image2d=image2d,
            noise=noise2d,
            timesteps=timesteps,
        )

        # Set the target
        if self.ddpmsch.prediction_type == "sample":
            target = image2d
        elif self.ddpmsch.prediction_type == "epsilon":
            target = noise2d
        elif self.ddpmsch.prediction_type == "v_prediction":
            target = self.ddpmsch.get_velocity(image2d, noise2d, timesteps)

        im2d_loss_dif = F.l1_loss(output, target)

        im2d_loss = im2d_loss_dif
        self.log(
            f"{stage}_im2d_loss",
            im2d_loss,
            on_step=(stage == "train"),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=B,
        )
        loss = self.train_cfg.alpha * im2d_loss

        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                sample = noise2d

                self.ddimsch.set_timesteps(num_inference_steps=self.model_cfg.timesteps//10)
                sample = self.inferer.sample(
                    input_noise=sample,
                    scheduler=self.ddimsch,
                    diffusion_model=self.unet2d_model,
                    verbose=False,
                )

                viz2d = torch.cat(
                    [
                        torch.cat([image2d, noise2d], dim=-2).transpose(2, 3),
                        torch.cat([output, sample], dim=-2).transpose(2, 3),
                    ],
                    dim=-2,
                )

                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(
                    viz2d, normalize=False, scale_each=False, nrow=1, padding=0
                ).clamp(0, 1)
                tensorboard.add_image(
                    f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx
                )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(
            f"train_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()  # free memory

    def sample(self, **kwargs: dict):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_cfg.lr, betas=(0.5, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,  # milestones=[100, 200, 300, 400], gamma=0.5
        )
        return [optimizer], [scheduler]


@hydra.main(version_base=None, config_path="./conf")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all str interpolation
    seed_everything(42)
    datamodule = UnpairedDataModule(
        train_image2d_folders=cfg.data.train_image2d_folders,
        val_image2d_folders=cfg.data.val_image2d_folders,
        test_image2d_folders=cfg.data.test_image2d_folders,
        shape=cfg.data.shape,
        batch_size=cfg.data.batch_size,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
    )

    model = NVLightningModule(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
    )
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks]
    logger = [hydra.utils.instantiate(c) for c in cfg.logger]

    trainer = Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    trainer.fit(
        model,
        # datamodule=datamodule,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        # ckpt_path=cfg.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
