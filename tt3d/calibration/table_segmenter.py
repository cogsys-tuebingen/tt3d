"""
Segmentation model for table tennis table
"""
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
from torchvision import transforms as T


class TableSegmenter(L.LightningModule):
    def __init__(
        self,
        arch="Unet",
        encoder_name="mobilenet_v2",
        in_channels=3,
        out_classes=1,
        loss="DICE",
        **kwargs,
    ):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Threshold for the mask from the heatmap
        self.thres = 0.8

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        if loss == "DICE":
            self.loss_fn = smp.losses.DiceLoss(
                mode=smp.losses.BINARY_MODE,
            )
        if loss == "BCE+DICE":
            self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()
            self.dice_loss = smp.losses.DiceLoss(
                mode=smp.losses.BINARY_MODE,
            )
        elif loss == "Jaccard":
            self.loss_fn = smp.losses.JaccardLoss(
                mode=smp.losses.BINARY_MODE,
            )
        elif loss == "Lovasz":
            self.loss_fn = smp.losses.LovaszLoss(
                mode=smp.losses.BINARY_MODE,
            )
        elif loss == "BCE":
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        else:
            raise ValueError("Loss not recognized")
        self.validation_step_metrics = []
        self.train_step_metrics = []
        self.save_hyperparameters()
        print("Model initialized")

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        output = self.model(image)
        return output

    def infer(self, image):
        logits_mask = self(image)
        mask = self.process_logits(logits_mask)
        return mask

    def process_logits(self, logits_mask):
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > self.thres).float()
        return pred_mask

    def wb_mask(self, img, pred_mask, true_mask):
        # pred_mask = np.zeros((320, 640))
        # pred_mask[50:100, 50:100] = 1
        return wandb.Image(
            img,
            masks={
                "prediction": {
                    "mask_data": pred_mask,
                    "class_labels": {0: "bg", 1: "table"},
                },
                "ground_truth": {
                    "mask_data": true_mask,
                    "class_labels": {0: "bg", 1: "table"},
                },
            },
        )

    def tensor2np(self, tensor):
        array = tensor.clone().detach().cpu().permute(1, 2, 0).numpy()
        return array

    def log_images(self, images, pred_masks, true_masks, stage):
        predictions = []
        table = wandb.Table(columns=["ID", "Image"])
        for id, values in enumerate(zip(images, pred_masks, true_masks)):
            img, pred_mask, true_mask = values
            img = self.tensor2np(img)
            pred_mask = np.squeeze(pred_mask.cpu().numpy()).astype(np.uint8)
            true_mask = np.squeeze(true_mask.cpu().numpy()).astype(np.uint8)
            # print(pred_mask.dtype)
            # plt.imshow(pred_mask)
            # plt.show()
            # pred_mask = np.zeros_like(pred_mask)
            # pred_mask[:50, :50] = 1
            mask_img = self.wb_mask(img, pred_mask, true_mask)
            predictions.append(mask_img)
            table.add_data(id, mask_img)

        wandb.log({f"{stage}_predictions": predictions})

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        # print(outputs)

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # print(tp[:10])
        # print(fp[:10])
        # print(fn[:10])
        # print(tn[:10])

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        if stage == "val":
            self.log_images(
                self.valid_images, self.valid_pred_masks, self.valid_true_masks, stage
            )
        if stage == "train":
            # print(self.train_true_masks[0])
            # print(self.train_true_masks[1])
            # f, axs = plt.subplots(2)
            # axs[0].imshow(self.tensor2np(self.train_pred_masks[0]))
            # axs[1].imshow(self.tensor2np(self.train_true_masks[0]))
            # plt.show()
            self.log_images(
                self.train_images, self.train_pred_masks, self.train_true_masks, stage
            )
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        image, mask = batch

        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        # transformed_img = T.ToPILImage()(image[0])
        # transformed_mask = T.ToPILImage()(mask[0])
        # plt.imshow(transformed_img)
        # plt.imshow(transformed_mask, alpha=0.5)
        # plt.show()

        logits_mask = self(image)
        if self.loss_name == "BCE+DICE":
            loss = 0.5 * self.dice_loss(logits_mask, mask) + 0.5 * self.bce_loss(
                logits_mask, mask
            )
        else:
            loss = self.loss_fn(logits_mask, mask)

        pred_mask = self.process_logits(logits_mask)

        metrics = self.get_metrics(pred_mask, mask)
        self.train_step_metrics.append(metrics)

        self.train_images = image.clone().detach()
        self.train_true_masks = mask.clone().detach()
        self.train_pred_masks = pred_mask.clone().detach()

        self.log("train_loss", loss)
        return loss

    def get_metrics(self, pred_mask, gt_mask):
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        # plt.imshow(gt_mask.clone().detach().cpu().long()[0, 0])
        # plt.imshow(pred_mask.clone().detach().cpu().long()[0, 0])
        # plt.show()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            gt_mask.long(),
            mode="binary",
        )
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        # print(metrics)
        return metrics

    def evaluate(self, batch, stage=None):
        image, mask = batch

        # plt.imshow(mask.clone().detach().cpu().long()[0, 0])
        # plt.imshow(pred_mask.clone().detach().cpu().long()[0, 0])
        # plt.show()
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        # print(torch.sum(mask == 1))

        logits_mask = self(image)
        if self.loss_name == "BCE+DICE":
            loss = 0.5 * self.dice_loss(logits_mask, mask) + 0.5 * self.bce_loss(
                logits_mask, mask
            )
        else:
            loss = self.loss_fn(logits_mask, mask)

        pred_mask = self.process_logits(logits_mask)

        metrics = self.get_metrics(pred_mask, mask)

        if stage == "val":
            self.valid_images = image
            self.valid_true_masks = mask
            self.valid_pred_masks = pred_mask
            self.validation_step_metrics.append(metrics)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            # self.log(f"{stage}_IOU", acc, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.train_step_metrics, "train")
        self.train_step_metrics.clear()
        # return None

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_metrics, "val")
        self.validation_step_metrics.clear()
        # return None

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    # def test_epoch_end(self):
    #     return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
