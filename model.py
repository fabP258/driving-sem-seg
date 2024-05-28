from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union, List
import segmentation_models_pytorch as smp
from data_handling import SegmentationDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F


class SegmentationModel:

    def __init__(
        self,
        data_path: Union[str, Path],
        logs_path: Union[str, Path],
        backbone: str = "efficientnet-b0",
        batch_size: int = 16,
        lr: float = 1e-4,
        eps: float = 1e-7,
        height: int = 14 * 32,
        width: int = 18 * 32,
        epochs: int = 50,
        weight_decay: float = 1e-3,
        class_values: List[int] = [42, 76, 90, 124, 161],
    ):
        self.data_path = Path(data_path)
        Path(logs_path).mkdir(parents=True, exist_ok=True)
        self.logs_path = logs_path
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.height = height
        self.width = width
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.class_values = class_values

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.__build_model()

    def __build_model(self):
        self.net = smp.Unet(
            self.backbone,
            classes=len(self.class_values),
            activation=None,
            encoder_weights="imagenet",
        )
        self.net.to(self.device)

        self.loss_func = lambda x, y: torch.nn.CrossEntropyLoss()(
            x, torch.argmax(y, axis=1)
        )
        self.loss_func = torch.nn.CrossEntropyLoss()

    def prepare_data(self):
        """Data download is not part of this script
        Get the data from https://github.com/commaai/comma10k
        """
        assert (self.data_path / "imgs").is_dir(), "Images not found"
        assert (self.data_path / "masks").is_dir(), "Masks not found"
        assert (
            self.data_path / "files_trainable"
        ).exists(), "Files trainable file not found"

        print("data ready")

    def setup_datasets(self):

        image_names = np.loadtxt(
            self.data_path / "files_trainable", dtype="str"
        ).tolist()

        # do we need to shuffle?
        # TODO: delete
        # random.shuffle(image_names)

        # train
        train_file_names = [
            x.split("masks/")[-1] for x in image_names if not x.endswith("9.png")
        ]
        self.train_dataset = SegmentationDataset(
            image_dir=self.data_path / "imgs",
            mask_dir=self.data_path / "masks",
            file_names=set(train_file_names),
            augment=True,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # validation
        valid_file_names = [
            x.split("masks/")[-1] for x in image_names if x.endswith("9.png")
        ]
        self.valid_dataset = SegmentationDataset(
            image_dir=self.data_path / "imgs",
            mask_dir=self.data_path / "masks",
            file_names=set(valid_file_names),
        )
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )

    def configure_optimizer(self):
        optimizer_kwargs = {"eps": self.eps}
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **optimizer_kwargs,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9
        )

    def train_loop(self):
        size = len(self.train_dataloader.dataset)

        # set net to train mode
        self.net.train()
        epoch_loss = 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            y_logits = self.net(X.to(self.device))
            loss = self.loss_func(y_logits, y.to(self.device))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * self.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        self.scheduler.step()

        return epoch_loss / len(self.train_dataloader)

    def test_loop(self, epoch=int):
        self.net.eval()
        size = len(self.valid_dataloader.dataset)
        num_batches = len(self.valid_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.valid_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.net(X)
                test_loss += self.loss_func(y_logits, y).item()
                correct += (y_logits.argmax(1) == y).type(torch.float).sum().item()

                if batch == 0:
                    # convert tensors to np
                    X_array = X.cpu().numpy()
                    y_mask = y_logits.argmax(1)
                    y_mask_array = y_mask.cpu().numpy()

                    # convert first batch entry label to grayscale
                    pred_mask = np.vectorize(self.valid_dataset.index_to_value.get)(
                        y_mask_array[0, :, :]
                    )
                    fig, ax = plt.subplots()
                    ax.imshow(np.transpose(X_array[0, :, :, :], (1, 2, 0)))
                    ax.axis("off")
                    ax.imshow(pred_mask, alpha=0.2, cmap="jet")
                    fig.savefig(self.logs_path / f"{str(epoch)}.png")

        width, height = pred_mask.shape[0], pred_mask.shape[1]
        test_loss /= num_batches
        correct /= size * width * height

        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

        return test_loss

    def train(self):
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross entropy loss (log)")

        epochs = []
        train_errors = []
        valid_errors = []
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = self.train_loop()
            valid_loss = self.test_loop(t)

            # logging
            epochs.append(t)
            train_errors.append(train_loss)
            valid_errors.append(valid_loss)

            if t % 5 == 0:
                ax.scatter(epochs, train_errors, color="b")
                ax.scatter(epochs, valid_errors, color="r")
                fig.savefig(self.logs_path / f"loss_{str(t)}.png")
                epochs = []
                train_errors = []
                valid_errors = []
