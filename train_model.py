from model import SegmentationModel
from pathlib import Path


def main():
    # pretrain on low resolution
    model = SegmentationModel(
        data_path=Path("/home/fabio/Data/comma10k"),
        logs_path=Path("/home/fabio/Repos/driving-sem-seg/logs/pre-train"),
        backbone="efficientnet-b4",
        batch_size=10,
        epochs=10,
        image_size=(576, 448),
        lr=1e-2,
    )
    model.prepare_data()
    model.setup_datasets()
    model.configure_optimizer()
    model.train()

    # fine-tune on full resolution
    model.batch_size = 2
    model.epochs = 10
    model.set_logs_path(Path("/home/fabio/Repos/driving-sem-seg/logs/fine-tune"))
    model.update_image_size((1184, 896))
    model.configure_optimizer()
    model.train()


if __name__ == "__main__":
    main()
