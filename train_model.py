from model import SegmentationModel
from pathlib import Path


def main():
    # pretrain on low resolution
    model = SegmentationModel(
        data_path=Path("/home/fabio/Data/comma10k"),
        logs_path=Path("/home/fabio/Repos/driving-sem-seg/logs/pre-train"),
        backbone="efficientnet-b0",
        batch_size=20,
        epochs=20,
        image_size=(576, 448),
    )
    model.prepare_data()
    model.setup_datasets()
    model.configure_optimizer()
    model.train()

    # fine-tune on full resolution
    # TODO: use padding instead of plain resize (pad_if_necessary())
    model.batch_size = 5
    model.epochs = 20
    model.set_logs_path(Path("/home/fabio/Repos/driving-sem-seg/logs/fine-tune"))
    model.update_image_size((1184, 896))
    model.train()


if __name__ == "__main__":
    main()
