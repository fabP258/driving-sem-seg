from model import SegmentationModel
from pathlib import Path


def main():
    model = SegmentationModel(
        data_path=Path("/home/fabio/Data/comma10k"),
        backbone="efficientnet-b4",
        batch_size=10,
    )
    model.prepare_data()
    model.setup_datasets()
    model.configure_optimizer()
    model.train()


if __name__ == "__main__":
    main()
