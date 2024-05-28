# driving-sem-seg

A semantic segmentation reimplementation in plain pytorch based on the [comma10k-baseline](https://github.com/YassineYousfi/comma10k-baseline).
The repository builds on top of
* the open source [comma10k dataset](https://github.com/commaai/comma10k) containing labelled images of driving videos
* a pre-assembled U-Net from [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)

## Next steps
* Enable two-stage training for
  * pre-training on lower resolution
  * fine-tuning on full resolution
