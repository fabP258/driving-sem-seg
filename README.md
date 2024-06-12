# driving-sem-seg

A semantic segmentation reimplementation in plain pytorch based on the [comma10k-baseline](https://github.com/YassineYousfi/comma10k-baseline).
The repository builds on top of
* the open source [comma10k dataset](https://github.com/commaai/comma10k) containing labelled images of driving videos
* a pre-assembled U-Net from [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)

## Next steps
* Add more image augmentations for training
* Add ONNX model export after training

## Experiments
* **Baseline**: 
  * Two-step training without augmentations, no learning rate scheduler, efficientnet-b0 encoder, learning rate 1e-4
  * 20 epochs on low resolution, 20 epochs on full resolution
  * Test loss: 0.0918
* **Baseline + Augmentations**
  * Baseline + 
    * horizontal flip with $p=0.5$
    * GaussianNoise with var_limits=(10.0,50.0)
    * Test loss: 1.3329
    * Train loss: 0.1272
  * Aug2
* **Baseline on bigger net**
  * Efficientnet-b4
  * No augmentations
  * Test loss: 0.0997
  * Train loss: 0.088
* **Baseline on bigger net with scheduler**
  * ExponentialLR with $\gamma = 0.9$
  * Train loss: 0.0616
  * Test loss: 0.0774
* **Efficientnet-B4 + Scheduler + Light augmentations**
  * horizontal flip with $p=0.5$
  * ExponentialLR with $\gamma = 0.9$
  * Train loss: 0.0642
  * Test loss: 0.0757
* **Efficientnet-B4 + Scheduler + More augmentations**
  * horizontal flip with $p=0.5$
  * GaussianNoise with var_limits=(10.0,20.0)
  * ExponentialLR with $\gamma = 0.9$
  * LR epochs: 50, HR epochs: 25
  * Train loss: 0.2332
  * Test loss: 0.2518