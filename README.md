![Glaucus Atlanticus](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Glaucus_atlanticus_1_cropped.jpg/247px-Glaucus_atlanticus_1_cropped.jpg)

# Glaucus

The Aerospace Corporation is proud to present our complex-valued encoder,
decoder, and a new loss function for RF DSP in PyTorch.

## Video (click to play)

[<img src="https://i.vimeocdn.com/video/1583946742-851ad3621192f133ca667bc87f4050276e450fcc721f117bbcd93b67cb0535f8-d_1000">](https://vimeo.com/787670661/ce13da4cd9)

## Using

### Install

* via PyPI: `pip install glaucus`
* via source: `pip install .`

### Testing

* `coverage run -a --source=glaucus -m pytest --doctest-modules; coverage html`
* `pytest .`

### Use pre-trained model with SigMF data

Load quantized model and return compressed signal vector & reconstruction.
Our weights were trained & evaluated on a corpus of 200GB of RF waveforms with
various added RF impairments for a 40TB training set.

```python
import torch
import sigmf
from glaucus import GlaucusAE

# create model
model = GlaucusAE(bottleneck_quantize=True, data_format='nl')
model = torch.quantization.prepare(model)
# get weights for quantized model
state_dict = torch.hub.load_state_dict_from_url(
    'https://github.com/the-aerospace-corporation/glaucus/releases/download/v1.1.0/glaucus-512-3275-5517642b.pth',
    map_location='cpu')
model.load_state_dict(state_dict)
# prepare for prediction
model.eval()
torch.quantization.convert(model, inplace=True)
# get samples into NL tensor
x_sigmf = sigmf.sigmffile.fromfile('example.sigmf')
x_tensor = torch.from_numpy(x_sigmf.read_samples())
# create prediction & quint8 signal vector
y_tensor, y_encoded = model(x_samples)
# get signal vector as uint8
y_encoded_uint8 = torch.int_repr(y_encoded)
```

### Pre-trained Model List

| desc      | link                                                                                                                                     | size  | params  | multiadds | provenance                                                    |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|-------|---------|-----------|---------------------------------------------------------------|
| fastest   | [glaucus-512-3275-5517642b](https://github.com/the-aerospace-corporation/glaucus/releases/download/v1.1.0/glaucus-512-3275-5517642b.pth) | 8.5 M | 2.030 M | 259 M     | .009 pfs-days on modulation-only Aerospace DSet               |
| accurate  | [glaucus-1024-761-c49063fd](https://github.com/the-aerospace-corporation/glaucus/releases/download/v1.1.0/glaucus-1024-761-c49063fd.pth) | 11 M  | 2.873 M | 380 M     | .035 pfs-days modulation & general waveform Aerospace Dset    |
| -pending- |                                                                                                                                          | 11 M  | 2.873 M | 380 M     | transfer learning from glaucus-1024-761-c49063fd w/Sig53 Dset |

#### Note on pfs-days

Per [OpenAI appendix](https://openai.com/blog/ai-and-compute/#appendixmethods) here is the correct math (method 1):

* `pfs_days` = (add-multiplies per forward pass) * (2 FLOPs/add-multiply) * (3 for forward and backward pass) * (number of examples in dataset) * (number of epochs) / (flop per petaflop) / (seconds per day)
* (number of examples in dataset) * (number of epochs) = steps * batchsize
* 1 `pfs-day` ≈ (8x V100 GPUs at 100% efficiency for 1 day) ≈ (100x GTX1080s at 100% efficiency for 1 day) ≈ (35x GTX 2080s at 100% efficiency for 1 day) ≈ [500 kWh](https://twitter.com/id_aa_carmack/status/1192513743974019072)

### Get loss between two RF signals

```python
import np
import torch
import glaucus

# create criterion
loss = glaucus.RFLoss(spatial_size=128, data_format='nl')

# create some signal
xxx = torch.randn(128, dtype=torch.complex64)
# alter signal with 1% freq offset
yyy = xxx * np.exp(1j * 2 * np.pi * 0.01 * np.arange(128))

# return loss
loss(xxx, yyy)
```

### Train model with TorchSig

*partially implemented pending update or replace with notebook example*

```python
import lightning as pl
from glaucus import GlaucusAE

model = GlaucusAE(data_format='nl')

# this takes a very long time if no cache is available
signal_data = torchsig.datasets.Sig53(root=str(cache_path))
# 80 / 10 / 10 split
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    signal_data,
    (len(signal_data)*np.array([0.8, 0.1, 0.1])).astype(int),
    generator=torch.Generator().manual_seed(0xcab005e)
)
class RFDataModule(pl.LightningDataModule):
    '''
    defines the dataloaders for train, val, test and uses datasets
    '''
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None,
                 num_workers=16, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, pin_memory=True)

loader = RFDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=batch_size, num_workers=num_workers)

trainer = pl.Trainer()
trainer.fit(model, loader)

# rewind to best checkpoint
model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
```

## Papers

This code is documented by the two following IEEE publications.

### Glaucus: A Complex-Valued Radio Signal Autoencoder

A complex-valued autoencoder neural network capable of compressing & denoising radio frequency (RF) signals with arbitrary model scaling is proposed. Complex-valued time samples received with various impairments are decoded into an embedding vector, then encoded back into complex-valued time samples. The embedding and the related latent space allow search, comparison, and clustering of signals. Traditional signal processing tasks like specific emitter identification, geolocation, or ambiguity estimation can utilize multiple compressed embeddings simultaneously. This paper demonstrates an autoencoder implementation capable of 64x compression hardened against RF channel impairments. The autoencoder allows separate or compound scaling of network depth, width, and resolution to target both embedded and data center deployment with differing resources. The common building block is inspired by the Fused Inverted Residual Block (Fused-MBConv), popularized by EfficientNetV2 \& MobileNetV3, with kernel sizes more appropriate for time-series signal processing

### Complex-Valued Radio Signal Loss for Neural Networks

A new optimized loss for training complex-valued neural networks that require reconstruction of radio signals is proposed. Given a complex-valued time series this method incorporates loss from spectrograms with multiple aspect ratios, cross-correlation loss, and loss from amplitude envelopes in the time \& frequency domains. When training a neural network an optimizer will observe batch loss and backpropagate this value through the network to determine how to update the model parameters. The proposed loss is robust to typical radio impairments and co-channel interference that would explode a naive mean-square-error approach. This robust loss enables higher quality steps along the loss surface which enables training of models specifically designed for impaired radio input. Loss vs channel impairment is shown in comparison to mean-squared error for an ensemble of common channel effects.

## Contributing

Do you have code you would like to contribute to this Aerospace project?

We are excited to work with you. We are able to accept small changes
immediately and require a Contributor License Agreement (CLA) for larger
changesets. Generally documentation and other minor changes less than 10 lines
do not require a CLA. The Aerospace Corporation CLA is based on the well-known
[Harmony Agreements CLA](http://harmonyagreements.org/) created by Canonical,
and protects the rights of The Aerospace Corporation, our customers, and you as
the contributor. [You can find our CLA here](https://aerospace.org/sites/default/files/2020-12/Aerospace-CLA-2020final.pdf).

Please complete the CLA and send us the executed copy. Once a CLA is on file we
can accept pull requests on GitHub or GitLab. If you have any questions, please
e-mail us at [oss@aero.org](mailto:oss@aero.org).

## Licensing

The Aerospace Corporation supports Free & Open Source Software and we publish
our work with GPL-compatible licenses. If the license attached to the project
is not suitable for your needs, our projects are also available under an
alternative license. An alternative license can allow you to create proprietary
applications around Aerospace products without being required to meet the
obligations of the GPL. To inquire about an alternative license, please get in
touch with us at [oss@aero.org](mailto:oss@aero.org).

## To-Do

* insert DOI links once papers are assigned DOI like [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5806615.svg)](https://doi.org/10.5281/zenodo.5806615)
* update this readme with published model weight path
* upload training notebook
