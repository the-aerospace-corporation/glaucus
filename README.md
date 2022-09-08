

![Glaucus Atlanticus](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Glaucus_atlanticus_1_cropped.jpg/247px-Glaucus_atlanticus_1_cropped.jpg)

# Glaucus

The Aerospace Corporation is proud to present our complex-valued encoder, decoder, and loss for RF DSP in PyTorch.

## Using

### Install

* via PyPI: `pip install glaucus`
* via source: `pip install .`

### Testing

* `coverage run -a --source=glaucus -m pytest --doctest-modules; coverage html`
* `pytest .`

### Use our pre-trained model

Load quantized model and return compressed signal vector & reconstruction.
Our weights were trained & evaluated on a corpus of 200GB of RF waveforms with
various added RF impairments for a 40TB training set.

```python
import torch
import sigmf
from glaucus import GlaucusAE

# create model
model = GlaucusAE(bottleneck_quantize=True)
model = torch.quantization.prepare(model)
# get weights for quantized model
state_dict = torch.hub.load_state_dict_from_url('https://pending-torch-hub-submission/ae-quantized.pth')
model.load_state_dict(state_dict)
# prepare for prediction
model.eval()
torch.quantization.convert(model), inplace=True)
# get samples into NCL tensor
x_sigmf = sigmf.sigmffile.fromfile('example.sigmf')
x_np = x_sigmf.read_samples()
x_tensor = torch.view_as_real(torch.from_numpy(x_np)).swapaxes(-1, -2).unsqueeze(0)
# create prediction & quint8 signal vector
y_tensor, y_encoded = model(x_samples)
# get signal vector as uint8
y_encoded_uint8 = torch.int_repr(y_encoded)
```

### Train model with some RF

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from glaucus import GlaucusAE
model = GlaucusAE() # or FullyConnectedAE
loader = DataModule() # Not provided
early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename='glaucus-{epoch:03d}-{val_loss:05f}')
# may want to specify GPUs/TPUs here
trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping_callback])
trainer.fit(model, loader)
# rewind to best checkpoint
model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False)
```

## Papers

This code is documented by the two following IEEE publications.

### Glaucus: A Complex-Valued Radio Signal Autoencoder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5806615.svg)](https://doi.org/10.5281/zenodo.5806615)

A complex-valued autoencoder neural network capable of compressing \& denoising radio frequency (RF) signals with arbitrary model scaling is proposed. Complex-valued time samples received with various impairments are decoded into an embedding vector, then encoded back into complex-valued time samples. The embedding and the related latent space allow search, comparison, and clustering of signals. Traditional signal processing tasks like specific emitter identification, geolocation, or ambiguity estimation can utilize multiple compressed embeddings simultaneously. This paper demonstrates an autoencoder implementation capable of 64x compression hardened against RF channel impairments. The autoencoder allows separate or compound scaling of network depth, width, and resolution to target both embedded and data center deployment with differing resources. The common building block is inspired by the Fused Inverted Residual Block (Fused-MBConv), popularized by EfficientNetV2 \& MobileNetV3, with kernel sizes more appropriate for time-series signal processing

### Complex-Valued Radio Signal Loss for Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5806615.svg)](https://doi.org/10.5281/zenodo.5806615)

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
