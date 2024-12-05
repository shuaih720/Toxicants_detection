AI-driven tool fusing deep learning and human-concept features enhances oocyte toxicity assessment, achieving high accuracy and supporting fertility research.
Companion code to the paper "Automated Detection and Recognition of Oocyte Toxicity by Fusion of Latent and Observable Features".

<!-- https://www.xxxxxxxxx.com/

--------------------

Citation:
```
Authors et al. Automated Detection and Recognition of Oocyte Toxicity by Fusion of Latent and Observable Features.
journal. doi
```

Bibtex:
```
@article{,
  title = {Automated Detection and Recognition of Oocyte Toxicity by Fusion of Latent and Observable Features},
  author = {},
  year = {},
  volume = {},
  pages = {},
  doi = {},
  journal = {},
  number = {}
}
```
-------------------- -->

## Abstract information

Oocyte quality is essential for successful embryo development, yet no standardized methods currently exist to assess the effects of toxic pollutants like per- and polyfluoroalkyl substances (PFAS) and short chain chlorinated paraffins (SCCP) on oocyte abnormalities. This study strengthen oocyte image analysis using a stepwise automated method focused on toxicity detection, subtype and strength classification. By fusing deep learning-extracted latent features with observable human-concept features, this method achieves performance surpassing human capabilities with ROC-AUC of 0.9087 for toxicity detection, 0.7956 to 0.9034 for subtype classification and 0.6434 to 0.9062 for toxicity strength classification based on 2,126 images from 16-hour exposure group. Notably, ablation experiments show fusing features from both domains outperforms using each domain's features independently, highlighting their complementary relationship. To improve interpretability, personalized heatmaps and feature importance are provided. This study provides an AI-based tool for assessing the toxic effects on oocyte quality, providing support for predicting pregnancy outcome. 

--------------------
## Requirements

This code was tested on Python 3 with Tensorflow `2.6.0`

In addition, the packages we are calling now is as follows:
- [x] tensorflow2.0     
- [x] sklearn
- [x] random
- [x] scipy
- [x] pandas
- [x] numpy
- [x] tabnet

## Install from Github
```python
python
>>> git clone https://github.com/shuaih720/Toxicants_detection

## Instructions for use
running the integrated version 
```python
python
>>> python celldet_model.py
```
Training the neural network and generating the neural network predictions on given datasets.
## License

This project is covered under the Apache 2.0 License.

