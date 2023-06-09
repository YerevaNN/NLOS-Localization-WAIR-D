# ML-based Approaches for Wireless NLOS Localization: Input Representations and Uncertainty Estimation

This code replicates the experiments from the following paper:

> Rafayel Darbinyan, Hrant Khachatrian, Rafayel Mkrtchyan, Theofanis P. Raptis
>
> [ML-based Approaches for Wireless NLOS Localization: Input Representations and Uncertainty Estimation](https://arxiv.org/abs/2304.11396)


## Abstract

The challenging problem of non-line-of-sight (NLOS) localization is critical for many wireless networking applications. The lack of available datasets has made NLOS localization difficult to tackle with ML-driven methods, but recent developments in synthetic dataset generation have provided new opportunities for research. This paper explores three different input representations: (i) single wireless radio path features, (ii) wireless radio link features (multi-path), and (iii) image-based representations. Inspired by the two latter new representations, we design two convolutional neural networks (CNNs) and we demonstrate that, although not significantly improving the NLOS localization performance, they are able to support richer prediction outputs, thus allowing deeper analysis of the predictions. In particular, the richer outputs enable reliable identification of non-trustworthy predictions and support the prediction of the top-K candidate locations for a given instance. We also measure how the availability of various features (such as angles of signal departure and arrival) affects the model's performance, providing insights about the types of data that should be collected for enhanced NLOS localization. Our insights motivate future work on building more efficient neural architectures and input representations for improved NLOS localization performance, along with additional useful application features.

## Project setup

1. Duplicate the `.env.sample` file and rename the copy as `.env`.
2. Download the WAIR-D dataset from the following link: https://www.mobileai-dataset.com/html/default/yingwen/DateSet/1590994253188792322.html?index=1
3. Open the `.env` file and insert the path of the downloaded dataset as the value for `RAW_DATA_DIR`.
4. Create a new conda environment using the command:
   ```commandline
   conda env create 
   ```
5. Activate the newly created conda environment with the command:
   ```commandline
   conda activate wair_d_distal
   ```