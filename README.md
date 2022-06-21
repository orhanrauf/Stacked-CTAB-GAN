# Stacked CTAB-GAN
This is the official git for paper Extending CTAB-GAN with StackGAN. The paper is published for Orhan Rauf Akdemir dissertation at the Computer Science and Engineering Bachelor programme at TU Delft. If you have any problems with running it, please contact o.r.akdemir@student.tudelft.com.


## Prerequisite

The required package version
```
numpy==1.21.0
torch==1.9.1
pandas==1.2.4
sklearn==0.24.1
dython==0.6.4.post1
scipy==1.4.1
```

## Example
`Stacked_Experiment_Script_Adult1.ipynb`, `Stacked_Experiment_Script_Adult2.ipynb`, `Stacked_Experiment_Script_Adult3.ipynb` are all example notebooks for training different types of Stacked CTAB-GAN with Adult dataset. The dataset is already under `Real_Datasets` folder. The evaluation code is also provided.

## For large dataset

If your dataset has large number of column, you may encounter the problem that our currnet code cannot encode all of your data since CTAB-GAN will wrap the encoded data into an image-like format. What you can do is changing the line 341 and 348 in `model/synthesizer/ctabgan_synthesizer.py`. The number in the `slide` list
```
sides = [4, 8, 16, 24, 32]
```
is the side size of image. You can enlarge the list to [4, 8, 16, 24, 32, 64] or [4, 8, 16, 24, 32, 64, 128] for accepting larger dataset.


```
