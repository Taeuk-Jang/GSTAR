# Group-Aware Threshold Adaptation for Fair Classification(GSTAR)
This repo contains the code of Group-Aware Threshold Adaptation for Fair Classification (GSTAR) in AAAI, 2022.

We include an example of CelebA data to demonstrate the work in the paper.
CelebA dataset can be downloaded from the link : https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8

Download pretrained Baseline ResNet model and put into experiments/celeba dataset
Pretrained model link : https://1drv.ms/u/s!AlCY8gQvkMJOhMUnfVK5VdAMEA8R-w?e=ixSJ43

## Description of the repo
- Source files are located in GSTAR folder.

- dataset is saved in the dataset folder. CelebA dataset is not included in there, 
    because of the size limitation for the submission. 
    Path and Sensitive attribute, label information of CelebA for train/valid/test dataset are saved in 
    train_data.txt, valid_data.txt, and test_data.txt respectively.
    
- Required packages and dependencies can be found in requirements.txt

- Demonstration is done in notebook files in notebooks folder.
    1) ./notebooks/GSTAR-celebA.ipynb : evaluate GSTAR post-processing on CelebA dataset.
    2) ./notebooks/GSTAR-pareto.ipynb : evaluate GSTAR pareto frontiers and illustrate the feasible regions comparing 
                            with FACT on CelebA dataset.
    3) ./notebooks/figures : includes all the experimental results on evaluated datasets.

## Citation
```
@inproceedings{jang2022group,
  title={Group-aware threshold adaptation for fair classification},
  author={Jang, Taeuk and Shi, Pengyi and Wang, Xiaoqian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={6},
  pages={6988--6995},
  year={2022}
}
```

