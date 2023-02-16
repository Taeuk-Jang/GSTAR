This includes demonstration files for Group-Aware Threshold Adaptation for Fair Classification(GSTAR) for the submission in AAAI, 2022.


We include an example of CelebA data to demonstrate the work in the paper.
CelebA dataset can be downloaded from the link : https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8

Download pretrained Baseline ResNet model and put into experiments/celeba dataset
Pretrained model link : https://1drv.ms/u/s!AlCY8gQvkMJOhMUnfVK5VdAMEA8R-w?e=ixSJ43

1. Source files are located in GSTAR folder.

2. dataset is saved in the dataset folder. CelebA dataset is not included in there, 
    because of the size limitation for the submission. 
    Path and Sensitive attribute, label information of CelebA for train/valid/test dataset are saved in 
    train_data.txt, valid_data.txt, and test_data.txt respectively.
    
3. Required packages and dependencies can be found in requirements.txt

4. Demonstration is done in notebook files in notebooks folder.
    1) ./notebooks/GSTAR-celebA.ipynb : evaluate GSTAR post-processing on CelebA dataset.
    2) ./notebooks/GSTAR-pareto.ipynb : evaluate GSTAR pareto frontiers and illustrate the feasible regions comparing 
                            with FACT on CelebA dataset.
    3) ./notebooks/figures : includes all the experimental results on evaluated datasets.


