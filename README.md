# Greedy InfoMax

We can train a neural network **without end-to-end backpropagation** and achieve competitive performance.

This repo provides the code for the experiments in our paper:

Sindy Löwe*, Peter O'Connor, Bastiaan S. Veeling* - [Putting An End to End-to-End: Gradient-Isolated Learning of Representations](https://arxiv.org/abs/1905.11786)

&ast;equal contribution


## What is Greedy InfoMax?

We simply divide existing architectures into gradient-isolated modules and optimize the mutual information between cross-patch intermediate representations.


![The Greedy InfoMax Learning Approach](media/architecture.png)


What we found exciting is that despite each module being trained greedily, it improves upon the representation of the previous module. This enables you to keep stacking modules until downstream performance saturates.

<p align="center"> 
    <img src="./media/LatentClassification.png" width="700">
</p>


## How to run the code

### Dependencies

- [Python and Conda](https://www.anaconda.com/)
- Setup the conda environment `infomax` by running:

    ```bash
    bash setup_dependencies.sh
    ```

Additionally, for the audio experiments:
- Install [torchaudio](https://github.com/pytorch/audio) in the `infomax` environment
- Download audio datasets 
    ```bash 
    bash download_audio_data.sh
    ```

### Usage

#### Vision Experiments
- To replicate the vision results from our paper, run

    ``` bash
    source activate infomax
    bash vision_traineval.sh
    ```
    This will train the Greedy InfoMax model as well as evaluate it by training a linear image classifiers on top of it
    
    

- View all possible command-line options by running

    ``` bash
    python -m GreedyInfoMax.vision.main_vision --help
    ```    
    
    Some of the more important options are:
    
    * in order to train the baseline CPC model with end-to-end backpropagation instead of the Greedy InfoMax model set: 
    ```bash
    --model_splits 1
    ```

    * If you want to save GPU memory, you can train layers sequentially, one at a time, by setting the module to be trained (0-2), e.g.
    
    ```bash 
    --train_module 0
    ```
    

- Download a GIM model pretrained on STL-10 [here](https://drive.google.com/file/d/1yxwVOpxlrdAFHrNtMYy4QkszsUQFrI6X/view?usp=sharing)


#### Audio Experiments
- To replicate the audio results from our paper, run

    ``` bash
    source activate infomax
    bash audio_traineval.sh
    ```
    This will train the Greedy InfoMax model as well as evaluate it by training two linear classifiers on top of it - one for speaker and one for phone classification.
    
    

- View all possible command-line options by running

    ``` bash
    python -m GreedyInfoMax.audio.main_audio --help
    ```    
    
    Some of the more important options are:
    
    * in order to train the baseline CPC model with end-to-end backpropagation instead of the Greedy InfoMax model set: 
    ```bash
    --model_splits 1
    ```

    * If you want to save GPU memory, you can train layers sequentially, one at a time, by setting the layer to be trained (0-5), e.g.
    
    ```bash 
    --train_layer 0
    ```
    
## Want to learn more about Greedy InfoMax?
Check out my [blog post](https://loewex.github.io/GreedyInfoMax.html) for an intuitive explanation of Greedy InfoMax. 

Additionally, you can watch my [presentation at NeurIPS 2019](https://slideslive.com/38923276). My slides for this talk are available [here](media/Presentation_GreedyInfoMax_NeurIPS.pdf).


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{lowe2019putting,
  title={Putting an End to End-to-End: Gradient-Isolated Learning of Representations},
  author={L{\"o}we, Sindy and O'Connor, Peter and Veeling, Bastiaan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3039--3051},
  year={2019}
}
```


## References 
- [Representation Learning with Contrastive Predictive Coding - Oord et al.](https://arxiv.org/abs/1807.03748)
