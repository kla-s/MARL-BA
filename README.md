# Multi Agent Reinforcement Learning with Attention for Landmark Detection in Pre- and Post-operative MR Scans of Brain Tumors



---
## Results


Worx: combination of multi-scaling, multi agent reinforcement learning and the correct choice of the best location can drastically improve the computational performance for this problem while slightly improving the
accuracy in comparison to the baseline [[1]](https://link.springer.com/content/pdf/10.1007/978-3-030-50120-4_8.pdf). See the pdf for more theory/explanation.

The code is hacky/researchy at some spots, maybe i`ll come around to smooth it all out some day.
Till then, please just ask me in the issues if anything is unclear


---

## Usage
```
usage: DQN.py [-h] [--gpu GPU] [--load LOAD] [--task {play,eval,train}]
              [--files FILES [FILES ...]] [--saveGif] [--saveVideo]
              [--logDir LOGDIR] [--name NAME] [--agents AGENTS]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             comma separated list of GPU(s) to use.
  --load LOAD           load model
  --task {play,eval,train}
                        task to perform. Must load a pretrained model if task
                        is "play" or "eval"
  --test_all_acc        use this flag when using eval mode to generate accuracies for all saved 
                        training checkpoints, not just the last one. In the load flag specify the path to the checkpoints file not to an individual model checkpoint. Results are saved as a tensorboard log file
  --algo                either 'DQN', 'Double', 'Dueling' or 'DuelingDouble', default='DQN'
  --files FILES [FILES ...]
                        Filepath to the text file that comtains list of
                        images. Each line of this file is a full path to an
                        image scan. For (task == train or eval) there should
                        be two input files ['images', 'landmarks']
  --saveGif             save gif image of the game
  --saveVideo           save video of the game
  --logDir LOGDIR       store logs in this directory during training
  --name NAME           name of current experiment for logs
  --agents AGENTS       number of agents to be trained simulteniously 
  --coords_init         Path to the text file that contains the paths to the init coords
                        Each line is a full path to a .npy init coord file. 
                        The ordering is exactly the same as in the argument provided to --files 
                        When using tum_box directly specify path of initial coordinates dont put it in a txt file
  --reward_strategy     Which reward strategy you want? 1 is simple, 2 is line based, 3 is      
                        agent based. Default = 1
  --load_config         specify the path of a config file relative to the configDir 
                        (default:configs/)
```

### Train
Computational requirements: 10 MRI scans with 5 Landmarks ~> 4h on a Titan Xp (~10/12GB) ~> 2.79mm mean distance
```
 python DQN.py --task train  --gpu 0 --files './data/filenames/image_files.txt' './data/filenames/landmark_files.txt'
```

### Evaluate
```
python DQN.py --task eval  --gpu 0 --load data/models/DQN_multiscale_brain_mri_point_pc_ROI_45_45_45/model-600000 --files './data/filenames/image_files.txt' './data/filenames/landmark_files.txt'
```

### Test
```
python DQN.py --task play  --gpu 0 --load data/models/DQN_multiscale_brain_mri_point_pc_ROI_45_45_45/model-600000 --files './data/filenames/image_files.txt'
```
---
### Data
Unfortunately I can neither provide data nor pre-trained models as the data is planned to be used for a (scientific) competition. :/
I`ll update this when the embargo is lifted.

### References
[1] D. Waldmannstetter, F. Navarro, B. Wiestler, J. S. Kirschke, A. Sekuboyina,
E. Molero, and B. H. Menze. “Reinforced Redetection of Landmark in
Pre- and Post-operative Brain Scan Using Anatomical Guidance for Im-
age Alignment.” In: Biomedical Image Registration. Springer International
Publishing, 2020, pp. 81–90. doi: [10.1007/978-3-030-50120-4_8](https://doi.org/10.1007/978-3-030-50120-4_8)

