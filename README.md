## Disclaimer for GitHub version

*Note*: [GitLab hosted by TU/e](https://gitlab.tue.nl/users/sign_in) was used to develop this project between September-November 2024, as a part of the course "Advanced Topics in AI (2AMM40)", by group 4, consisting of Balázs Pelok (me), Mathieu-van-Luijken and Tom Harmsen. The link to the final report found below does not work on GitHub, instead the reports can be found in this [Release](https://github.com/HaGeza/ParticleCollisionDL/releases/tag/v1).

# 2AMM40 Particle Tracking - Group 4

This repository contains code for the [Final Report](uploads/728fc5b720b27c8b239dbcfa77dc5722/FinalReport.pdf)
of Group 4 for the course 2AMM40.

## Running the Project

This section contains instructions for running the code, and information about how the results presented
in the Final Report were generated.

### Installing Dependencies

The repo was tested with the following python version(s): Python 3.11.10., 3.10.4

To install dependencies, run `setup.py`, which does the following:
1. Installs `requirements.txt`
2. Installs `trackml-library`:
    - Clone the repository next to the project directory: `git clone https://github.com/LAL/trackml-library`
    - Install it with: `pip install ../trackml-library`

### Data Precomputation

The small dataset used for architecture exploration is available [here](https://www.kaggle.com/c/trackml-particle-identification/data?select=train_sample.zip).
The large dataset used for the final training is the combination of the first two parts of the large dataset, i.e.
[train-1](https://www.kaggle.com/c/trackml-particle-identification/data?select=train_1.zip) and [train-2](https://www.kaggle.com/c/trackml-particle-identification/data?select=train_2.zip).

Whilst the models can be run with the original datasets, this is slow and thus not recommended. Instead, we suggest
to run model trainings with precomputed datasets as described in the report. The zipped version of the precomputed
small dataset is available in the repository in `data/precomputed.zip`. To run trainings with it, unzip it in
the data directory first.

Generating the precomputed large dataset takes substantial time and storage. Precomputation for a downloaded dataset
is done using `scripts/create_precomputed_dataset.py`, and precomputing large datasets is done with `scripts/preprocess_large_dataset.py`.
Note that the later script creates separate zip files for each part of the large dataset, which then have to be combined
manually, paying special attention to the `gt_sizes.json` file within each part. The contents of these json files have
to be combined into a new valid json, and placed into the final large dataset (as `gt_sizes.json`).

### Training Models

Use the `main.py` to train models. Run `python main.py --help` to see the parameters available, together with
their default values and descriptions. Models and logs are saved within the `runs` directory, in an automatically
generated subdirectory. Within each run subdirectory, `info.json` describes the model architecture and training
setup. To continue training a model from a checkpoint, run `python main.py -c <subdir_id>`, or `python main.py -c latest`
to continue training the newest run.

For each run, the following files are saved:
- `info.json` - described above
- `min_loss.pth` - checkpoint for the model with the minimum loss so far
- `latest.pth` - checkpoint for the model of the most recent epoch
- `train.log` - contains size prediction and loss information for each batch during training
- `eval.log` - contains loss and metric information for each epoch

### Reproducing Results

The models described in the paper were run using the following parameters:

| Name | Parameters | 
| ---- | ---------- |
| Default | `-b 8` |
| Top-5 Pooling | `--pooling_levels 5 -b 8` |
| Variational Encoding | `--variational --encoder_loss_w 0.00005 -b 8` |
| Many Steps | `--ddpm_num_steps 250 --ddpm_processor_layers 1 -b 8` |
| Reverse Posterior | `--ddpm_use_reverse_posterior -b 8` |
| PointNet | `--encoder point_net --pooling_levels 5 --encoder_layers 5 --ddpm_processor point_net --ddpm_num_steps 150 -b 8` |


The visuals and (contents of) tables used for model comparison were produced using `scripts/train_log_visualizations.py`.
To reproduce them, unzip the contents of `runs/runs.zip` and run the following:
```sh
python scripts/train_log_visualizations.py -i 1zxoi65f 38hyf9sx ak1vrjnv b9m80o2r r9oudocu rek8pk3y --make_table
python scripts/train_log_visualizations.py -i f1cbfno6
python scripts/train_log_visualizations.py -i 38hyf9sx --use_log
```

The visuals used for analyzing the best model (38hyf9sx) and for hit-cloud visualization were generated with
`notebooks/plot_hits.ipynb`. Note that this takes a random element from the small dataset, and calculates all
necessary pairings and predictions for the visualizations for it. The data used in the paper is available in
`notebooks/plot_hits_out/data.zip`. Unzip the file and run the notebook to reproduce the visuals.

## File structure

The repository is structured as follows:

```
.
├── README.md                           │  <- This document
├── main.py                             │  <- Used for training models
├── notebooks                           │  <- Jupyter notebooks
│   ├── ...                             │  
│   └── plot_hits.ipynb                 │  <- Hit-cloud visualization
├── requirements.txt                    │  <- Requirements
├── setup.py                            │  <- Project setup
├── scripts                             │  <- Contains various scripts
│   ├── ...                             │  
│   ├── create_precomputed_dataset.py   │  <- Precompute one dataset
│   ├── preprocess_large_dataset.py     │  <- Precompute large dataset in parts
│   └── train_log_visualizations.py     │  <- Visualizations for model comparison
└── src                                 │  <- Implementation of pipeline
```

Focusing on the `src` directory, we have the following structure:

```
└── src                                     
    ├── Data                                    │ <- Classes for data loading
    │   ├── CollisionEventLoader.py             │ <- Load non-precomputed data
    │   ├── IDataLoader.py                      │ 
    │   └── PrecomputedDataLoader.py            │ <- Load precomputed data (used in the report)
    ├── Modules                                 │ <- Parts of the pipeline
    │   │                                       │ 
    │   ├── HitSetGenerativeModel.py            │ <- Main class for the model
    │   ├── HitSetEncoder                       │ <- Encoders
    │   │   ├── GlobalPoolingEncoder.py         │ 
    │   │   ├── HitSetEncoderEnum.py            │ 
    │   │   └── IHitSetEncoder.py               │ 
    │   ├── HitSetGenerator                     │ <- Set generators
    │   │   ├── AdjustingSetGenerator.py        │ <- Places points with a strategy (no learning)
    │   │   ├── DDPM                            │ <- DDPM specific
    │   │   │   ├── BetaSchedules.py            │ <- Beta Schedules for DDPM
    │   │   │   └── DDPMSetGenerator.py         │ <- DDPM set generator
    │   │   ├── HitSetGeneratorEnum.py          │ 
    │   │   └── IHitSetGenerator.py             │
    │   ├── HitSetProcessor                     │ <- Processors
    │   │   ├── HitSetProcessorEnum.py          │
    │   │   ├── IHitSetProcessor.py             │
    │   │   ├── LocalGNNProcessor.py            │ <- GNN
    │   │   └── PointNetProcessor.py            │ <- PointNet
    │   └── HitSetSizeGenerator                 │ <- Size generators
    │       ├── GaussianSizeGenerator.py        │ <- Gaussian size generator (used in the report)
    │       ├── HitSetSizeGeneratorEnum.py      │
    │       └── IHitSetSizeGenerator.py         │
    ├── Pairing                                 │ <- Strategies for pairing hits
    │   ├── GreedyStrategy.py                   │
    │   ├── HungarianAlgorithmStrategy.py       │ <- Hungarian pairing (used in the report)
    │   ├── IPairingStrategy.py                 │
    │   ├── PairingStrategyEnum.py              │
    │   ├── RepeatedKDTreeStrategy.py           │
    │   └── VectorizedGreedyStrategy.py         │
    ├── TimeStep                                │ <- Time steps
    │   ├── ForAdjusting                        │ <- Time steps with support for adjusting set generation
    │   │   ├── ITimeStepForAdjusting.py        │
    │   │   ├── PlacementStrategy               │ <- Strategies for placing hits
    │   │   │   ├── EquidistantStrategy.py      │ <- Equidistant placement (used in the report)
    │   │   │   ├── IPlacementStrategy.py       │
    │   │   │   ├── PlacementStrategyEnum.py    │
    │   │   │   └── SinusoidStrategy.py         │ <- Sinusoidal placement (used in the interim report)
    │   │   ├── PrecomputedTimeStep.py          │ <- Wrapper for time steps when precomputed data is used (used in the report)
    │   │   └── VolumeLayer                     │ <- Volume-layer based time-step definition specific
    │   │       ├── VLMaps.py                   │
    │   │       ├── VLRings.py                  │
    │   │       └── VLTimeStep.py               │ <- Volume-layer time-step (used in the report, wrapped)
    │   ├── ITimeStep.py                        │
    │   └── TimeStepEnum.py                     │
    ├── Trainer                                 │
    │   ├── HSGMTrainer.py                      │ <- Class for training models
    │   └── TrainingRunIO.py                    │ <- Class for logging trainings and saving/loading models
    └── Util                                    │ <- Utility classes
        ├── CoordinateSystemEnum.py             │
        ├── CoordinateSystemFuncs.py            │ <- Coordinates system conversions, and other functions
        ├── Distributions.py                    │ <- Distribution functions, mainly used in DDPM
        └── Paths.py                            │ <- Variables related to project structure
```
