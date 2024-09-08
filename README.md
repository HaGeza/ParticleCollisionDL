# 2AMM40_ParticleTracking
Advanced Topics in AI (2AMM40) @ TU/e - Particle Tracking Project

## Particle Tracking @ CERN
This is the repository for the Particle Tracking assignment for the ATAI (2AMM40) course. 

The data used in this project is from a Kaggle competition [TrackML - Particle Tracking](https://www.kaggle.com/c/trackml-particle-identification/overview). They provide the following description:

"
To explore what our universe is made of, scientists at CERN are colliding protons, essentially recreating mini big bangs, and meticulously observing these collisions with intricate silicon detectors.

While orchestrating the collisions and observations is already a massive scientific accomplishment, analyzing the enormous amounts of data produced from the experiments is becoming an overwhelming challenge.

Event rates have already reached hundreds of millions of collisions per second, meaning physicists must sift through tens of petabytes of data per year. And, as the resolution of detectors improve, ever better software is needed for real-time pre-processing and filtering of the most promising events, producing even more data.

To help address this problem, a team of Machine Learning experts and physics scientists working at CERN (the world largest high energy physics laboratory), has partnered with Kaggle and prestigious sponsors to answer the question: can machine learning assist high energy physics in discovering and characterizing new particles?
"

The goal of your research is slightly different. Instead of labeling the tracks, we are interested in generating probable hits/tracks given the initial conditions of a simulation. For more information, read through the assignment description on Teams/canvas.

### Getting started

Now you're ready to check out the `notebooks/01_getting_started.ipynb` notebook! It features a short introduction to the data, as well as a guide on installing the data and libraries. 

## File structure
This section explains the file structure provided in this repository. Feel free to change it to meet your requirements. 

    data/               -   Directory for your data files.
        - NOTE: Not tracked in git history by default.

    figures/            -   Directory for storing any figures/images.
        - NOTE: Not tracked in git history by default.

    models/             -   Directory for storing model files (.pt/.pth etc.)
        - NOTE: Not tracked in git history by default.

    notebooks/          -   Directory for your Jupyter notebooks. 
        - NOTE: avoid using notebooks for source code. We recommend writing all code for models, data manipulation, and plotting in the src/ folder
        - Use the notebooks for creating nice visualization dashboards or reports.

    results/            -   Directory for storing tables, or other text-based files.
        - NOTE: Not tracked in git history by default.

    src/                -   Directory that should house (most) of your code.

    main.py             -   Main access point of your system. Should be accessible via command line. A trivial example is provided