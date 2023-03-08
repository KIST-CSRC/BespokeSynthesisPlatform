# MasterPlatform

## Introduction

This repository contains source code of Bayesian Optimization, Logger, Fitness function, etc. 

## Bayesian optimization with Earlystopping
<p align="center">
  <img src="img\earlystopping_effect.png" width="50%" height="50%" />
</p>


## Target result

We target 3 nanoparticles in [this literature](https://www.sciencedirect.com/science/article/pii/S2468217921000125). This target nanopalte is known to improve the performance of organic solar cell. Therefore, we tried to challenge bespoke nanoparticle synthesis. Here is our the results.

## Fitness function

$$ -0.9 * \min (λ_{max} – λ_{max, target}) -0.07*\min(1-intensity) -0.03 * \min FWHM $$


### 1. 513nm

**UV spectrum image**
<p align="center">
  <img src="img\513nm.png" width="30%" height="30%" />
</p>

**AI-decision proccess**
<p align="center">
  <img src="img\513nm.gif" width="70%" height="70%" />
</p>

### 2. 573nm

**UV spectrum image**
<p align="center">
  <img src="img\573nm.png" width="30%" height="30%" />
</p>

**AI-decision proccess**
<p align="center">
  <img src="img\573nm.gif" width="70%" height="70%" />
</p>

### 3. 667nm

**UV spectrum image**
<p align="center">
  <img src="img\667nm.png" width="30%" height="30%" />
</p>

**AI-decision proccess**
<p align="center">
  <img src="img\667nm.gif" width="70%" height="70%" />
</p>


## Installation

**Using conda**
```bash
conda env create -f requirements_conda.txt
```
**Using pip**
```bash
pip install -r requirements_pip.txt
```

## Script architecture
```
MasterPlatform
├── AI
│   └── Bayesian
│       └── bayesian_optimization.py
│       └── event.py
│       └── logger.py
│       └── observer.py
│       └── parallel_opt.py
│       └── target_space.py
│       └── util.py
│   └── Fitness
│       └── UV_fitness.py
│   └── SaveModel
├── DB
├── img
└── Log
│   └── Logging_Class.py
```


## Reference
1. Phengdaam, Apichat, et al. "Improvement of organic solar cell performance by multiple plasmonic excitations using mixed-silver nanoprisms." Journal of Science: Advanced Materials and Devices 6.2 (2021): 264-270.
