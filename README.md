# Bespoke Nanoparticle Synthesis and Chemical Knowledge Discovery Via Autonomous Experimentations

## Introduction

This repository recommend optimized synthesis recipe via AI models. All of experiment equipments are controlled by Master Node, and are modularized according to the purpose of platform, such as [BatchSynthesisModule](https://github.com/KIST-CSRC/BatchSynthesisModule), [UV-VisModule](https://github.com/KIST-CSRC/UV-VisModule), and others (might be added more platform later). Master Node contains source code of AI models, Logger, etc.

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
Master Node
├── AI
│   └── Bayesian: Bayesian optimization
│   └── Fitness: Fitness function
│   └── SaveModel: pickle of model
└── Log
│   └── Logging_Class.py: write log for all actions in autonomous laboratory
├── Result: store data for each project
│   └── 1_Chemistry_discovery
│       └── AI_decision_process
│       └── DB
│       └── Optimization_result
```

## Reference
