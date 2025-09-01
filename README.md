# protein-backbone-diffusion
PyTorch implementation of **score-based diffusion models** for **protein backbone generation** using **graph neural networks**.

This repository contains the code and documentation developed as part of my MSc dissertation at the University of Oxford.

## Overview
- Implements **score-based diffusion models** for generative modelling.  
- Represents proteins as **C-alpha backbone graphs** with both spatial and sequential edges.  
- Provides several score network architectures:
  - **UNet baseline** on padded sequences  
  - **`TransformerConv`-based GNN** (via PyTorch Geometric) with and without sinusoidal positional embeddings  
- Benchmarked on the **CATH S40** dataset  
- Evaluated with **RMSD** and **TM-score**, with qualitative PyMOL visualisations

## Documentation
For full details of the methodology, experiments, and references, please see the [dissertation](./docs/dissertation.pdf) and the accompanying [bibliography](./docs/references.bib).
