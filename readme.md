# DiffCkt Open Source Circuit Dataset README

## 1. Dataset Overview

This Dataset is the official dataset accompanying the paper *"DiffCkt: A Diffusion Model-Based Hybrid Neural Network Framework for Automatic Transistor-Level Generation of Analog Circuits"*. Designed to advance automated analog circuit design, this dataset provides high-quality training and research resources for the field. Constructed based on the TSMC 65nm CMOS process, it contains **over 400,000 pairs of amplifier structures and performance metrics** .

### Key Features:

- **Process Technology**: TSMC 65nm CMOS, a widely used PDK for analog integrated circuit design.
- **Data Composition**: Includes single-stage and multistage amplifiers with diverse topologies, ensuring broad coverage of circuit structures and parameters.
- **Performance Metrics**: 13 critical metrics are recorded for each circuit, including power consumption, DC gain, gain-bandwidth product (GBW), phase margin (PM), and more .
- **Graph Representation**: Circuits are encoded as graphs with node attributes (device types and parameters) and edge attributes (port connections), enabling compatibility with graph-based machine learning models .1

### Dataset Structure:

- **Single-Stage Amplifiers**: ~60,000 samples.
- **Multistage Amplifiers**: ~90,000 samples per topology, covering 28 distinct amplifier structures derived from 5 multistage and 8 single-stage topologies .
- **File Format**: Graph data is stored by .pt format.
