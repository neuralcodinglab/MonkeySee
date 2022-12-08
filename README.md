# MonkeySee
MonkeySee: Reconstructing natural images from multiunit activity


## Introduction

In this repository you will find the source code for the Spatial-based and Temporal-based experiments in the "MonkeySee: Reconstructing natural images from multiunit activity" paper. Both folders contain the data preprocessing codes ([`data_prep_spatial.py`](SpatialBased/data_prep_spatial.py) and [`data_prep_spatial.py`](TemporalBased/data_prep_temporal.py)) and the training codes ([`train_spatial.py`](SpatialBased/train_spatial.py) and [`train_temporal.py`](TemporalBased/train_temporal.py)). 


## Spatial-based Training 
For the spatial-based model, we splitted the channels into 10 electrodes. The locations of these excitations are shown in the RFstatic locations figure.


![RFstatic locations](Figures/RFStatic.png "RFstatic locations")


## Temporal-based Training 

Here we split the data into 5 time chunks.

<img src="Figures/time_windows.png" alt="time windows" width="500"/> 

The reconstructions (R) with their corresponding targets (T) of the entire test dataset are shown in the following figure:

<img src="Figures/recons_times.png" alt="reconstructions temporal based" width="700"/> 