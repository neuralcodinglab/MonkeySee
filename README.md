# MonkeySee: Reconstructing natural images from multiunit activity

## Introduction

In this repository you will find the source code for the Spatial-based and Temporal-based experiments in the "MonkeySee: Reconstructing natural images from multiunit activity" paper. Both folders contain the data preprocessing codes ([`data_prep_spatial.py`](SpatialBased/data_prep_spatial.py) and [`data_prep_spatial.py`](TemporalBased/data_prep_temporal.py)) and the training codes ([`train_spatial.py`](SpatialBased/train_spatial.py) and [`train_temporal.py`](TemporalBased/train_temporal.py)). 


## Spatial-based Training 
For the spatial-based model, we splitted the channels into 10 electrodes. The locations of these excitations are shown in the RFstatic locations figure.


![RFstatic locations](Figures/RFStatic.png "RFstatic locations")


## Temporal-based Training 

We also train a model that is split into ROIs and time chunks of 5. This means we take a timewindow of the original time windows that we train the spatial-based model on, and split them onto 5. The time windows are 100ms originally. We train them on 25 ms for this experiment giving each channel 25ms. When doing the time experiment we then perform a forward pass with just the time that we are interested in.

<img src="Figures/time_windows.png" alt="time windows" width="500"/> 

The reconstructions (R) with their corresponding targets (T) of the entire test dataset are shown in the following figure:

<img src="Figures/recons_times.png" alt="reconstructions temporal based" width="700"/> 