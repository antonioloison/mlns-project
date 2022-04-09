# mlns-project
Deep Learning based Graph embedding for nodes clustering
Project for the Machine Learning in Network Science class at CentraleSupélec.

## Structure of the repository

### Non-deep algorithms
Non-deep algorithms including:
- KMeans
- Spectral Clustering
- RMSC
are in the file `clustering.py`, with rather clean names.
Can be launched with Python CLI to get the visualization results (can be a bit long).

### VAEs
VAE and IWAE are in the file `vae_clustering.py`.
Can be launched with Python CLI to get the visualization results (can be a bit long).



### Visualization functions
In the file `visualization.py`.
 
## DAEGC 

This algorithm is seperated from the rest. You should go to the daegc directory:

```
cd daegc/
```

Then, to launch this algorithm, you should first pretrain the auto-encoder. For this you should run this command from the
`daegc` folder:

```
python pretrain.py
```

To train the other algorithm, you can launch:

```
python training.py
```
