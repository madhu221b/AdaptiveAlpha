# AdaptiveAlpha / Flowalk 
Flowalk Random Walker where random walks of nodes are biased towards high in-degree nodes in local &amp; non-local neighborhoods.
 - Some code snippets have been borrowed from - https://github.com/gesiscss/Homophilic_Directed_ScaleFree_Networks

## Installation: 
```
wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh
vim -i  ~/.bashrc
```
Add the following line to the end of the file - ```export PATH="/home/<<username>>/anaconda3/bin:$PATH"```

```
conda create -n lr_env python=3.8
source activate lr_env
pip install networkx
pip install tqdm
pip install numpy
pip install gensim
pip install joblib
pip install scikit-learn
pip install node2vec
pip install pandas
pip install matplotlib
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

```


## Generate Recommendations:

(1) generate_recos_real_ds_model_based.py : Generate Recommendations with Utility & Fairness Scores for Real Datasets

Usage - 

```python generate_recos_real_ds_model_based.py --model <<>> --name <<>>```

| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | n2v (Node2Vec) , fw (Fairwalk), cw (Crosswalk), adaptivealpha (Adaptive Alpha), nlindlocalind (NonLocal Indegree Walker) | Link Recommenders to generate Recommendations  |
| name | rice, pokec, tuenti | Dataset Name |
| alpha | 1.0 | Optional used with model - nlindlocalind. As part of sanity check - alpha_g = 1 |


(2) generate_recos_walker.py - Generate Recommendations with Fairness Scores for Synthetic Datasets

```python generate_recos_walker.py --model << >> --hmm << >> --hMM << >>```
| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | n2v (Node2Vec) , fw (Fairwalk), cw (Crosswalk), adaptivealpha (Adaptive Alpha) | Link Recommenders to generate Recommendations  |
| hMM | {0.0,..0.9} |In-class Majority Class Homophily |
| hmm | {0.0,..0.9} | In-class Minority Class Homophily        |
| fm | {0.1,0.2,0.3,0.4} | Minority Size Fraction      |

##  Visualization Plots:
(1) Generate Heatmap
```python generate_heatmap_centrality.py --model <<>> --reco after --group 0 --centrality betweenness ```

An example  - 

``` python generate_heatmap_centrality.py --model fw_p_1.0_q_1.0_fm_0.3 --reco after --group 0 --centrality betweenness --diff ```
| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | n2v_p_x_q_x_fm_x (Node2Vec) , fw_p_x_q_x_fm_x(Fairwalk), cw_alpha_x_p_x_fm_x (Crosswalk), adaptivealpha_beta_x_fm_x (Adaptive Alpha) | Add the appropriate parameters of the model you used while generating recommendations in the x space |
| reco | before/after |before generates recommendations of DPAH (baseline) model |
| diff | true/false | Visibility - True , Fair Betweenness Centrality (Fairness) - False  |
| group | 0/1 | 0 - Majority , 1 - Minority |


