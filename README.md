# AdaptiveAlpha / Flowalk 
Flowalk Random Walker where random walks of nodes are biased towards high in-degree nodes in local &amp; non-local neighborhoods.
 - Some code snippets have been borrowed from - https://github.com/gesiscss/Homophilic_Directed_ScaleFree_Networks

## Algorithms 
![image](https://github.com/user-attachments/assets/b04ab08d-d678-470f-ba2f-852035ec8f1d)


![image](https://github.com/user-attachments/assets/dac146a3-7b7e-4229-89c5-85c2109776c6)


## Dataset Information
(1) Synthetic Datasets - Dataset from the paper - [Inequality and inequity in network-based ranking and recommendation algorithms](https://www.nature.com/articles/s41598-022-05434-1). Available in the repository. 


![image](https://github.com/user-attachments/assets/63c9c0ed-981c-4225-bdde-c48ceb7cfd62)




(2) Empirical Datasets - 
- Rice. Dataset from the paper - [Crosswalk](https://github.com/ahmadkhajehnejad/CrossWalk/tree/master/data/rice).
- Pokec and Tuenti - Datasets available from the authors of the paper - [The Effect of Homophily on Disparate Visibility of Minorities in People Recommender Systems](https://ojs.aaai.org/index.php/ICWSM/article/view/7288)

  ![image](https://github.com/user-attachments/assets/9e295187-8a01-4993-8f73-0ff717044fd4)

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

### (1.1) Generate Recommendations with Utility & Betweenness Centrality Disparity for Real Datasets

Usage - 

```python generate_recos_real_ds_model_based.py --model model_arg --name name_arg```

| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | ffw (Fairwalk), fcw (Crosswalk), fastadaptivealphatestfixed (Flowalk with varying alpha), fastadaptivealphatest (Flowalk with fixed alpha), fpr (Fairness Aware PageRank) | Link Recommenders to generate Recommendations  |
| name | rice, pokec, tuenti | Dataset Name |
| alpha | {0.3, 0.5, 0.7}  | Used with fastadaptivealphatest (Flowalk with fixed alpha) |
| psi | {0.2 (Rice), 0.5 (Pokec), 0.4 (Tuenti) }  | Used with fpr (Fairness Aware PageRank)  |
|seed | {42, 420, 4200}  | Setting fixed run for every run|

### (1.2) Generate Random Recommendations with Utility & Betweenness Centrality Disparity for Real Datasets 

Usage - 

```python generate_recos_at_random_ds.py  --name name_arg```

| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| name | rice, pokec, tuenti | Dataset Name |
|seed | {42, 420, 4200}  | Setting fixed run for every run|


###  (2) Generate Recommendations for Betweenness Centrality Disparity for Synthetic Datasets

```python generate_recos_walker.py --model model_arg --hmm hmm_arg --hMM hMM_arg```
| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | ffw (Fairwalk), fcw (Crosswalk), fastadaptivealphatestfixed (Flowalk with varying alpha), fastadaptivealphatest (Flowalk with fixed alpha), fpr (Fairness Aware PageRank) | Link Recommenders to generate Recommendations  |
| hMM | {0.0,..0.9} |In-class Majority Class Homophily |
| hmm | {0.0,..0.9} | In-class Minority Class Homophily |
| fm | {0.1,0.2,0.3,0.4} | Minority Size Fraction |
| alpha | {0.3, 0.5, 0.7}  | Used with fastadaptivealphatest (Flowalk with fixed alpha) |
| psi | {0.3}  | Used with fpr (Fairness Aware PageRank)  |

(We provide ```start``` and ```end``` parameter to run a range of homophilic values parallely - see main caller of the file ```generate_recos_walker.py```)
##  Visualization Plots:
### (1) Generate Heatmap of Betweenness Centrality Disparity
```python generate_heatmap_centrality.py --model model_arg --reco reco_arg --group group_arg  --diff```

An example  - 

``` python generate_heatmap_centrality.py --model ffw_p_1.0_q_1.0_fm_0.3 --reco after --group 0 --diff ```
| argument | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | ffw_p_x_q_x_fm_x(Fairwalk), fcw_alpha_x_p_x_fm_x (Crosswalk), fastadaptivealphatest_beta_2.0_fm_0.3 (Flowalk with varying alpha), fastadaptivealphatestfixed_alpha_xx_beta_2.0_fm_0.3 (Flowalk with fixed alpha), fpr_psi_0.3_fm_0.3 (Fair PageRank) | Add the appropriate parameters of the model you used while generating recommendations in the x space |
| reco | before/after |before generates recommendations of DPAH (baseline) model |
| group | 0/1 | 0 - Majority , 1 - Minority |

### (2) Generate  Betweenness Centrality Disparity and Utility for Real Datasets - 

 Betweenness Centrality Disparity:  In ```visualize_plots.py``` function - ```plot_fair_metrics_v2()```
 
 Utility:  In ```visualize_plots.py``` function - ```plot_utility_metrics()```

Betweenness Centrality for Pokec and Tuenti Datasets is too expensive and time-intensive to calculate by just Networkx Library - used  [GPU accelerated NetworkX backend](https://rapids.ai/nx-cugraph/)

Notebook is [https://github.com/madhu221b/AdaptiveAlpha/blob/main/Copy_of_accelerated_networkx_demo.ipynb](here).

### (3) Generate Indegree Centrality Disparity, Statistical Parity and Equality of Representation for Real Datasets - 

 Indegree Centrality Disparity:  In ```visualize_plots.py``` function - ```get_models_vs_indegree()```
 
 Statistical Parity:  In ```visualize_plots.py``` function - ```get_statistical_imparity_all()```

 Equality of Representation:  In ```visualize_plots.py``` function - ```get_er_network_all()```

