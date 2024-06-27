# AdaptiveAlpha
Adaptive Alpha Random Walker where random walks of nodes are biased towards high in-degree nodes in local &amp; non-local neighborhoods.
 - Some code snippets have been borrowed from - https://github.com/gesiscss/Homophilic_Directed_ScaleFree_Networks


-> Scripts to Generate Recommendations:

(1) generate_recos_real_ds_model_based.py : Generate Recommendations with Utility & Fairness Scores for Real Datasets

Usage - 

```python generate_recos_real_ds_model_based.py --model <<>> --name <<>>```

| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | n2v (Node2Vec) , fw (Fairwalk), adaptivealpha (Adaptive Alpha), nlindlocalind (NonLocal - Local Indegree Walker) | Link Recommenders to generate Recommendations  |
| name | rice, facebook | Dataset Name |
| alpha | 1.0 | Optional used with model - nlindlocalind. As part of sanity check - alpha_g = 1 |


(2) generate_recos_walker.py - Generate Recommendations with Fairness Scores for Synthetic Datasets

```python generate_recos_walker.py --model << >> --hmm << >> --hMM << >>```
| argument      | values|description                                                                  |
|----------------------|-------|-----------------------------------------------------------------------|
| model | n2v (Node2Vec) , fw (Fairwalk), adaptivealpha (Adaptive Alpha) | Link Recommenders to generate Recommendations  |
| hMM | {0.0,..0.9} |In-class Majority Class Homophily |
| hmm | {0.0,..0.9} | In-class Minority Class Homophily        |
| fm | {0.1,0.2,0.3,0.4} | Minority Size Fraction      |

-> Script to Generate Heatmap Plots:

-> Script for Visualization Plots:
