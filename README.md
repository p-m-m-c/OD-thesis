# README

This repository accompanies my master thesis titled: "Minimising Regret in Outlier Detection". It contains:

  - Code: Jupyter Notebooks converted to plain Python files
  - Data: links to data sources are found in this README file, with the code to transform it in the Code folder
  - Virtual environment blueprint for reproducibility purposes (env.txt)

## Abstract

> Ensemble methods for outlier detection have gained considerable research attention in the past 15 years. Despite that, recent literature presents limited evidence regarding the effectiveness of these methods. Using a framework originally applied to several individual outlier detection algorithms, we reliably identify the effectiveness of ensemble methods over several datasets and parameter settings. In addition, we propose a formal decision criterion for choosing an outlier detector in a completely unsupervised scenario: minimising maximal regret. The results show that the ensemble approach is very consistent in its eventual scoring, sometimes even outperforming the best individual detector but, most importantly, being the preferred choice in terms of the newly proposed regret metric.

## Data

Data was obtained from two major sources:
    - [ODDS datasets][odds] in matlab format
    - [Datasets used by Campos et al, (2016)][campos et al datasets] in arff format

## References
Campos, G. O., Zimek, A., Sander, J., Campello, R. J., Micenková, B., Schubert, E., ... & Houle, M. E. (2016). 
On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Mining and Knowledge Discovery, 30(4), 891-927.

[//]: # 


   [odds]: <http://odds.cs.stonybrook.edu/>
   [campos et al datasets]: <http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/>
