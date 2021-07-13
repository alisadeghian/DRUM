[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drum-end-to-end-differentiable-rule-mining-on/link-prediction-on-wn18rr)](https://paperswithcode.com/sota/link-prediction-on-wn18rr?p=drum-end-to-end-differentiable-rule-mining-on) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drum-end-to-end-differentiable-rule-mining-on/link-prediction-on-fb15k-237)](https://paperswithcode.com/sota/link-prediction-on-fb15k-237?p=drum-end-to-end-differentiable-rule-mining-on)

# DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs 

This is the implementation of DRUM, proposed in the following paper:

[DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs](https://papers.nips.cc/paper/9669-drum-end-to-end-differentiable-rule-mining-on-knowledge-graphs.pdf) -- [\[arXive + appendix\]](https://arxiv.org/abs/1911.00055).
Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, Daisy Zhe Wang.
NeurIPS 2019.

## Requirements
- Python 2.7
- Numpy 
- Tensorflow 1.13.1

## Quick start
The following command starts training a dataset about family relations, and stores the experiment results in the folder `exps/demo/`.

```
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo
```

To get the best performance, use different ranks for different datasets, default value is set to 3.

## Evaluation
To evaluate the prediction results, follow the steps below. The first two steps is preparation so that we can compute _filtered_ ranks (see [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) for details).

We use the experiment from Quick Start as an example. Change the folder names (datasets/family, exps/dev) for other experiments.
```
. eval/collect_all_facts.sh datasets/family
python eval/get_truths.py datasets/family
python eval/evaluate.py --preds=exps/demo/test_predictions.txt --truths=datasets/family/truths.pckl
```

This code partially is borrowed from [Neural LP](https://github.com/fanyangxyz/Neural-LP).
