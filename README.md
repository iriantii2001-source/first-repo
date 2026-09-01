# Prototypical Networks for Few-shot Learning

Code for the NIPS 2017 paper [Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf).

If you use this code, please cite our paper:

```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
 ```

## Training a prototypical network

### Install dependencies

* Create and activate a virtualenv: `python3 -m venv venv && source venv/bin/activate`.
* Install the protonets package and its dependencies by running `pip install -e .`
  (installs `torch`, `torchvision`, `tqdm`, `pyiqa`).

### Dataset

* This project is set up to train on the `dataset ice ice/` folder, a 2-class seaweed
  disease dataset (`sehat` = healthy, `ice ice` = ice-ice disease), each already split
  into `train/`, `valid/` and `test/` subfolders. No download step is required.
* The original Omniglot dataset is still supported; run `sh download_omniglot.sh` and
  pass `--data.dataset omniglot` to train on it instead.

### Train the model

* Baseline (standard mean-prototype ProtoNet):
  `python scripts/train/few_shot/run_train.py`.
  This runs 2-way, 5-shot training on the ice-ice dataset and places the results
  into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
  * If you are running on a GPU you can pass in the option `--data.cuda`.
* QA-ProtoNet (quality/typicality-weighted prototype, the method proposed in
  `DRAFT_PROPOSAL_SOTA_FINAL.md`):
  ```
  python scripts/train/few_shot/run_train.py \
    --model.model_name qa_protonet_conv --data.compute_quality \
    --log.exp_dir results/qa_protonet
  ```
  * `--data.compute_quality` scores every support image once with the pretrained
    ARNIQA no-reference IQA model (`pyiqa`) and caches the scores in
    `dataset ice ice/.arniqa_cache.json` (only new images get re-scored on later
    runs). The first run needs internet access to download the ARNIQA weights.
  * The prototype weight for each support image is a softmax over
    `alpha * quality_score + (1 - alpha) * typicality_score`, where typicality is
    the (negative, normalized) distance of that image's embedding to its class's
    unweighted mean embedding. Tune with `--model.alpha` (0..1, default 0.5) and
    `--model.tau` (softmax temperature, default 1.0).
  * `--data.compute_quality` only has an effect on the `iceice` dataset; the
    baseline `protonet_conv` model ignores `sample['qs']` even if it's present, so
    both models can be trained on the exact same episodes for a fair comparison.
* Re-run in trainval mode `python scripts/train/few_shot/run_trainval.py`. This will save your model into `results/trainval` by default.

### Evaluate

* Run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`.
# first-repo
