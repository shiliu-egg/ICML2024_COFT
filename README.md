# Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models

This is the official codebase of the paper Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models in ICML2024.

## Setup

You can find the dependencies in `requirements.txt`. A script for installation is shown as follows:

```shell
conda create -n COFT python=3.10.13
conda activate COFT

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118
```

## File Tree
- recaller.py  `The recaller part of COFT`
- scorer.py `The scorer part of COFT`
- threshold.py `To calculate the dynamic threshold of highlightings`
- evaluation.py `To evaluate the performance of COFT`

- README.md
- requirements.txt
- run.sh   `To run COFT`
- all.jsonl `The FELM dataset`


## Reproduction

To run COFT,we take the Sci/Tech domain as an example, you can use the following command. For other domains, you can simply modify the file path to obtain the results.
```shell
bash run.sh
```

To reproduce the results reported in the paper, just adjust the hyperparameters to the corresponding hyperparameters. You can feel free to test other set of hyperparameters.

## Citation

If you find our work useful your research, please cite our paper:

```
@inproceedings{lvcoarse,
  title={Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models},
  author={Lv, Qitan and Wang, Jie and Chen, Hanzhu and Li, Bin and Zhang, Yongdong and Wu, Feng},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
