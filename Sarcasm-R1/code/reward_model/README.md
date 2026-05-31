# Reward Model - Generative Verifier

An optional generative verifier that can assess reasoning quality as an additional reward signal during GRPO training.

## Usage

```bash
# Train the verifier (requires reasoning annotations)
python generative_verifier.py --mode train \
    --data ../../data/processed/semeval_reasoning.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output ./output/verifier

# Use verifier to score reasoning
python generative_verifier.py --mode verify \
    --data ../../data/processed/semeval_reasoning.jsonl \
    --model ./output/verifier
```

## Reference

For training the reward model, we referenced the method proposed in the following paper:

@inproceedings{DBLP:conf/iclr/ZhangHBKKA25,
  author       = {Lunjun Zhang and
                  Arian Hosseini and
                  Hritik Bansal and
                  Mehran Kazemi and
                  Aviral Kumar and
                  Rishabh Agarwal},
  title        = {Generative Verifiers: Reward Modeling as Next-Token Prediction},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  publisher    = {OpenReview.net},
  year         = {2025},
  url          = {https://openreview.net/forum?id=Ccwp4tFEtE},
  timestamp    = {Thu, 15 May 2025 17:19:06 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/ZhangHBKKA25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}