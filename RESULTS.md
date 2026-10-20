# Results

All numbers below are copied from the training logs. Nothing is estimated.

## Setup
- Dataset: VQA v2, train2014 questions and annotations
- Image features: ResNet-50 (ImageNet pretrained) with a fixed random projection to 1024-d, from Person A (82,783 images)
- Answer classes: 3000 (coverage of questions by top 3000 answers: 93.08%)
- Vocabulary size: 18,098 words
- Hardware: Colab T4 GPU
- Training settings: batch size 64, learning rate 1e-3, dropout 0.5, epochs 5, patience 3

## Run 1: small test run
- Images used (--limit_images): 2000
- Train questions: 8103 | Validation questions: 2163 | Skipped: 30721
- Time per epoch: 2-3 seconds

| Epoch     | Train loss | Val loss | Val accuracy | Yes/No | Number | Other |
| 4.9938	| 4.7167	 | 20.30%	| 25.77%	   | 64.29%	| 0.00%	 | 0.83%.| 
| 4.5832	| 4.7016	 | 20.67%	| 25.63%	   | 63.93%	| 0.22%	 | 0.76% | 
| 4.4763	| 4.6836	 | 21.31%	| 26.46%	   | 65.85%	| 0.11%	 | 0.96% | 
| 4.1575	| 3.9755	 | 21.45%	| 26.38%	   | 64.64%	| 0.22%	 | 1.78% | 
| 3.4497	| 3.6701	 | 25.06%	| 31.56%	   | 59.80%	| 28.22% | 8.66% | 

## Run 2: main run
- Images used: [ ]
- Train questions: [ ] | Validation questions: [ ] | Skipped: [ ]
- Best epoch: [ ] | Stopped by early stopping: yes / no
- Best checkpoint: [file name in Drive]

| Epoch | Train loss | Val loss | Val accuracy | Yes/No | Number | Other |
|---|---|---|---|---|---|---|

## Final result (best epoch)
- VQA accuracy: 31.56% | Exact match: 25.06% (542 / 2,163)
- Yes/No: 59.80% | Number: 28.22% | Other: 8.66%
- Accuracy rule used: [VQA accuracy min(matches/3, 1) / exact match]
- Evaluated on: held-out images from train2014 (20% by image), not the official test set

## Observations
- Run 1 is a smoke test on about 2% of the available questions, so it shows the pipeline works and is not a final result.
- In epochs 1-4 yes/no accuracy was about 64% while number and other stayed near 0%, which suggests the model relied mostly on question patterns.
- At epoch 5, number accuracy rose to 28.22% and other to 8.66%, while yes/no dipped to 59.80%. Validation loss was still falling (3.98 to 3.67), so training stopped at the 5-epoch limit with the model still improving.
- Training loss (3.45) is slightly below validation loss (3.67), a small gap that suggests mild overfitting at this stage.

## Sample wrong predictions
| Image | Question | Predicted | True | Likely cause |
|---|---|---|---|---|