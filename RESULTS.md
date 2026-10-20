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

| Epoch| Train loss | Val loss | Val accuracy | Yes/No | Number | Other |
| 1 | 4.9938 | 4.7167 | 25.77% | 64.29% | 0.00% | 0.83% |
| 2 | 4.5832 | 4.7016 | 25.63% | 63.93% | 0.22% | 0.76% |
| 3 | 4.4763 | 4.6836 | 26.46% | 65.85% | 0.11% | 0.96% |
| 4 | 4.1575 | 3.9755 | 26.38% | 64.64% | 0.22% | 1.78% |
| 5 | 3.4497 | 3.6701 | 31.56% | 59.80% | 28.22% | 8.66% |

VQA accuracy is `min(matches/3, 1)` over the 10 human answers. Exact match (prediction equals the most common human answer) at epoch 5 was 25.06% (542 / 2,163).
## Run 2: main run
- Images used: [ ]
- Train questions: [ ] | Validation questions: [ ] | Skipped: [ ]
- Best epoch: 8 | Stopped by early stopping: yes / no
- Best checkpoint: train_log_full.txt

|Epoch|	Train Loss	|Val Loss	|Val Accuracy	|Yes/No	    |Number	|Other
|1	|3.1480	        |2.6243	    |34.63%	         |67.14%	|32.07%	|25.43%
|2	|2.5141	        |2.3671	    |37.22%	         |67.08%	|33.75%	|31.61%
|3	|2.2709	        |2.2407	    |38.63%	         |67.54%	|34.15%	|34.60%
|4	|2.1206	        |2.1567	    |40.17%	         |68.87%	|35.16%	|37.17%
|5	|2.0133	        |2.1238	    |40.77%	         |69.27%	|35.36%	|38.17%
|6	|1.9292	        |2.1008	    |41.25%	         |69.03%	|35.44%	|39.30%
|7	|1.8592	        |2.0733	    |41.39%	         |69.12%	|35.70%	|40.04%
|8	|1.7973	        |2.0555	    |41.76%	         |69.43%	|36.01%	|40.48%
|9	|1.7381	        |2.0655	    |41.94%	         |68.62%	|36.22%	|41.25%
|10	|1.6889	        |2.0978	    |41.80%	         |68.70%	|36.07%	|41.35%
|11	|1.6428	        |2.1139	    |41.89%	         |68.57%	|36.58%	|41.58%
|12	|1.5953	        |2.1344	    |41.84%	         |68.11%	|35.78%	|42.13%
|13	|1.5523	        |2.1586	    |41.82%	         |68.20%	|35.99%	|41.85%
|14	|1.5146	        |2.1562	    |41.86%	         |68.37%	|34.90%	|42.20%

## Final result (Run 1, epoch 5)
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