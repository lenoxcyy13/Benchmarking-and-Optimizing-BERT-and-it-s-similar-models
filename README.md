# Benchmarking and Optimizing BERT and it's similar models

The goal is to fine-tune pretrained BERT, RoBERTa, and DistilBERT, in order to improve its performance on the general language understanding, and train itself to perform sentiment analysis on Yelp Reviews. 

## Project milestones

Model Selection and Pre-training --_COMPLETED_
- Choose pre-trained language models (BERT, RoBERTa, and DistilBERT).
- Implement fine-tuning strategies (whole model, last few layers) and  
   explore hyperparameters.
   
Training on Cloud GPUs --_COMPLETED_
- Set up the training environment on the NYU HPC platform and Google Colab.
- Experiment with different GPUs to find optimal performance.

Fine-tuning Techniques --_COMPLETED_
- Implement various fine-tuning techniques, adjusting learning rates, batch sizes, and epochs.

Evaluation and Benchmarking --_COMPLETED_
- Evaluate the models using sentiment benchmark metrics.
- Compare the performance of different models and fine-tuning methods.
    
## Description of the repository and code structure

Each folder with the name of model contains the training code, evaluating code and all the outputs related to the corresponding model.
-   `README.md`
	Project documentation, including setup instructions and usage guidelines.
- `BERT`
	- `bert-fine-tune.py`
		Python script for training the sentiment analysis model using the BERT architecture
	- `evaluate.ipynb`
		Jutypernotebook for evaluating the trained BERT model on test data
	- `bert-based-cased.out`
		Output of the bert-base-cased training process
	- `bert-case-uncased.out`
		Output of the bert-base-uncased training process
	- `bert-cased.png`
		Image of confution matrix of tesing data with bert-base-cased
	- `bert-uncased.png`
		Image of confution matrix of tesing data with bert-base-uncased
	
## Example commands to execute the code

**Model Training (BERT):** 

    python BERT/bert-fine-tune.py
   
  **Model Training (BERT):** 
1. Download the notebook from BERT/evaluate.ipynb 
2. Start the notebook server from the command line/terminal with command `jupyter notebook`
3. Run the notebook


## Results and observations

**Summary of Main Results**
|**Model**|BERT (uncased/cased)|**RoBERTa**|**DistilBERT**|
|--|--|--|--|
|**Accuracy**|82.38% / 78%|93.14%|76.28%|
|**Parameters**|109M / 108M|335M|64M|
|**Response Time**|0.041s / 0.021s|0.065s|0.023s|

**BERT**
The results of our experiment showed that the uncased version of BERT outperformed the cased version, achieving an accuracy of 82.38% compared to 75.1%. However, both models demonstrated effective sentiment analysis prediction on the test data, indicating that BERT is a powerful tool for natural language processing.

Uncased BERT
![image](https://github.com/lenoxcyy13/Benchmarking-and-Optimizing-BERT-and-it-s-similar-models/blob/main/BERT/bert-uncased.png?raw=true)

Cased BERT
![image](https://github.com/lenoxcyy13/Benchmarking-and-Optimizing-BERT-and-it-s-similar-models/assets/55534873/2fce429f-034c-4b93-8a21-3042032a8fc5)


-   The model retrieves better results when ratings are split into positive (3-5) and negative (1-2) stars.
	- 5 classes training allows a more fine-grained analysis of sentiment. It enables the model to capture and distinguish between different levels or nuances of sentiment
	- Converting to binary class allows keeping the benefits of fine-grained sentiment analysis during model development while eventually adopting a simpler binary classification for deployment
    
-   Capitalization is not crucial in this experiment since we are focusing on the sequence and meaning of the sentences rather than grammatical conventions.
    
-   RoBERTa outperforms BERT because of its dynamic masking, which enables it to learn more robust features.
