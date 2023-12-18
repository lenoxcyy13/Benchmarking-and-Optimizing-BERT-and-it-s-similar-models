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
	- `yelp_model-128-16`
		Trained bert-base-cased model
	- `yelp_model-128-16-un`
		Trained bert-base-uncased model
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
