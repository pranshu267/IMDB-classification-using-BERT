import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:

    
    return dataset.map(lambda sentences: tokenizer(sentences["text"], padding = "max_length", truncation = True, max_length = 512, return_tensors="pt"), batched = True)
    

def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:

    model = BertForSequenceClassification.from_pretrained(model_name)
    
    if use_bitfit:
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False
                
    return model
    
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)

def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:

    training_args = TrainingArguments(
    output_dir="/checkpoints",  # Save models to 'checkpoints' directory
    num_train_epochs=4,  # Total number of training epochs
    do_train = True,
    evaluation_strategy = "epoch",
    save_strategy = "epoch"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model_init = lambda: init_model(None, model_name, use_bitfit),  # Dynamic model initialization
        args = training_args,
        train_dataset = train_data,
        eval_dataset = val_data,
        compute_metrics = compute_metrics,  # Compute accuracy for evaluation
    )

    return trainer

def hyperparameter_search_settings() -> Dict[str, Any]:

    lr = [3e-4, 1e-4, 5e-5, 3e-5]
    batch_size = [8, 16, 32, 64, 128]
    n_trials = len(lr) * len(batch_size)
    
    def search_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", lr ),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", batch_size)
            }
    
    trial_params = {"learning_rate" : lr,
            "per_device_train_batch_size":batch_size}
    
    return {
         "direction": "maximize",
         "backend": "optuna",
         "n_trials": n_trials,  
         "hp_space": search_space,
         "sampler": optuna.samplers.GridSampler(trial_params) 
            }

if __name__ == "__main__":  
    model_name = "prajjwal1/bert-tiny"

    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
