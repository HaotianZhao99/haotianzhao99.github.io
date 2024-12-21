```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
```

Before diving into the implementation, we need to set up our authentication with the [Hugging Face Hub](https://huggingface.co/). [Hugging Face](https://huggingface.co/) is a platform that hosts thousands of pre-trained models and datasets, making it an essential resource for modern NLP tasks. This step is crucial if you plan to work with private models or want to save your fine-tuned model to the Hub later.


```python
from huggingface_hub import login

# Log in to Hugging Face Hub using authentication token
# Required for accessing private models and pushing models to Hub
login("your_token")
```

## Data Loading and Preparation
For this sentiment analysis task, we'll use a Chinese social media dataset containing 100,000 Weibo posts with sentiment labels. The dataset is hosted on the [Hugging Face Hub](https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k) and can be easily loaded using the `datasets` library.


```python
from datasets import load_dataset
import pandas as pd

# Load sentiment analysis dataset from Hugging Face Hub
# Dataset contains 100k Weibo posts with sentiment labels
ds = load_dataset("dirtycomputer/weibo_senti_100k")
ds
```




    DatasetDict({
        train: Dataset({
            features: ['label', 'review'],
            num_rows: 119988
        })
    })




```python
# Convert Hugging Face dataset to pandas DataFrame
df = pd.DataFrame(ds['train'])

# Display basic information about the DataFrame
print("Dataset Overview:")
print(f"Number of samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nLabel distribution:")
print(df['label'].value_counts())
print("\nSample reviews:")
print(df['review'].head())

# Check for any missing values
if df.isnull().sum().any():
    print("\nWarning: Dataset contains missing values!")
```

    Dataset Overview:
    Number of samples: 119988
    Columns: ['label', 'review']
    
    Label distribution:
    label
    0    59995
    1    59993
    Name: count, dtype: int64
    
    Sample reviews:
    0                ﻿更博了，爆照了，帅的呀，就是越来越爱你！生快傻缺[爱你][爱你][爱你]
    1    @张晓鹏jonathan 土耳其的事要认真对待[哈哈]，否则直接开除。@丁丁看世界 很是细心...
    2    姑娘都羡慕你呢…还有招财猫高兴……//@爱在蔓延-JC:[哈哈]小学徒一枚，等着明天见您呢/...
    3                                           美~~~~~[爱你]
    4                                    梦想有多大，舞台就有多大![鼓掌]
    Name: review, dtype: object
    

The dataset contains 119,988 samples. The dataset is perfectly balanced with 59,995 negative samples (label 0) and 59,993 positive samples (label 1). 

## Data Splitting
Although our dataset is pre-organized, we'll create our own train-test split to ensure we have a fresh evaluation set. We'll use 80% of the data for training and reserve 20% for testing the model's performance.


```python
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets
# - test_size=0.2: 80% training, 20% testing
# - shuffle=True: randomly shuffle before splitting
# - random_state=42: set seed for reproducibility
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
```

The key parameters:

1. `test_size=0.2`: Creates an 80-20 split, with ~96,000 training samples and ~24,000 test samples
2. `shuffle=True`: Ensures random distribution of data, preventing ordering bias
3. `random_state=42`: Sets a seed for reproducible results

I run the project on [Google Colab](https://colab.research.google.com), a cloud-based Jupyter notebook environment. Colab provides free GPU access, making it an excellent choice for users without local GPU resources to run deep learning models like BERT.


```python
# Check for available CUDA device and set up GPU/CPU
# Colab typically provides a single GPU, if available
if torch.cuda.is_available():
   device = torch.device("cuda")
   # Print GPU information
   print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
   device = torch.device("cpu")
   print("No GPU available, using CPU")
```

    Using GPU: NVIDIA A100-SXM4-40GB
    GPU Memory: 42.48 GB
    

## Tokenizer Initialization
For our Chinese sentiment analysis task, we'll use the `bert-base-chinese` tokenizer. This pre-trained tokenizer is specifically designed for Chinese text.

The tokenizer is crucial for preparing our text data for BERT. It converts Chinese text into tokens that BERT can understand


```python
# Initialize the BERT Chinese tokenizer
# Uses bert-base-chinese pre-trained model's vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
```

To efficiently handle our data during training, we need to create a custom Dataset class that inherits from PyTorch's Dataset class. This class will take care of text encoding and provide a standardized way to access our samples.

It serves as a data pipeline that:
1. Transforms Chinese text into BERT-compatible token IDs
2. Ensures consistent input dimensions through padding and truncation
3. Efficiently delivers batched data during training


```python
from torch.utils.data import Dataset
import torch
class TextDataset(Dataset):
   """
   Custom Dataset class for text data, inheriting from PyTorch's Dataset.

   Parameters:
   tokenizer (Tokenizer): Tokenizer object for text encoding
   texts (list): List of text samples
   labels (list): List of corresponding labels
   """
   def __init__(self, tokenizer, texts, labels):
       # Encode texts with padding and truncation
       self.encodings = tokenizer(
           texts,
           truncation=True,
           padding=True,
           max_length=512,  # Explicitly set max length for BERT
           return_tensors='pt'  # Return PyTorch tensors directly
       )
       # Convert labels to tensor
       self.labels = torch.tensor(labels)

   def __getitem__(self, idx):
       """
       Get a single sample by index.

       Args:
           idx (int): Sample index

       Returns:
           dict: Dictionary containing encoded text data and label
       """
       return {
           'input_ids': self.encodings['input_ids'][idx],
           'attention_mask': self.encodings['attention_mask'][idx],
           'labels': self.labels[idx]
       }

   def __len__(self):
       """
       Get dataset length.

       Returns:
           int: Number of samples in dataset
       """
       return len(self.labels)
```

Let's verify our label distribution and create an explicit mapping for our sentiment classes. While our labels are already in a binary format (0 and 1), maintaining an explicit mapping is a good practice for code clarity and future modifications.


```python
# Print unique labels in the dataset
print("Unique labels in training data:", sorted(set(train_df['label'])))

# Create explicit label mapping
label_mapping = {
    0: 0,  # Negative
    1: 1   # Positive
}

# Map labels using explicit mapping
train_labels = [label_mapping[label] for label in train_df['label']]
test_labels = [label_mapping[label] for label in test_df['label']]

# Verify label distribution after mapping
print("\nLabel distribution after mapping:")
print("Training:", pd.Series(train_labels).value_counts())
print("Testing:", pd.Series(test_labels).value_counts())
```

    Unique labels in training data: [0, 1]
    
    Label distribution after mapping:
    Training: 0    48151
    1    47839
    Name: count, dtype: int64
    Testing: 1    12154
    0    11844
    Name: count, dtype: int64
    


```python
# Explicit label mapping to ensure correct sentiment assignment
label_to_index = {
    0: 0,  # Keep negative as 0
    1: 1   # Keep positive as 1
}

# Map labels using explicit mapping
train_labels = [label_to_index[label] for label in train_df['label']]
test_labels = [label_to_index[label] for label in test_df['label']]

# Create datasets with verified labels
train_dataset = TextDataset(tokenizer, train_df['review'].tolist(), train_labels)
test_dataset = TextDataset(tokenizer, test_df['review'].tolist(), test_labels)

# Verify final mapping
print("\nFinal verification:")
print("Training set label distribution:", pd.Series(train_labels).value_counts())
print("Test set label distribution:", pd.Series(test_labels).value_counts())
```

    
    Final verification:
    Training set label distribution: 0    48151
    1    47839
    Name: count, dtype: int64
    Test set label distribution: 1    12154
    0    11844
    Name: count, dtype: int64
    

## Model Training Setup
To fine-tune BERT for our sentiment analysis task, we'll follow these key steps:

1. **Model Initialization**: Load the pre-trained Chinese BERT model
2. **Training Configuration**: Set up training parameters using `TrainingArguments`
3. **Metrics Setup**: Define evaluation metrics for model performance monitoring
4. **Trainer Setup**: Initialize the Hugging Face `Trainer` class with:
   - The BERT model
   - Training arguments
   - Training and evaluation datasets
   - Metrics computation function
5. **Training Process**: Use `trainer.train()` and `trainer.evaluate()` for model fine-tuning and evaluation

The Hugging Face Trainer API simplifies the training process by handling the training loops, device management, and model optimization automatically.

The code below implements these steps:


```python
# Load pre-trained Chinese BERT model and configure for binary classification
model = BertForSequenceClassification.from_pretrained(
   'bert-base-chinese',
   num_labels=2  # Binary classification (negative/positive)
)
model = model.to(device)

# Define training arguments for model fine-tuning
training_args = TrainingArguments(
   output_dir='sentiment-weibo-100k-fine-tuned-bert-test',  # Directory to save model checkpoints
   num_train_epochs=3,                                 # Number of training epochs
   per_device_train_batch_size=32,                    # Number of samples per training batch
   per_device_eval_batch_size=64,                     # Number of samples per evaluation batch
   warmup_steps=500,                                  # Steps for learning rate warmup
   weight_decay=0.01,                                 # L2 regularization factor
   logging_dir='./logs',                             # Directory for training logs
   logging_steps=100,                                # Log metrics every 100 steps
   evaluation_strategy="epoch",                      # Evaluate after each epoch
   save_strategy="epoch",                           # Save model after each epoch
   load_best_model_at_end=True,                    # Load best model after training
   push_to_hub=True,                               # Push model to Hugging Face Hub
   learning_rate=2e-5,                             # Initial learning rate
   gradient_accumulation_steps=1                   # Update model after every batch
)

def compute_metrics(pred):
   """
   Compute evaluation metrics for the model
   Args:
       pred: Contains predictions and label_ids
   Returns:
       dict: Dictionary containing accuracy, F1, precision, and recall scores
   """
   labels = pred.label_ids
   preds = pred.predictions.argmax(-1)
   precision, recall, f1, _ = precision_recall_fscore_support(
       labels,
       preds,
       average='binary',
       pos_label=1  # Define positive class for binary metrics
   )
   acc = accuracy_score(labels, preds)

   return {
       'accuracy': acc,
       'f1': f1,
       'precision': precision,
       'recall': recall
   }

# Initialize trainer with model and training configuration
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_dataset,
   eval_dataset=test_dataset,
   compute_metrics=compute_metrics
)

# Print training configuration summary
print("Training Configuration:")
print(f"Model: bert-base-chinese")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Number of epochs: {training_args.num_train_epochs}")

# Start training
trainer.train()

# Evaluate model performance
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)
```


    model.safetensors:   0%|          | 0.00/412M [00:00<?, ?B/s]


    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-chinese and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    /usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1568: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    [34m[1mwandb[0m: [33mWARNING[0m The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
    

    Training Configuration:
    Model: bert-base-chinese
    Training samples: 95990
    Test samples: 23998
    Batch size: 32
    Number of epochs: 3
    

    [34m[1mwandb[0m: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
    


    <IPython.core.display.Javascript object>


    [34m[1mwandb[0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
    [34m[1mwandb[0m: You can find your API key in your browser here: https://wandb.ai/authorize
    wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:

     ··········
    

    [34m[1mwandb[0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc
    


Tracking run with wandb version 0.18.7



Run data is saved locally in <code>/content/wandb/run-20241218_150159-oimgwuen</code>



Syncing run <strong><a href='https://wandb.ai/zhaohaotian99-huazhong-university-of-science-and-technology/huggingface/runs/oimgwuen' target="_blank">sentiment-weibo-100k-fine-tuned-bert</a></strong> to <a href='https://wandb.ai/zhaohaotian99-huazhong-university-of-science-and-technology/huggingface' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br/>



View project at <a href='https://wandb.ai/zhaohaotian99-huazhong-university-of-science-and-technology/huggingface' target="_blank">https://wandb.ai/zhaohaotian99-huazhong-university-of-science-and-technology/huggingface</a>



View run at <a href='https://wandb.ai/zhaohaotian99-huazhong-university-of-science-and-technology/huggingface/runs/oimgwuen' target="_blank">https://wandb.ai/zhaohaotian99-huazhong-university-of-science-and-technology/huggingface/runs/oimgwuen</a>




    <div>

      <progress value='9000' max='9000' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [9000/9000 38:15, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.033300</td>
      <td>0.039774</td>
      <td>0.984040</td>
      <td>0.983992</td>
      <td>1.000000</td>
      <td>0.968488</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.034900</td>
      <td>0.037043</td>
      <td>0.983790</td>
      <td>0.983737</td>
      <td>1.000000</td>
      <td>0.967994</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.037600</td>
      <td>0.036484</td>
      <td>0.984040</td>
      <td>0.983992</td>
      <td>1.000000</td>
      <td>0.968488</td>
    </tr>
  </tbody>
</table><p>


    No files have been modified since last commit. Skipping to prevent empty commit.
    WARNING:huggingface_hub.hf_api:No files have been modified since last commit. Skipping to prevent empty commit.
    



<div>

  <progress value='375' max='375' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [375/375 00:50]
</div>



    
    Evaluation Results: {'eval_loss': 0.03648396208882332, 'eval_accuracy': 0.9840403366947246, 'eval_f1': 0.9839916405433646, 'eval_precision': 1.0, 'eval_recall': 0.9684877406615107, 'eval_runtime': 50.6448, 'eval_samples_per_second': 473.849, 'eval_steps_per_second': 7.405, 'epoch': 3.0}
    
