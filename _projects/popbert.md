---
layout: page
title: PopBert：Unveiling Populist Language in Political Discourse
description: In this project, I am learning and replicating the key steps in building the PopBERT model from the research "PopBERT. Detecting Populism and Its Host Ideologies in the German Bundestag" by Erhard et al. (2024). This study develops a transformer-based model to detect populist language in German parliamentary speeches, focusing on the moralizing references to "the virtuous people" and "the corrupt elite." 
img: assets/img/project/bert.png
importance: 5
category: work
related_publications: false
selected: true
---

<!-- 在本地的修改：gemfile中   #gem 'mini_racer' 注释掉了；source更换掉了；2020-09-28-twitter.md删掉了; external-posts.rb无效了;config中  #- jekyll-twitter-plugin和imagemagick:
enabled: true改成了false
  
  
在提交的适合都要改回来，防止有问题

只提交修改的project md和asset文件

-->

PopBERT is a transformer-based language model developed by researchers at the University of Stuttgart to detect populist language in German political discourse. The study focuses on analyzing speeches from the German Bundestag (parliament) between 2013-2021. The model was trained to identify two core dimensions of populism: anti-elitism and people-centrism, along with their associated left-wing and right-wing host ideologies. Using GBERTLarge as its foundation, PopBERT demonstrates strong performance in detecting these populist elements in political speeches. The researchers validated the model through multiple approaches, including comparison with expert surveys and out-of-sample testing. This tool enables researchers to conduct dynamic analyses of how German-speaking politicians and parties employ populist language as a strategic device.  

In this project, we will study and reproduce the key steps of PopBERT, focusing on its transformer-based architecture for detecting populist language. We'll pay particular attention to understanding its data preprocessing steps, model architecture based on GBERTLarge, and the multi-label classification approach for identifying both core populist dimensions and their host ideologies.

# Training the Model

The code used in this project comes from the original authors' open-source repository on [GitHub](https://github.com/luerhard/PopBERT). I have cloned the repository to Google Drive and will be running the experiments on [Google Colab](https://colab.research.google.com/) for its accessible GPU resources and easy integration with Drive.  

The following sections will walk through the key steps of model training and implementation.  


## A Glimpse into the Dataset

| id     | text | username | elite | centr | left | right |
|--------|------|----------|-------|-------|------|-------|
| 446633 | Ihre willkürliche Zusammenstellung und Auflistung alter Forderungen zeigt, dass Sie nicht willens sind, einen verantwortlichen und gesellschaftlich akzeptierbaren Beitrag zum Tierschutz in Deutschland zu leisten, sondern dass Sie Emotionen schüren wollen. | riedel | True | False | False | False |
| 446633 | Ihre willkürliche Zusammenstellung und Auflistung alter Forderungen zeigt, dass Sie nicht willens sind, einen verantwortlichen und gesellschaftlich akzeptierbaren Beitrag zum Tierschutz in Deutschland zu leisten, sondern dass Sie Emotionen schüren wollen. | richter | False | False | False | False |
| 446633 | Ihre willkürliche Zusammenstellung und Auflistung alter Forderungen zeigt, dass Sie nicht willens sind, einen verantwortlichen und gesellschaftlich akzeptierbaren Beitrag zum Tierschutz in Deutschland zu leisten, sondern dass Sie Emotionen schüren wollen. | grabsch | False | False | False | False |
| 119028 | Einen ganz herzlichen Dank an diese Kolleginnen und Kollegen, an die Mitarbeiter des Ausschusses und alle Bürgerinnen und Bürger, die sich aktiv für unser Gemeinwohl einsetzen. | schadt | False | False | False | False |
| 119028 | Einen ganz herzlichen Dank an diese Kolleginnen und Kollegen, an die Mitarbeiter des Ausschusses und alle Bürgerinnen und Bürger, die sich aktiv für unser Gemeinwohl einsetzen. | coudry | False | False | False | False |  

[35180 rows x 7 columns]  
Total annotations: 35180  
Number of labels: 4 (elite, centr, left, right)  
Number of unique texts: 7036  
Samples with labels: 3515 (49.96%)  
Samples without labels(all annotators marked all 4 label dimensions as zero): 3521 (50.04%)  

## Create PopBERT Model


<!-- 通过 GitHub Gist 嵌入 notebook -->
<p>
<script src="https://gist.github.com/HaotianZhao99/77ff120266c5a221c0f18701c6c5d45e.js"></script>
</p>


In the code above, we can see the main framework of the training process, including optimizer setup, learning rate scheduling strategy, and the main training loop. However, to deeply understand how the model is specifically trained, we need to examine the implementation of train_epoch and eval_epoch functions in the `training.py` file.


## `training.py`: Understanding the Training Implementation


The author encapsulates the specific implementations through different Python modules in the src(source) directory. This modular design makes the code structure clearer and easier to maintain and reuse. 

Next, we will delve into the implementation details in the training.py file. This file contains the core logic of model training. By analyzing the code line by line, we can better understand the specific operational steps, loss calculation methods, and evaluation approaches during the BERT model training process. We will pay special attention to the implementation of two key functions: train_epoch and eval_epoch.

```python
import numpy as np
import torch
from sklearn.metrics import f1_score

import src.bert.utils as bert_utils
```

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
Uses "cuda" if GPU is available. This `DEVICE` variable will be used to specify where the model and data should run

```python
def train_epoch(model, dataloader, optimizer, lr_scheduler, clip):
    train_loss = 0.0                         # Initialize total loss
    model.train()                            # Set model to training mode
    for batch in dataloader:
        optimizer.zero_grad()                # Clear previous gradients
        
        # Move data to device and forward propagation
        encodings = batch["encodings"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(**encodings, labels=labels)

        out.loss.backward()                  # Calculate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # Gradient clipping
        optimizer.step()                     # Update model parameters
        lr_scheduler.step()                  # Update learning rate

        train_loss += out.loss.item()        # Accumulate batch loss

    return train_loss
```
Key Steps
- Gradient Zeroing  `optimizer.zero_grad()`  

  In machine learning, a gradient is a vector that represents the rate of change of the loss function at the current point. Imagine you're on a mountain, blindfolded, and need to find your way to the bottom (similar to finding the minimum of a loss function in machine learning). In this case, the gradient is like the slope of the ground beneath your feet. The gradient tells you two important pieces of information:
  1. The direction of the slope (whether it's going up or down)
  2. The steepness of the slope (whether it's steep or gentle)  

  Therefore, if we want to minimize the loss, we need to move in the opposite direction of the gradient.  

  Before each batch, we must clear previously accumulated gradients. Without zeroing, new gradients would add to old ones, leading to inaccurate parameter updates.  

- Forward Propagation `out = model(**encodings, labels=labels)`  

  In a BERT model, when we input a sentence, it first gets converted into numbers (word embeddings). These numbers then pass through each layer of the model sequentially. Each layer performs specific mathematical operations and transformations on these numbers, ultimately producing the result we need (like text classification predictions).  

  The data flows forward like a stream, transforming layer by layer until we get our desired output. Each layer contributes its own processing to the final prediction. This is why we call it "forward" propagation, as data flows from input towards output direction.  

- Backward Propagation `out.loss.backward()`  

  In deep learning, backpropagation is the model's process of "looking back." After making a prediction, the model calculates the error (loss) between predicted and actual values, then works backwards from the output layer, calculating how much each parameter contributed to this error. It's like unraveling a chain of cause and effect to figure out "which parameters need adjustment and by how much" to reduce prediction errors.  

  In PyTorch, the `backward()` method is the core of automatic differentiation. When we call `out.loss.backward()`, PyTorch starts from the loss value and performs backpropagation through the computational graph, calculating how each parameter influenced the loss (gradients).  

- Gradient Clipping `torch.nn.utils.clip_grad_norm_(model.parameters(), clip)`    

  In deep learning, when gradient values become too large, model training can become unstable. Gradient clipping works by setting a threshold - when gradients exceed this threshold, they are proportionally scaled down to keep them within a reasonable range. This prevents "explosion" phenomena during model training and makes the training process more stable.  

- Parameter Update `optimizer.step()`  
  
  Updates model parameters using the optimizer (in this study, AdamW) based on computed gradients. The optimizer determines how to adjust parameters using gradient information.  
  
  AdamW is a widely used optimization algorithm, an improved version of Adam optimizer. It combines two important ideas: adaptive learning rates and weight decay. Like an experienced teacher, it knows when to take big steps in learning (larger learning rate) and when to slow down for careful consideration (smaller learning rate).  

- Learning Rate Adjustment `lr_scheduler.step()`  

  Adjusts learning rate according to a preset strategy.  

These steps form the basic training loop in deep learning  

--------

```python
def eval_epoch(model, dataloader):
    eval_loss = 0.0
    y_true = []
    y_pred = []
    model.eval()             # Switch the model to evaluation mode
    with torch.inference_mode():
        for batch in dataloader:
            encodings = batch["encodings"]
            encodings = encodings.to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(**encodings, labels=labels)        #Feed the feature vectors into the model, obtaining the output out which contains logits and loss.
            if model.config.problem_type == "multi_label_classification":
                preds = torch.nn.functional.sigmoid(out.logits)
            elif model.config.problem_type == "single_label_classification":
                preds = torch.nn.functional.softmax(out.logits)

            eval_loss += out.loss.item()
            y_true.extend(batch["labels"].numpy())
            y_pred.extend(preds.to("cpu").detach().numpy())

    y_true = np.array(y_true)
    if model.config.problem_type == "multi_label_classification":
        y_pred_bin = np.where(np.array(y_pred) > 0.5, 1, 0)
        thresh_finder = bert_utils.ThresholdFinder(type="single_task")
        thresholds = thresh_finder.find_thresholds(np.array(y_true), np.array(y_pred))
    elif model.config.problem_type == "single_label_classification":
        y_pred_bin = np.argmax(np.array(y_pred), axis=1)
        thresholds = {k: 0.5 for k in range(model.config.num_labels)}

    score = f1_score(np.array(y_true), np.array(y_pred_bin), average="macro", zero_division=0)

    return eval_loss, score, thresholds
```
This code defines a function eval_epoch to evaluate the performance of the model.  

- Single-label classification:  
  
  In single-label classification, the model outputs a probability distribution processed through `softmax`, where each class's probability sums to 1. During prediction, the class with the highest probability is selected as the final prediction.

- Multi-label classification:  
 In multi-label classification, the model outputs independent probabilities for each label, indicating the likelihood of the sample belonging to each class. During prediction, a threshold (e.g., 0.5) is usually applied: if the probability of a class exceeds this threshold, the sample is considered to belong to that class.

- F1 Score Calculation
  In single-label classification, `argmax` is used to select the predicted class for each sample. This means the model makes one prediction per sample.   

  In multi-label classification, since each sample can belong to multiple labels, F1 scores are computed by comparing the binarized predictions with the true labels. In the code, `np.where` is used to convert probabilities into binary values (0 or 1), and then a macro average F1 score is calculated.

--------

# Testing with Our Trained Model

Now that we've successfully trained our model, let's test it on a few example instances. Using the fine-tuned model, we can input several sentences and observe how well the model performs in classifying them.  

<!-- 通过 GitHub Gist 嵌入 notebook -->
<p>
<script src="https://gist.github.com/HaotianZhao99/6e37a259ded5212a476068866a71cdc9.js"></script>
</p>

Overall, our trained model performed well, successfully distinguishing Anti-elitism, People-centrism, Left-wing, and Right-wing, as well as identifying sentences that exhibit both Anti-elitism and People-centrism.

