---
layout: page
title: How Terrifying the Dream Is：A Word Embedding Approach
description: This project uses word embedding technology to quantitatively analyze and measure the emotional tone of dream descriptions, revealing the underlying sentiments of our subconscious narratives.
img: assets/img/project/wordembedding.png
importance: 2
category: work
related_publications: false
---


# Word Embedding
<!--#什么是词嵌入模型
词嵌入模型可以通过夹角来比较大小 -->

Word embedding is a technique in natural language processing (NLP) and machine learning where words or phrases are represented as vectors of real numbers. These vectors capture semantic and syntactic relationships between words, allowing machines to process and understand language more effectively.


Suppose we are calculating a text that contains a total of four words: cat, dog, cow, sheep. Each position in the vector represents a word.

<div class="row justify-content-center">
    <div class="col-md-6 col-lg-4 mt-3 mt-md-0">
        <figure class="figure">
            {% include figure.liquid loading="eager" path="assets/img/project/wordembedding/word1.png" title="word embedding" class="img-fluid rounded z-depth-1" %}
            <figcaption class="figure-caption text-center">Word Embedding</figcaption>
        </figure>
    </div>
</div>

<style>
    .figure {
        max-width: 100%;
        height: auto;
    }
    .figure img {
        max-width: 100%;
        height: auto;
    }
</style>

<div class="caption">
    When words are semantically similar, they are also close in space.
</div>


A standard approach to measure the similarity between embedding vectors is to compute their cosine similarity. To measure the similarity between two embedding vectors using cosine similarity, the formula is:

$$
\text{cosine similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}
$$
where:
- $$ \vec{A} $$ and $$ \vec{B} $$ are the embedding vectors.
- $$ \vec{A} \cdot \vec{B} $$ is the dot product of vectors $$ \vec{A} $$ and $$ \vec{B} $$.
- $$ \|\vec{A}\| $$ and $$ \|\vec{B}\| $$ are the magnitudes (or norms) of vectors $$ \vec{A} $$ and $$ \vec{B} $$, respectively.

# How Terrifying the Dream Is: A Word Embedding Approach


Using a rich textual [dataset](https://www.kaggle.com/datasets/sarikakv1221/dreams) comprising over 30,000 dream descriptions from Kaggle, we are able to train word embedding models. This diverse dataset encompasses a wide spectrum of dream narratives, ranging from mundane everyday occurrences to surreal fantasies, from spine-chilling nightmares to blissful reveries. Such variety enabled our models to capture the intricate nuances and multifaceted nature of dreams. The trained models are not only capable of identifying semantically similar words within dream texts but also further reveal the profound implications embedded within dreams.For instance, the features and characteristics of dreams: Is it a pleasant dream or a nightmare? To what extent is it a nightmare?


## Modelling approach
<!--#
首先将所有的梦的data训练成word embedding
然后再这个模型中，我们可以探索性地看一下在关于梦的描述中，一些概念的详尽程度。
这种描述是否能够构成一种测量？在一些研究中，我们看到了通过词嵌入技术，学者们可以测量疾病的污名化程度、文本的创新程度、文本的道德等
把这些研究是如何做的写一下。
最后，我们也可以使用词嵌入，来测量一个梦的文本的不同属性。-->

When delving into textual data describing dreams, we can employ word embedding models. Initially, algorithms such as Word2vec or GloVe are utilized to transform all dream-related textual data into word vectors within a high-dimensional space. These word vectors capture the semantic and syntactic information of words, thereby laying the groundwork for subsequent analysis.

In some studies, we have observed that through word embedding techniques, scholars are able to measure the degree of stigmatization of diseases [(Best & Arseniev-Koehler, 2023)](https://journals.sagepub.com/doi/10.1177/00031224231197436), the level of novelty in texts [(Zhou, 2022)](https://journals.sagepub.com/doi/10.1177/00031224221123030), and the morality of the political rhetoric [(Kraft & Klemmensen, 2024)](https://www.cambridge.org/core/journals/british-journal-of-political-science/article/lexical-ambiguity-in-political-rhetoric-why-morality-doesnt-fit-in-a-bag-of-words/BF369893D8B6B6FDF8292366157D84C1), among other aspects.

Following their approaches, once the model training is complete, we can conduct a series of exploratory analyses to assess the comprehensiveness of concepts within dream descriptions. For instance, we can calculate the cosine similarity between word vectors to explore how common concepts in dreams are interrelated. Furthermore, by analyzing the average of word vectors, we can evaluate the emotional tendencies and themes of the entire dream narrative.


## Training our Own Embedding
 

<!--#
Patrick的链接
https://drive.google.com/drive/folders/1nkfANtyojnRbmvp5u_7m4DqbK-i8mkid?usp=sharing


Rachael Kee
  上午 9:25
https://www.kaggle.com/datasets/sarikakv1221/dreams
kaggle.comkaggle.com
dreams
Kaggle is the world’s largest data science community with powerful tools and resources to help you achieve your data science goals.




|   | dreams_text |
|------|------------|
| 0    | 001 Nightmare in Cambodia. In the dream we are... |
| 1    | 002 The enemy is above, in the sky. We are not soldiers...|
| 2    | 003 We are on a firebase. It is night time...|
| 3   | 004 We are on an LZ; I am. saying ...|
-->



{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/wordembedding.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}



Through the implementation of word embedding techniques, we have successfully developed a preliminary methodology for quantifying the perceived terror level in dream narratives. However, it is crucial to acknowledge that the aforementioned code serves merely as a proof of concept and exhibits several limitations that warrant further refinement.

Firstly, the constrained scope of our training corpus may impede the model's ability to discern nuanced semantic variations or potentially introduce unintended biases. This limitation could be addressed by expanding and diversifying the training dataset to encompass a broader range of dream descriptions and emotional contexts.

Secondly, the reliance on a single lexical item—"terrifying"—as the sole metric for terror assessment presents an oversimplified approach to a complex psychological phenomenon. Dreams often evoke a spectrum of emotions and sensations that may not be adequately captured by a unidimensional measure.

To enhance the robustness and validity of our model, future iterations could rely on pre-trained word embeddings (GloVe), or incorporate multi-dimensional analysis techniques. Drawing upon established research in oneirology and psycholinguistics, we propose integrating lexicon-based methods to create a more comprehensive framework for dream attribute measurement. This approach would involve developing a curated dictionary of terror-related terms, accounting for various intensities and manifestations of fear in dream experiences.

Furthermore, the incorporation of advanced natural language processing techniques, such as sentiment analysis and emotion detection algorithms, could provide a more nuanced understanding of the emotional landscape within dream narratives. This multifaceted approach would not only improve the accuracy of terror quantification but also offer insights into the broader emotional and cognitive aspects of dream content.