---
layout: page
title: Analyzing Content Controversy in Online Communities Using Word Embedding
description: An experimental exploration of measuring controversy on Zhihu using word embeddings, revealing unexpected patterns in how controversial content drives user engagement.
img: assets/img/project/zhihu.png
importance: 3
category: work
related_publications: false
---
How can we objectively identify and quantify controversy in online communities where controversial content often sparks heated discussions? This article introduces a word embedding-based approach to analyze the degree of controversy in social platform content and its relationship with user interactions.

Our analysis leverages the ZhihuRec Dataset, a comprehensive collection of user interactions from Zhihu (知乎), which can be considered China's equivalent to Quora. Like Quora, Zhihu is a question-and-answer platform where users can ask questions, provide answers, and engage with content through various interactions such as upvotes, downvotes, and comments. Both platforms share similar core features:

This rich dataset, collaboratively released by Tsinghua University and Zhihu, encompasses: around 100M interactions collected within 10 days, 798K users, 165K questions, 554K answers, 240K authors, 70K topics, and more than 501K user query logs. There are also descriptions of users, answers, questions, authors, and topics, which are anonymous. 

What makes this dataset particularly valuable for controversy analysis is its sophisticated representation of textual content: each piece of text is encoded as a 64-dimensional word embedding vector, enabling direct mathematical comparison of semantic content. While traditional controversy analysis often relies on manual coding or simple keyword matching, our word embedding approach offers a more nuanced and scalable method for identifying and measuring controversial content. This is particularly valuable in the context of knowledge-sharing platforms, where controversies often manifest through subtle differences in perspective rather than explicit confrontation.

## What Are Word Embeddings?

At its core, word embedding transforms text into dense numerical vectors within a high-dimensional space. In our implementation, each word is mapped to a 64-dimensional vector through advanced machine learning algorithms. These vectors capture subtle semantic relationships by learning from millions of text examples.

Semantic Relationships
- Words with similar meanings cluster together in vector space
- Semantic relationships are captured through vector arithmetic
- Example: The vectors for "innovative" and "groundbreaking" would be closely aligned

Mathematical Operations 
- Support vector addition and subtraction for semantic composition
- Enable efficient similarity calculations through vector operations
- Facilitate large-scale processing and analysis
- Allow for quantitative comparison of text segments


## Measuring Controversy Through Vector Analysis
We hypothesize that controversial content tends to diverge significantly from the semantic mainstream of related discussions. Our approach quantifies this divergence through vector mathematics:

Vector Representation 
```python
# Transform an answer into its vector representation
answer_vector = np.mean([word_vectors[word] for word in answer_words])
```
- Extract word vectors for each term
- Compute a weighted average to represent the entire text

Semantic Distance Calculation
```python
# Measure semantic distance between answers
divergence = cosine(answer1_vector, answer2_vector)
```
- Implement cosine similarity metrics
- Higher values indicate greater conceptual disagreement

Controversy Scoring 
```python
# Generate controversy score
controversy_score = mean([cosine(answer_vector, other_vector) 
                         for other_vector in other_answers])
```
- Calculate average semantic distance from peer responses
- Normalize scores for cross-platform comparison

Having established our theoretical framework for analyzing controversy through word embeddings, let's dive into the practical implementation. We'll walk through a detailed Python implementation that processes millions of user interactions from the ZhihuRec Dataset. Our code demonstrates how to transform abstract mathematical concepts into concrete measurements of controversy, leveraging pandas for data manipulation, numpy for numerical operations, and scipy for statistical analysis. 



{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/zhihu_embedding.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}




# Analysis of the Relationship Between Content Controversy and User Engagement

We conducted a comprehensive statistical analysis of 54,758 questions to investigate the relationship between content controversy and user engagement patterns. The analysis employed both Pearson and Spearman correlation coefficients to examine linear and non-linear relationships, complemented by descriptive statistics to understand the underlying distributions.

## Correlation Analysis Results
Our investigation into the relationship between content controversy and user engagement revealed complex patterns that cannot be adequately captured by simple linear correlations. The Pearson correlation coefficients demonstrated remarkably weak linear relationships, with values hovering near zero across all engagement metrics. The highest linear correlation was observed with likes (r = 0.0234), while collections showed the strongest negative correlation (r = -0.0162). These minimal coefficients suggest that the relationship between controversy and engagement does not follow a straightforward linear pattern.

However, the Spearman correlation analysis unveiled more nuanced relationships, with coefficients consistently higher than their Pearson counterparts, ranging from 0.07 to 0.17. This marked difference between Pearson and Spearman correlations suggests the presence of significant non-linear relationships between controversy and user engagement behaviors. Of particular interest is the finding that controversial content tends to elicit polarized responses, as evidenced by the strongest correlations being with dislikes (ρ = 0.1665) and likes (ρ = 0.1526).

The disparity between Pearson and Spearman correlations, particularly notable in the case of dislikes (0.0187 vs 0.1665), reveals a sophisticated pattern of user behavior. While increased controversy does not directly translate to a proportional increase in negative feedback, more controversial content consistently attracts higher levels of polarized engagement. This pattern suggests that users tend to adopt strong positions on controversial content, either strongly supporting or opposing it, with relatively few neutral responses.

## Distribution Analysis
The controversy score distribution analysis reveals a predominantly moderate discourse environment on the platform. With a mean controversy score of 0.2253 (SD = 0.1625), and 75% of content falling below a controversy score of 0.2899, the data suggests that the majority of content maintains a relatively measured level of controversy. However, the maximum controversy score of 1.4645 indicates the presence of highly contentious outliers, representing instances of significant opinion divergence within the community.

The engagement metrics demonstrate a characteristic long-tail distribution, with a mean of 8,070 interactions but a substantially larger standard deviation of 38,421. This highly skewed distribution is further illustrated by the stark contrast between the median engagement of 206 and the maximum of approximately 1.52 million interactions. This pattern aligns with the well-documented Pareto principle in social media engagement, where a small proportion of content attracts a disproportionately large share of user interactions.

## Summary

Before drawing any final conclusions, we should acknowledge some important limitations of our analysis. Perhaps the most significant one is that we're working somewhat "blindly" here - we haven't actually looked at the real text content of these discussions. We're relying on word embeddings to capture different opinions, but this approach has its limitations:
- Word embeddings might miss crucial context: While they're good at capturing semantic relationships, they might not fully grasp the nuanced ways people express opposing viewpoints. For instance, two comments might use similar words but express completely different opinions through tone, sarcasm, or complex reasoning.
- Controversy isn't just about word choice: Real controversy often lies in the underlying arguments, personal experiences, and value systems that people bring to a discussion. These deeper elements might not be fully captured by our current measurement approach.
- Missing qualitative insights: Without examining the actual content, we can't verify whether what we're measuring as "controversy" aligns with what human readers would consider controversial. Some topics might appear controversial in our metrics but actually represent healthy debate, while others might appear mild but contain deeply divisive undertones.

These findings have several important implications for understanding content dynamics on social platforms. First, the weak linear correlations but stronger non-linear relationships suggest that controversy's influence on user engagement operates through complex mechanisms rather than direct causal relationships. Second, the polarized nature of engagement with controversial content indicates that such material tends to segment the audience into distinct opposing groups, potentially contributing to the formation of echo chambers or opinion clusters within the platform.

The moderate overall controversy levels, combined with the presence of high-controversy outliers, suggests an effectively self-regulating content ecosystem where extreme positions exist but do not dominate the discourse. Meanwhile, the highly skewed engagement distribution highlights the platform's capacity to occasionally produce viral content that achieves engagement levels orders of magnitude higher than typical posts.

These insights contribute to our understanding of how controversy interacts with user engagement in social media environments, while also raising important questions about the role of controversial content in driving user participation and community polarization. Future research might explore the temporal dynamics of these relationships and investigate the specific characteristics of content that achieves both high controversy and high engagement metrics.

 While our findings provide interesting insights, they should be considered alongside other forms of analysis, including qualitative review of actual content and community feedback. The patterns we've identified are valuable starting points for discussion, but they're not the whole story.

