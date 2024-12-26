---
layout: page
title: 🚧 PopBert：Unveiling Populist Language in Political Discourse
description: In this project, I am learning and replicating the key steps in building the PopBERT model from the research "PopBERT. Detecting Populism and Its Host Ideologies in the German Bundestag" by Erhard et al. (2024). This study develops a transformer-based model to detect populist language in German parliamentary speeches, focusing on the moralizing references to "the virtuous people" and "the corrupt elite." 
img: 
importance: 8
category: work
related_publications: false
selected: true
---

<!-- 本地 Jupyter Notebook 显示 -->
{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/02-create_popbert_model.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

<!-- Colab 版本嵌入 -->
<div class="mt-4">
  <iframe
    src="https://colab.research.google.com/drive/13Dud9TfNVx-AmA7QrjXi5_ZPEclv5EoP/embed"
    width="100%"
    height="800px"
    frameborder="0"
    allowfullscreen>
  </iframe>
</div>

------------


{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/01-annotator_performance.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

