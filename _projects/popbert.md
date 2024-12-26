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

{::nomarkdown}
<!-- 嵌入 GitHub Gist 并控制大小 -->
<div style="width: 100%; height: 1200px; overflow: auto;">
    <script src="https://gist.github.com/HaotianZhao99/77ff120266c5a221c0f18701c6c5d45e.js"></script>
</div>
{:/nomarkdown}

---------

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/01-annotator_performance.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

