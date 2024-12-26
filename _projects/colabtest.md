---
layout: page
title: 🚧Test
description: 
img:
importance: 10
category: work
related_publications: false
selected: true
---




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



{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/test.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

