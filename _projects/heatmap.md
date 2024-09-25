---
layout: page
title: Distribution Heat Map
description: Generating a Heatmap Illustrating the Spatial Distribution of China's Intangible Cultural Heritage through Python.
img: assets/img/12.jpg
<!-- 头图更换一下 -->
importance: 1
category: work
related_publications: true
---

中国的非物质文化遗产（Intangible Cultural Heritage, ICH）是指那些不以物质形态存在，而是以非物质形态传承的文化遗产。这些遗产涵盖了广泛的文化实践，包括口头传统、表演艺术、社会风俗、节日庆典以及传统手工艺技能等。从概念上讲，非物质文化遗产可以被视为一种无形的文化遗产，它包含了诸如京剧、昆曲、剪纸、刺绣、中医以及中草药等传统知识。

联合国教科文组织（United Nations Educational, Scientific and Cultural Organization, UNESCO）的非物质文化遗产名录收录了来自中国的43个项目。自2006年起，中国政府已经公布了五批国家级非物质文化遗产代表性项目，目前总计有3000个条目。国家层面对非物质文化遗产的认定和记录，对于其保护、传承和传播具有重要意义。

在一项旨在探索非物质文化遗产与农村及偏远地区经济发展关系的数据新闻研究中，研究者们尝试利用“中国非物质文化遗产网·中国非物质文化遗产数字博物馆”上记录的3610条目，并通过百度API将这些数据定位到县级精度，进而绘制出非物质文化遗产的分布热力图。

这种可视化方法能够直观地展示非物质文化遗产的地理分布情况，并且可以与中国的少数民族分布以及经济发展的区域性差异进行比较分析。通过这种比较，可以更深入地理解非物质文化遗产在促进地方经济发展和文化多样性保护方面的潜在价值。


# 数据获取与清洗
数据来源于中国非物质文化遗产网收录的国家级非物质文化遗产代表性项目名录（包括3610个子项）。通过爬虫可以获得每一项的名称、类别、时间、申报地区或单位和保护文化遗产的组织。为了绘制分布热力图，我们保留申报地区。县级精度数据示例如下

| 城市名称 |
|----------|
| 贵州省台江县 |
| 贵州省黄平县 |
| 湖南省花垣县 |
| 贵州省贵阳市清镇市 |
| 广西壮族自治区田阳县 |
| 云南省梁河县 |
| 云南省思茅市 |
| ... |

<table align="center">
<tr><th>城市名称</th></tr>
<tr><td>贵州省台江县</td></tr>
<tr><td>贵州省黄平县</td></tr>
<tr><td>湖南省花垣县</td></tr>
<tr><td>贵州省贵阳市清镇市</td></tr>
<tr><td>广西壮族自治区田阳县</td></tr>
<tr><td>云南省梁河县</td></tr>
<tr><td>云南省思茅市</td></tr>
</table>


# 获取地理信息
省级精度的数据可以在python中通过省份名称直接绘制分布图。但我们需要县级精度数据，这就需要使用百度地图的API来获取每一个县的经纬度数据，再通过经纬度数据绘制热力图。

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/heatmap1.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/heatmap1.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

最后包含经纬度数据应该如下所示。

| 城市名称       | 经度         | 纬度         |
|----------------|--------------|--------------|
| 贵州省台江县   | 108.3285516  | 26.67237254  |
| 贵州省黄平县   | 107.9235478  | 26.91128864  |
| 湖南省花垣县   | 109.4885618  | 28.57790993  |
| 贵州省贵阳市清镇市 | 106.4775226 | 26.5619879   |
| 广西壮族自治区田阳县 | 108.3345212 | 22.821269   |
| 云南省梁河县   | 98.30313363  | 24.81078446  |
| 云南省思茅市   | 100.9835551  | 22.79249798  |
| ...   | 

#### 替代性方法
当然，如果你需要的精度不高，或者数据数量是肉眼或手动可以检视的，你也可以尝试使用GPT来返回经纬度数据。不过prompt中可能需要GPT尽可能精确或参照网上公开数据。但是，GPT对于这种问题可能会一本正经地犯错。

获取经纬度的方式根据StimuMing的方法改编。



# 绘制热力图
绘制热力图可以通过pyecharts实现。https://github.com/pyecharts/pyecharts

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/heatmap2.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/heatmap2.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

Every project has a beautiful feature showcase page.

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/notebookblog.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/notebookblog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}


It's easy to include images in a flexible 3-column grid format.
Make your photos 1/3, 2/3, or full width.


Embed the heatmap using an iframe:

<iframe src="https://haotianzhao99.github.io/assets/html/china_cities_map_test.html" width="100%" height="500px" style="border:none;"></iframe>

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images, even citations {% cite einstein1950meaning %}.
Say you wanted to write a bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %}
