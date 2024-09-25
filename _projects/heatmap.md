---
layout: page
title: Heatmap Visualization of Intangible Cultural Heritage 
description: This project uses Python and Baidu Map API to create a spatial heatmap visualization of China's Intangible Cultural Heritage (ICH), revealing its distribution across the country at the county level.
img: assets/img/project/heat.png
importance: 1
category: work
related_publications: false
---
China's Intangible Cultural Heritage (ICH) refers to cultural heritage that exists in non-material forms and is transmitted through intangible means. This heritage encompasses a wide range of cultural practices, including oral traditions, performing arts, social customs, festivals, and traditional craftsmanship skills. Conceptually, ICH can be viewed as a form of intangible cultural legacy, encompassing traditional knowledge such as Beijing Opera, Kunqu Opera, paper-cutting, embroidery, Traditional Chinese Medicine, and herbal medicine.

The United Nations Educational, Scientific and Cultural Organization (UNESCO) has inscribed 43 elements from China on its [Lists of the Intangible Cultural Heritage of Humanity](https://ich.unesco.org/en/lists). Since 2006, the Chinese government has announced five batches of nationally representative ICH projects, currently totaling 3,000 entries. The national-level recognition and documentation of ICH are crucial for its protection, transmission, and dissemination.

In a data journalism research project aimed at exploring the relationship between ICH and economic development in rural and remote areas, we attempted to utilize 3,610 entries recorded on the "[China Intangible Cultural Heritage · China Intangible Cultural Heritage Digital Museum](https://www.ihchina.cn/project#target1)" Using the [Baidu Map API](https://lbsyun.baidu.com/), we geocoded these data to county-level precision, subsequently generating a heat map of ICH distribution.

<!--
中国的非物质文化遗产（Intangible Cultural Heritage, ICH）是指那些不以物质形态存在，而是以非物质形态传承的文化遗产。这些遗产涵盖了广泛的文化实践，包括口头传统、表演艺术、社会风俗、节日庆典以及传统手工艺技能等。从概念上讲，非物质文化遗产可以被视为一种无形的文化遗产，它包含了诸如京剧、昆曲、剪纸、刺绣、中医以及中草药等传统知识。

联合国教科文组织（United Nations Educational, Scientific and Cultural Organization, UNESCO）的非物质文化遗产名录收录了来自中国的43个项目。自2006年起，中国政府已经公布了五批国家级非物质文化遗产代表性项目，目前总计有3000个条目。国家层面对非物质文化遗产的认定和记录，对于其保护、传承和传播具有重要意义。

在一项旨在探索非物质文化遗产与农村及偏远地区经济发展关系的数据新闻研究中，研究者们尝试利用“中国非物质文化遗产网·中国非物质文化遗产数字博物馆”上记录的3610条目，并通过百度API将这些数据定位到县级精度，进而绘制出非物质文化遗产的分布热力图。

这种可视化方法能够直观地展示非物质文化遗产的地理分布情况，并且可以与中国的少数民族分布以及经济发展的区域性差异进行比较分析。通过这种比较，可以更深入地理解非物质文化遗产在促进地方经济发展和文化多样性保护方面的潜在价值。
-->

# Data crawling
The data source is the list of nationally representative ICH projects (including 3,610 sub-items) recorded on the China Intangible Cultural Heritage Network. Through web scraping, information such as the name, category, time, nominating region or unit, and the organization responsible for protecting the cultural heritage can be obtained for each item. To create the distribution heat map, we retained the nominating region. An example of county-level precision data is as follows:

<!--
# 数据获取与清洗
数据来源于中国非物质文化遗产网收录的国家级非物质文化遗产代表性项目名录（包括3610个子项）。通过爬虫可以获得每一项的名称、类别、时间、申报地区或单位和保护文化遗产的组织。为了绘制分布热力图，我们保留申报地区。县级精度数据示例如下
-->
<table align="center"> 
<tr><th>County Name</th></tr> 
<tr><td>Guizhou Province Taijiang County</td></tr> 
<tr><td>Guizhou Province Huangping County</td></tr> 
<tr><td>Hunan Province Huyuan County</td></tr> 
<tr><td>Guizhou Province Guiyang City Qingzhen City</td></tr> 
<tr><td>Guangxi Zhuang Autonomous Region Tianyang County</td></tr> 
<tr><td>Yunnan Province Lianghe County</td></tr> 
<tr><td>Yunnan Province Simao City</td></tr> 
<tr><td>...</td></tr> </table>


# Obtaining Geographic Information
In Python, it is possible to draw distribution maps with provincial-level accuracy by using the names of provinces. However, to achieve county-level accuracy, we need to obtain the latitude and longitude data for each county using the API provided by Baidu Maps. After obtaining the latitude and longitude data, we can then proceed to draw a heat map.
<!--# 获取地理信息
省级精度的数据可以在python中通过省份名称直接绘制分布图。但我们需要县级精度数据，这就需要使用百度地图的API来获取每一个县的经纬度数据，再通过经纬度数据绘制热力图。-->

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/heatmap1.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/heatmap1.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

The final inclusion of latitude and longitude data should be as follows:

| County Name                 | Longitude       | Latitude       |
|---------------------------|-----------------|----------------|
| Guizhou Province Taijiang County   | 108.3285516    | 26.67237254   |
| Guizhou Province Huangping County  | 107.9235478    | 26.91128864   |
| Hunan Province Huayuan County     | 109.4885618    | 28.57790993   |
| Guizhou Province Guiyang Qingzhen City | 106.4775226   | 26.5619879    |
| Guangxi Zhuang Autonomous Region Tianyang County | 108.3345212  | 22.821269    |
| Yunnan Province Lianghe County     | 98.30313363    | 24.81078446   |
| Yunnan Province Simao City         | 100.9835551    | 22.79249798   |
| ...                           | ...             | ...            |


>The method of obtaining latitude and longitude has been adapted according to [StimuMing's approach](https://www.cnblogs.com/fole-del/p/14810401.html).

#### Alternative Methods
Certainly, if high precision is not a requisite or if the quantity of data points is amenable to manual or visual inspection, one might consider employing GPT to generate latitude and longitude coordinates. However, it is advisable to construct a prompt that encourages GPT to strive for maximum accuracy or to reference publicly available data sources. 

Nevertheless, it is important to note that GPT may produce erroneous results with an air of authority when confronted with such geospatial queries.



<!--
#### 替代性方法
当然，如果你需要的精度不高，或者数据数量是肉眼或手动可以检视的，你也可以尝试使用GPT来返回经纬度数据。不过prompt中可能需要GPT尽可能精确或参照网上公开数据。但是，GPT对于这种问题可能会一本正经地犯错。
获取经纬度的方式根据StimuMing的方法改编。
-->


# Creating a Heatmap
Creating a heatmap can be effectively accomplished using the [pyecharts](https://github.com/pyecharts/pyecharts) library. Pyecharts is a powerful visualization tool that leverages the capabilities of the ECharts framework, which is developed by Baidu. This library allows for a wide array of data visualizations, including the creation of heatmaps, which are particularly useful for representing the intensity of data points across a two-dimensional plane.



{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/heatmap2.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/heatmap2.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

Now, you have acquired a distribution heat map of intangible cultural heritage (ICH) elements. This visualization reveals a pattern that demonstrates a notable correlation with economic development levels—exhibiting a pronounced east-west gradient and a south-north disparity. The Beijing-Tianjin-Hebei region and the Yangtze River Delta (encompassing Jiangsu, Zhejiang, and Shanghai) unequivocally emerge as hotspots of ICH concentration.

Concomitantly, areas with significant ethnic minority populations, such as Guizhou, also manifest as regions of high ICH density. This phenomenon warrants further investigation into the interplay between cultural diversity, economic development, and the preservation of intangible cultural assets.

The observed distribution pattern raises several pertinent questions for investigations, data journalism, and even academic research:

1. To what extent does economic prosperity facilitate or impede the preservation and documentation of ICH?
2. How do urbanization and modernization processes influence the spatial distribution of ICH elements?
3. What role do ethnic minority communities play in maintaining and transmitting intangible cultural practices, particularly in economically less developed regions?
4. Are there policy implications to be drawn from this spatial distribution, particularly regarding the allocation of resources for ICH preservation and promotion?
5. How might this distribution pattern evolve over time, and what are the potential implications for cultural sustainability and national identity?

This spatial analysis provides a foundation for more nuanced investigations into the complex relationships between geography, economics, ethnicity, and cultural heritage in the Chinese context.

