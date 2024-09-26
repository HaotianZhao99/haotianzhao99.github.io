---
layout: page
title: Proportional Symbol Charts in ggplot2
description: Using R's ggplot2, we create a Proportional Symbol Chart to illustrate ICH policy patterns across Chinese provinces over time, offering detailed control for enhanced data representation.
img: assets/img/project/plot.png
importance: 2
category: work
related_publications: false
---
The Proportional Symbol Chart offers a quick overview of data distribution by comparing the sizes of symbols. These symbols, typically circles or squares, are proportional in size to the values they represent. This visualization technique can be effectively combined with maps to illustrate geographical distributions, as well as showcase horizontal and vertical data distribution patterns.

In a data journalism project, we aimed to analyze the distribution of policies related to Intangible Cultural Heritage (ICH) across various provinces in China. Our focus was on several aspects: identifying years when individual provinces concentrated on policy implementation, comparing policy enactment across provinces during the same period, and highlighting specific time points when multiple provinces simultaneously introduced numerous policies. For this purpose, we chose the Proportional Symbol Chart as our primary visualization strategy.

Although modern visualization platforms like [Flourish](https://flourish.studio/) can generate Proportional Symbol Charts with simple operations, we opted to use the ggplot2 package in R. This choice was motivated by ggplot2's capacity to offer more refined control, allowing us to make flexible and precise adjustments to the visualization based on our specific requirements.

<!--
Proportional Symbol Chart 可以通过比较比例的大小，来对数据的分布情况有个快速的概览。符号通常是圆形或者是正方形，大小与数据所代表的值成正比。这种技术可以通过与地图结合，实现可视化地理分布，也可以数据的横向和纵向分布情况。

在一项数据新闻中，我们想要知道中国各个省份出台有关非遗的政策的分布情况：某个省份从历时性角度来说哪一年集中出台政策；同期各个省份出台政策的对比；某个时间节点上各个省份同时出台大量政策。在此，我们选择了Proportional Symbol Chart作为可视化策略。

尽管现有平台，比如Flourish已经可以通过简单操作就绘制处Proportional Symbol Chart，我们在此依然选择了R语言中的ggplot2，因为这样可以对可视化效果实现更好精细地调整。
-->

# Data Collection and Cleaning
Policy data related to Intangible Cultural Heritage (ICH) was primarily sourced from the [China Intangible Cultural Heritage Network](https://www.ihchina.cn/zhengce). This website has compiled various national, ministerial, and local regulatory documents concerning ICH since 2000. We used web scraping techniques to collect these regulatory documents. Some of the entries are as follows:


|  Document Title   | Year  |
|  ----  | ----  |
| Notice of the People's Government of Tibet Autonomous Region on Announcing the Sixth Batch of Representative Projects of Intangible Cultural Heritage at the Autonomous Region Level  | 2024 |
| Notice of the People's Government of Ningxia Hui Autonomous Region on Announcing the Seventh Batch of Representative Projects of Intangible Cultural Heritage at the Autonomous Region Level   | 2024 |
| Notice of Shanghai Municipal People's Government on Announcing the Seventh Batch of Shanghai Municipal Intangible Cultural Heritage Representative Project List and the Extended Project List  | 2024 |
| ...  | ...|


After analyzing the data, we found that geographical information is typically contained within the first few characters of the document titles. By extracting the names of provinces, autonomous regions, or municipalities (or simply extracting the first two Chinese characters), we were able to create a streamlined dataset containing provinces and years:


|  Region   | Year  |
|  ----  | ----  |
| Tibet | 2024 |
| Ningxia  | 2024 |
| Shanghai  | 2024 |
| ...  | ...|

<!--
# 数据获取与清洗
有关非遗的政策数据来源于中国非遗网，其中收录了自2000年以来的国家级、部级和地方级的各种有关非遗的国内法规文件。爬取的法规文件的条目如下：

|  文件   | 时间  |
|  ----  | ----  |
| 西藏自治区人民政府关公布第六批自治区级非物质文化遗产代表性项目名录的通知（藏政函 〔2024〕40号）  | 2024 |
| 宁夏回族自治区人民政府关于公布第七批自治区级非物质文化遗产代表性项目名录的通知（宁政发〔2024〕16号  | 2024 |
| 上海市人民政府关于公布第七批上海市非物质文化遗产代表性项目名录和上海市非物质文化遗产代表性项目名录扩展项目名录的通知（沪府发〔2024〕4号）  | 2024 |
| ...  | ...|

观察数据后发现，地理信息一般都包含在文件开头的几个字内，以省、自治区或市为分隔符后（或者更简单只保留前两个汉字），我们可以获得省份和年份的数据集。
|  文件   | 时间  |
|  ----  | ----  |
| 西藏 | 2024 |
| 宁夏  | 2024 |
| 上海  | 2024 |
| ...  | ...|
-->

# Visualization with ggplot2

Firstly, we need to tally the number of regulatory documents enacted by each province every year.
```r
library(readxl)
library(dplyr)
library(ggplot2)

data <- read_xlsx("data.xlsx")
data <- data %>%
  group_by(year, province) %>%
  summarise(count = n(), .groups = 'drop')
```
Therefore, we have generated data on the number of regulations enacted each year and by each province. Next, we can proceed to create a Proportional Symbol Chart.

```r
ggplot(datap, aes(x = date, y = province, size = count)) +
  geom_point() +
  theme_minimal() +
  labs(x = "Year", y = "Province")
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project/project2/proplot1.png" title="proportional symbol chart" class="img-fluid rounded z-depth-1" style="width: 50%; height: auto;" %}
    </div>
</div>

We can further enhance the appearance of the image by adjusting the parameters.

```r
#Convert to a factor to adjust the size of the points
datap$count <- as.factor(datap$count) 
datap$col <- factor(datap$count, level = 1:9, ordered = TRUE)#Divide into 9 levels, one level for one regulation

plot <- ggplot(datap, aes(x = year, y = province, size = col)) +
  geom_point(color = "#87481f", alpha = 0.5) +  # Set the color of all points to the specified color.
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),     #Adjust the angle of the x-axis labels.
    panel.background = element_blank(),  # Remove the background color.
    panel.grid.major.y = element_line(color = "#faf5f0"),  # Change the color of the major horizontal grid lines.
    panel.grid.minor.y = element_line(color = "#faf5f0"),  # Change the color of the minor horizontal grid lines.
    panel.grid.major.x = element_blank(),  # Remove the major vertical grid lines.
    panel.grid.minor.x = element_blank()   # Remove the minor vertical grid lines.
  ) +
  scale_size_manual(values = c(3, 3.5, 5, 6, 7, 8, 10, 12)) +
  labs(x = " ", y = " ", title = " ") +
  guides(size = guide_legend(title = " "))

print(plot)
```
Finally, the plotted image looks as follows.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project/project2/proplot2.png" title="proportional symbol chart" class="img-fluid rounded z-depth-1" style="width: 50%; height: auto;" %}
    </div>
</div>

It looks a bit messy. Let's place the National(国家级) and Ministerial(省部级) levels at the bottom, and arrange the other provinces in order to form the following image.

```r
datap$province <- factor(datap$province, levels = c("国家级", "部级", setdiff(unique(datap$省份), c("国家级", "部级"))))

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/project/project2/proplot3.png" title="proportional symbol chart" class="img-fluid rounded z-depth-1" style="width: 50%; height: auto;" %}
    </div>
</div>
