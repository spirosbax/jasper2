---
layout: page
current: publications
title: Publications
navigation: true
logo: 'assets/images/ghost.png'
class: page-template
subclass: 'post page'
years: [2020]
---

<!-- <p style="text-align: center; line-height: 3em;">
{% capture site_tags %}{% for tag in site.tags %}{{ tag | first }}{% unless forloop.last %},{% endunless %}{% endfor %}{% endcapture %}
{% assign tags = site_tags | split:',' | sort: 'title' %}
{% include tagcloud.html %}
</p> -->

{% for y in page.years %}
  <h3 class="year">{{y}}</h3>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}
