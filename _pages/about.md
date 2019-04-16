---
layout: archive
title: "About"
permalink: /about/
author_profile: True
header:
  image: "/images/redbanner.jpg"
---

I am an ambitious technical professional with a keen interest in Data Science.             

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
