---
title: "{{ replace .Name "-" " " | title }}"
summary:  "This is a summary of the post that briefly summarizes the post"
draft: false
article_type: technical
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
bibFile: biblio.json    
tags: []
---

This is some content.
