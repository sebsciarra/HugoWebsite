---
title: "About"
draft: false
output:
  bookdown::html_document2:
     keep_md: true
always_allow_html: true
tags: []
---   




<div class="figure">
  <div class="figDivLabel">
    <caption>
     <span class = 'figLabelAbout'><span> 
    </caption>
  </div>
   <div class="figTitle">
  </div>
    <img src="images/about_picture.jpg" width="70%" height="70%"> 
  
  <div class="figNote">
      <span><em>Note. </em> Generated using a variety of artificial intelligence tools for image processing and image generation (specifically, <a href="https://www.fotor.com">fotor</a>, <a href="https://openai.com/dall-e-2/">DALL-E2</a>, and <a href="https://deepdreamgenerator.com">Deep Dream Generator)</a></span> 
  </div>
</div>

I'm Sebastian Sciarra and I am currently a PhD candidate in the Industrial-Organizational Psychology program at the University of Guelph. I am largely interested in statistics, coding, and machine learning, and have used my time as a graduate student learning a variety of topics in these areas while completing my dissertation. 


## Organization of Blog

I have set up this blog to contain the following three types of content: 

1) <a href="/technical_content">Technical Content</a>: posts dive into the fundamentals of an analysis and explain the necessary calculus and algebra.
2) <a href="/coding_tricks">Coding Demonstrations & Tricks</a>: posts explain useful pieces of code I have come across or constructed (whether in R, Python, SQL, Javascript, CSS, etc.). 
3) <a href="/simulation_exps">Simulation Experiments</a>: posts show how an analysis works by coding the underlying math and/or how an analysis performs under a variety of conditions. 

I have also set up the <a href="/mlresources">ML Resources</a> page as a running list of useful resources I have come across and continue to use.


<script type="text/javascript">
//set width of figLabel, figTitle, and figNote to width of <img> element
const figures = document.querySelectorAll('div.figure');

 for (let f = 0; f < figures.length; f++) {

      const img = figures[f].querySelector('.figure img');

      figures[f].querySelector('.figDivLabel').style.width = img.clientWidth + 'px';

      figures[f].querySelector('.figTitle').style.width =  img.clientWidth + 'px';
      figures[f].querySelector('.figNote').style.width =  img.clientWidth + 'px';
    }


function myFunction() {
  const screenWidth = window.innerWidth;
  const figures = document.querySelectorAll('div.figure');

  if (screenWidth < 1350) {
    for (let f = 0; f < figures.length; f++) {

      const img = figures[f].querySelector('.figure img');

      figures[f].querySelector('.figDivLabel').style.width = img.clientWidth + 'px';

      figures[f].querySelector('.figTitle').style.width =  img.clientWidth + 'px';
      figures[f].querySelector('.figNote').style.width =  img.clientWidth + 'px';
    }
  }
  if (screenWidth < 750) {
     for (let f = 0; f < figures.length; f++) {

      const img = figures[f].querySelector('.figure img');

      figures[f].querySelector('.figDivLabel').style.width = img.clientWidth + 'px';
        figures[f].querySelector('.figLabel').style.width = img.clientWidth + 'px';

      figures[f].querySelector('.figTitle').style.width =  img.clientWidth + 'px';
      figures[f].querySelector('.figNote').style.width =  img.clientWidth + 'px';
     }
  }
}

window.addEventListener('resize', myFunction);
</script>
