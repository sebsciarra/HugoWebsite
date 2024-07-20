

// Animation numbering
// Animation labels to modify
const anim_caption_tags = document.querySelectorAll('div.animation span.animLabel');

// identify all p element that reference an animation (i.e., contain Animation \\ref\{anim:)
const potential_anim_references = document.querySelectorAll('p, tfoot, caption, .figNote, .animNote');
const anim_refs = [];


for (const ref of potential_anim_references) {
  if (ref.innerHTML.includes('Animation \\ref\{anim:')) {
    anim_refs.push(ref);
  }
}

const anim_nums_list = {};

// fix all animation titles to be of the format Animation i <br> table_title
for (let i = 0; i < anim_caption_tags.length; i++) {

  // identify string to replace (i.e., matching string) by extracting strings of the format `(\#anim:` followed by any number of
  // characters until a `)` is reached. ([^)]+)\)/
  const match = anim_caption_tags[i].innerHTML.match(/\\ref{anim:[^}]+}/g);
  const anim_label = anim_caption_tags[i].innerHTML.match(/anim:(.*?)}/);

  //add animation tag to list
  anim_nums_list[anim_label[1]]= i+ 1;

  // replace each match of these instances with <span id="anim:anim_label">Table i: </span> --> makes animation have a corresponding
  // label with number (e.g., Animation 1)
  anim_caption_tags[i].innerHTML = anim_caption_tags[i].innerHTML.replace(match, '<span id="anim:'+ anim_label[1] + '">' +"&nbsp;" + (i+1) + '</span>');

}



// for each label, iterate through all identified <p> elements containing an animation reference and replace the latex reference with an HTML        // reference that includes the animation number that corresponds to that animation label
for (let i = 0; i < Object.keys(anim_caption_tags).length; i++) {

  //extract the animation label
  anim_label = Object.keys(anim_nums_list)[i];
  anim_number = Object.values(anim_nums_list)[i];


 for (let j = 0; j < anim_refs.length; j++){

    //get animation label from the rth <p> element that contains an animation reference and extract the specific label
    label_to_match = new RegExp('\\\\ref\\{anim:' + anim_label + '\\}', 'g')

   // replace original latex animation reference (\ref{anim:anim_name}) with html reference
   text = anim_refs[j].innerHTML.replace(label_to_match, '<a href="#anim:' + anim_label + '">' + anim_number + '</a>');

   anim_refs[j].innerHTML = text;
  }

}

//set width of animLabel, animTitle, and animNote to width of <video> element
const animation = document.querySelectorAll('div.animation');
 for (let f = 0; f < animation.length; f++) {

      const video = animation[f].querySelector('.animation video');

      animation[f].querySelector('.animDivLabel').style.width = video.clientWidth + 'px';
      animation[f].querySelector('.animLabel').style.width = video.clientWidth + 'px';

      animation[f].querySelector('.animTitle').style.width =  video.clientWidth + 'px';
      animation[f].querySelector('.animNote').style.width =  video.clientWidth + 'px';
    }


function myFunction() {
  const screenWidth = window.innerWidth;
  const animation = document.querySelectorAll('div.animation');

  if (screenWidth < 1350) {
    for (let f = 0; f < animation.length; f++) {

      const video = animation[f].querySelector('.figure video');

      // animation[f].querySelector('.animDivLabel').style.width = video.clientWidth + 'px';
      animation[f].querySelector('.animLabel').style.width = video.clientWidth + 'px';

      animation[f].querySelector('.animTitle').style.width =  video.clientWidth + 'px';
      animation[f].querySelector('.animNote').style.width =  video.clientWidth + 'px';
    }
  }
  if (screenWidth < 750) {
     for (let f = 0; f < animation.length; f++) {

      const video = animation[f].querySelector('.anim video');

      // animation[f].querySelector('.animDivLabel').style.width = video.clientWidth + 'px';
      animation[f].querySelector('.animLabel').style.width = video.clientWidth + 'px';

      animation[f].querySelector('.animTitle').style.width =  video.clientWidth + 'px';
      animation[f].querySelector('.animNote').style.width =  video.clientWidth + 'px';
     }
  }
}

window.addEventListener('resize', myFunction);

