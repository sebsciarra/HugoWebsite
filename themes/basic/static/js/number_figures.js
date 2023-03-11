

// FIGURE MODIFICATIONS
// figure labels to modify
const fig_caption_tags = document.querySelectorAll('div.figure span.figLabel');
// identify all p element that referece a table (i.e., contain Table \\ref\{tab:)
const potential_fig_references = document.querySelectorAll('p, tfoot, caption, .figNote');
const figure_refs = [];


for (const ref of potential_fig_references) {
  if (ref.innerHTML.includes('Figure \\ref\{fig:')) {
    figure_refs.push(ref);
  }
}

const figure_nums_list = {};

// fix all table titles to be of the format Table i <br> table_title
for (let i = 0; i < fig_caption_tags.length; i++) {

  // identify string to replace (i.e., matching string) by extracting strings of the format `(\#tab:` followed by any number of   characters until a `)` is reached. ([^)]+)\)/
  const match = fig_caption_tags[i].innerHTML.match(/\\ref{fig:[^}]+}/g);
  const fig_label = fig_caption_tags[i].innerHTML.match(/fig:(.*?)}/);

  //add table tag to list
  //table_nums.set(match[0], i+ 1);
  figure_nums_list[fig_label[1]]= i+ 1;

  // replace each match of these instances with <span id="tab:table_name">Table i: </span> --> makes tables have a corresponding   label with number(e.g., Table 1)
  fig_caption_tags[i].innerHTML = fig_caption_tags[i].innerHTML.replace(match, '<span id="fig:'+ fig_label[1] + '">' + (i+1) + '</span>');

}



  // for each label, iterate through all identified <p> elements containing a table reference and replace the latex reference with an HTML //reference that includes the table number that corresponds to that table label
for (let i = 0; i < Object.keys(fig_caption_tags).length; i++) {

  //extract the table label
  figure_label = Object.keys(figure_nums_list)[i];
  figure_number = Object.values(figure_nums_list)[i];


 for (let j = 0; j < figure_refs.length; j++){

    //get figure label from the rth <p> element that contains a table reference and extract the specific label
    label_to_match = new RegExp('\\\\ref\\{fig:' + figure_label + '\\}', 'g')

   // replace original latex table reference (\ref{tab:table_name}) with html reference
   text = figure_refs[j].innerHTML.replace(label_to_match, '<a href="#fig:' + figure_label + '">' + figure_number + '</a>');

   figure_refs[j].innerHTML = text;
  }

}

//set width of figLabel, figTitle, and figNote to width of <img> element
const figures = document.querySelectorAll('div.figure');
for (let f = 0; f < figures.length; f++) {

  const img = figures[f].querySelector('.figure img');

  figures[f].querySelector('.figDivLabel').style.width = img.clientWidth + 'px';

  figures[f].querySelector('.figTitle').style.width =  img.clientWidth + 'px';
  figures[f].querySelector('.figNote').style.width =  img.clientWidth + 'px';

}



