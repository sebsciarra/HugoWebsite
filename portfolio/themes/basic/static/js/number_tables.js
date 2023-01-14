
// TABLE MODIFICATIONS
// find all instances where a <caption> inside tables. Also create a list (Map object) that tracks each unique table label (e.g., (\\#tab:parameterValues)) and assigns its a number (beginnning from 1) and increases by 1 \\)
const caption_tags = document.querySelectorAll('table caption');

//list of table labels and numbers
const table_nums = {};

// fix all table titles to be of the format Table i <br> table_title
for (let i = 0; i < caption_tags.length; i++) {

  // identify string to replace (i.e., matching string) by extracting strings of the format `(\#tab:` followed by any number of   characters until a `)` is reached. ([^)]+)\)/
  const match = caption_tags[i].innerHTML.match(/\(\\#tab:([^)]+)\)/);

  //add table tag to list
  //table_nums.set(match[0], i+ 1);
  table_nums[match[1]] = i+ 1;

  // replace each match of these instances with <span id="tab:table_name">Table i: </span> --> makes tables have a corresponding   label with number(e.g., Table 1)
  caption_tags[i].innerHTML = caption_tags[i].innerHTML.replace(match[0], '<span id="tab:'+ match[1] + '">Table ' + (i+1) + '<br></span>');
}

// identify all p element that referece a table (i.e., contain Table \\ref\{tab:)
const p_elements = document.querySelectorAll('p, tfoot, caption, .figNote');
const table_refs = [];

for (const element of p_elements) {
  if (element.innerHTML.includes('Table \\ref\{tab:')) {
    table_refs.push(element);
  }
}

  // for each label, iterate through all identified <p> elements containing a table reference and replace the latex reference with an HTML reference that includes the table number that corresponds to that table label
for (let i = 0; i < Object.keys(table_nums).length; i++) {

  //extract the table label
  table_label = Object.keys(table_nums)[i];
  table_number = Object.values(table_nums)[i];

 for (let j = 0; j < table_refs.length; j++){

    //get table label from the rth <p> element that contains a table reference and extract the specific label
    label_to_match = new RegExp('\\\\ref\\{tab:' + table_label + '\\}', 'g');

   // replace original latex table reference (\ref{tab:table_name}) with html reference
   text = table_refs[j].innerHTML.replace(label_to_match, '<a href="#tab:' + table_label + '">' + table_number + '</a>');


   table_refs[j].innerHTML = text;
  }
}
