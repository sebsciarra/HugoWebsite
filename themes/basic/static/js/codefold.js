


code_blocks = document.querySelectorAll('.highlight, code[class*="-code"]');

var target_code_blocks = [];


for (code_block = 0; code_block < code_blocks.length; code_block++) {


  if (code_blocks[code_block].className.includes("-code")) {
    const parent = code_blocks[code_block].parentNode;
    target_code_blocks.push(parent);
  }

  else{
       target_code_blocks.push(code_blocks[code_block]);
  }
}


var cumulative_length = 0;
var starting_indices = [0];

for (block = 0; block < target_code_blocks.length; block++){

  if(target_code_blocks[block].outerHTML.startsWith('<pre><code class=')){

    var code = target_code_blocks[block].outerHTML;
    var lines = code.split("\n");
    //extract preamble code that will wrap table (<pre><code> elements)
    var preamble = lines[0].match(/^[^>]*>([^>]*>)/)[0];

   ////remove preamble code from first element of lines and add copy button to the endof this element
   lines[0] = lines[0].replace(preamble, ''); //+ '<button class="copy-button" data-button="Copy"></button>';
   table_end = lines[lines.length - 1]; //save last element
   lines.splice(lines.length - 1, 1); //delete last element
  } else {

     var code = target_code_blocks[block].innerHTML;
     var lines = code.split("\n");

    //extract preamble code that will wrap table (<pre><code> elements)
    var preamble = lines[0].match(/^[^>]*>([^>]*>)/)[0];

    //remove preamble code from first element of lines and add copy button to the endof this element
    lines[0] = lines[0].replace(preamble, '') ;//+ '<button class="copy-code-button" type="button">Copy</button>';
    table_end = lines[lines.length - 1]; //save last element
    lines.splice(lines.length - 1, 1); //delete last element
  }


  //make sure empty lines have <br> element so that empty lines are included whenever code is copied
  for (br = 0; br < lines.length; br++) {
    if(lines[br].includes('</span></span><span class="line"><span class="cl">')){
       lines[br] = lines[br] + "<br>";
    }
  }


  let codeTable = document.createElement("table");
  codeTable.setAttribute('id', "codeTable");

  //starting and cumulative index numbers
  starting_indices.push(lines.length);
  cumulative_length += lines.length;
  starting_index = starting_indices.slice(0, block + 1).reduce((acc, cur) => acc + cur);

  //add rows to table by adding each element of lines
  for (t=starting_index, line_num = 0; t < cumulative_length; t++, line_num++) {

    let row = codeTable.insertRow(-1);

    let newCell1 = row.insertCell(0); //insert line number
    let newCell2 = row.insertCell(1);
    let newCell3 = row.insertCell(2);

    newCell1.innerHTML = "<span class= 'line-number' data-number='" + (t+1)  + "'" + "id = '" + (t+1) + "'></span>";
    newCell2.innerHTML = lines[line_num];
    newCell3.innerHTML = "";
  }


  //add hide/Expand button
  codeTable.rows[0].cells[2].innerHTML = '<button id ="collapseButton" data-button = "Hide"></button>';


   let pattern = /<pre><code class="(.*?)">/;
   let result = pattern.exec(target_code_blocks[block].outerHTML);
  //const classAttribute = result[1];
  if (result) {
    var language = result[1].split('-')[0];

    if(language == 'r'){
      language = "R";
    }

  } else {
       var attributeValue = target_code_blocks[block].attributes.language;
       var language = attributeValue ? attributeValue.value : "R";
    }

  //add copy button along with coding language tag
   codeTable.rows[0].cells[1].innerHTML +=  '<div class="language-box" data-language = "' + language +  '"></div>' + '<button class ="copy-code-button" data-button = "Copy"></button>';


      //add appropriate padding-right
      //tag = child_table.querySelector('.language-box');

      // var afterElement = document.createElement("div");
      //afterElement.innerHTML = tag.dataset.language;
      //afterElement.style.display = "inline-block";
      //afterElement.style.visibility = "hidden";
      //tag.appendChild(afterElement);
      //var tag_language_width = afterElement.offsetWidth;
      //tag.removeChild(afterElement);   //remove child




  //compile complete table
  complete_codeTable = preamble + codeTable.outerHTML + table_end;

   if(target_code_blocks[block].outerHTML.startsWith('<pre><code class=')){
      target_code_blocks[block].outerHTML = complete_codeTable;
   } else{
       target_code_blocks[block].innerHTML = complete_codeTable;
   }

}


tags = document.querySelectorAll('.language-box');

tags.forEach(function(tag){

 //get width of language tag element
 var afterElement = document.createElement("div");
 afterElement.innerHTML = tag.dataset.language;
 afterElement.style.display = "inline-block";
 afterElement.style.visibility = "hidden";
 tag.appendChild(afterElement);
 var tag_language_width = afterElement.offsetWidth;
 tag.removeChild(afterElement);   //remove child


 //set padding right of first row in second column for code blocks
 if (tag.previousSibling && tag.previousSibling.style && tag.parentNode.innerHTML.startsWith('<span')) {
 tag.previousSibling.style.paddingRight = 60 + tag_language_width + "px";
 }
 //set padding right for code output blocks
 else{
  tag.parentNode.style.paddingRight = 60 + tag_language_width + "px";
 }

});


codeTables_2 = document.querySelectorAll('.highlight, pre code[class*="-code"]');

codeTables_2.forEach(function(table){

  //extract parent
  if(table.outerHTML.startsWith('<code class=')) {
     code_table = table.parentNode;
  }
  else{
     code_table = table;
  }

  //extract table
  var child_table = table.querySelector('#codeTable');


  //Save the contents of first and second columns before collapsing the table
  var firstColContent = [];
  var secondColContent = [];

  for (var i = 0; i < child_table.rows.length; i++) {
    firstColContent.push(child_table.rows[i].cells[0].innerHTML);
    secondColContent.push(child_table.rows[i].cells[1].innerHTML);
  }

  //extract coding language for tag
  let pattern = /<pre><code class="(.*?)">/;
  let result = pattern.exec(code_table.outerHTML);


  if (result) {
    var coding_language = result[1].split('-')[0];

    //capitalize 'r'
    if(coding_language == 'r'){
     coding_language = "R";
    }

  }
  else {
  //add language to the code table
  attributeValue = code_table.attributes.language;
   coding_language = attributeValue ? attributeValue.value : "R";
  }


  // Get the button by its id
  var button = child_table.querySelector("#collapseButton");
  var isCollapsed = true;


  button.addEventListener("click", function() {

    if (isCollapsed) {

      //collapse first column such that line number range is shown
      var first_row_num = child_table.rows[0].cells[0].innerHTML;
      var last_row_num = child_table.rows[child_table.rows.length-1].cells[0].innerHTML;

      child_table.rows[0].cells[0].innerHTML = first_row_num + "–" + last_row_num;

        //for (var x = 0; x < child_table.length; x++) {
//
        //  child_table.rows[x].cells[1].innerHTML = ""; //replace all values of second column with an empty tring
        //  child_table.rows[x].cells[1].style.width = secondColWidth + "px"; //set width of second column to original width
        //}

      // replace contents of entire table beggining at the second second row with nothing
      for (var j = 1; j < child_table.rows.length; j++) {
          child_table.rows[j].style.display = "none";
      }


      //for the first row of the second column, replace contents with empty line
          child_table.rows[0].cells[1].innerHTML = '</span></span><span class="line"><span class="cl"><div class="language-box-collapsed" data-language = "'+ coding_language + '"></div><br>';
      child_table.rows[0].style.height = '20px'; //set height of button
      console.log(child_table.rows[0].style.height);
   button.setAttribute("data-button",  'Expand');
   isCollapsed = false;

  //set padding right of first row in second column
  if (tag.previousSibling && tag.previousSibling.style) {
  tag.previousSibling.style.paddingRight = 60 + tag_language_width + "px";
  }
  }
    else {
        child_table.rows[0].style.height = '14px';
        child_table.rows[0].cells[0].innerHTML = '';

    // Expand the table and show the contents of the second column again
       for (var t = 0; t < child_table.rows.length; t++) {
           child_table.rows[t].cells[1].innerHTML = secondColContent[t];
           child_table.rows[t].cells[0].innerHTML = firstColContent[t];
           child_table.rows[t].style.display = "";
       }
//
        button.setAttribute("data-button",  'Hide');
        isCollapsed = true;
    }


  });
});


const copy_buttons = document.querySelectorAll(".copy-code-button");
copy_buttons.forEach(function(copyBtn) {
  copyBtn.addEventListener("click", function(event) {

    navigator.clipboard.writeText('');  //refresh clipboard to eliminate overwriting from cache or local storage

    //copy lines of code line by line so that unnecesary \t and "" elements are not added in output (occurs in Google Chrome)
    const codeTable = copyBtn.closest('table');

    let table_code = '';

    for (let row = 0; row < codeTable.rows.length; row++) {

        table_code += codeTable.rows[row].cells[1].textContent + '\n';
      }

    var originalText = copyBtn.dataset.button;
    copyBtn.dataset.button = "Copied!";

    setTimeout(function() {
    copyBtn.dataset.button = originalText;
  }, 750);

   navigator.clipboard.writeText(table_code).then(function() {
     console.log("Copied to clipboard");
   });

  });
});


