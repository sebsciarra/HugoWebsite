
codeBlocks = document.querySelectorAll('.highlight, pre:has(code[class*="-code"])');

var cumulative_length = 0;
var starting_indices = [0];


for (block = 0; block < codeBlocks.length; block++){

  if(codeBlocks[block].outerHTML.startsWith('<pre><code class=')){

    var code = codeBlocks[block].outerHTML;
    var lines = code.split("\n");
    //extract preamble code that will wrap table (<pre><code> elements)
    var preamble = lines[0].match(/^[^>]*>([^>]*>)/)[0];

   ////remove preamble code from first element of lines and add copy button to the endof this element
   lines[0] = lines[0].replace(preamble, ''); //+ '<button class="copy-button" data-button="Copy"></button>';
   table_end = lines[lines.length - 1]; //save last element
   lines.splice(lines.length - 1, 1); //delete last element
  } else {

     var code = codeBlocks[block].innerHTML;
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

  //add copy button
  // only add button if browser supports Clipboard API
   codeTable.rows[0].cells[1].innerHTML +=  '<button class ="copy-code-button" data-button = "Copy"></button>';

  //compile complete table
  complete_codeTable = preamble + codeTable.outerHTML + table_end;

   if(codeBlocks[block].outerHTML.startsWith('<pre><code class=')){
      codeBlocks[block].outerHTML = complete_codeTable;
   } else{
       codeBlocks[block].innerHTML = complete_codeTable;
   }

}




codeTables = document.querySelectorAll('#codeTable');

codeTables.forEach(function(codeTable){

  // Get the button by its id
  var isCollapsed = true;

  // Save the contents of first and second columns before collapsing the table
  var firstColContent = [];
  var secondColContent = [];

  for (var i = 0; i < codeTable.rows.length; i++) {
    firstColContent.push(codeTable.rows[i].cells[0].innerHTML);
    secondColContent.push(codeTable.rows[i].cells[1].innerHTML);
}



  var button = codeTable.querySelector("#collapseButton");

  // Add event listener to the button
  button.addEventListener("click", function() {

    if (isCollapsed) {

    //collapse first column such that line number range is shown
    var first_row_num = codeTable.rows[0].cells[0].innerHTML;
    var last_row_num = codeTable.rows[codeTable.rows.length-1].cells[0].innerHTML;

    codeTable.rows[0].cells[0].innerHTML = first_row_num + "–" + last_row_num;

      for (var x = 0; x < codeTable.length; x++) {

        //codeTable.rows[0].cells[0].innerHTML = firstRow + "–" + last_row_num;
        codeTable.rows[x].cells[1].innerHTML = ""; //replace all values of second column with an emptys tring
        codeTable.rows[x].cells[1].style.width = secondColWidth + "px"; //set width of second column to original width
      }


    // replace contents of entire table begging at the second second row with nothing
    for (var j = 1; j < codeTable.rows.length; j++) {
        codeTable.rows[j].style.display = "none";
    }

    //for the first row of the second column, replace contents with empty line
    codeTable.rows[0].cells[1].innerHTML = '</span></span><span class="line"><span class="cl"><br>';
    codeTable.rows[0].style.height = '20px';

         button.setAttribute("data-button",  'Expand');
        isCollapsed = false;
    }

        else {
         codeTable.rows[0].style.height = '14px';
        codeTable.rows[0].cells[0].innerHTML = '';

    // Expand the table and show the contents of the second column again
        for (var t = 0; t < codeTable.rows.length; t++) {
            codeTable.rows[t].cells[1].innerHTML = secondColContent[t];
            codeTable.rows[t].cells[0].innerHTML = firstColContent[t];
            codeTable.rows[t].style.display = "";
        }

        button.setAttribute("data-button",  'Hide');
        isCollapsed = true;
    }
});



});

   // Defining a custom filter function
//function myFilter(elm){
//    return ( elm !== "");
//}


const cope_buttons = document.querySelectorAll(".copy-code-button");
cope_buttons.forEach(function(copyBtn) {
  copyBtn.addEventListener("click", function(event) {

    navigator.clipboard.writeText('');  //refresh clipboard to eliminate overwriting from cache or local storage

    //copy lines of code line by line so that unnecesary \t and "" elements are not added in output (occurs in Google Chrome)
    const codeTable = copyBtn.closest('table');

    let table_code = '';

    for (let row = 0; row < codeTable.rows.length; row++) {

        table_code += codeTable.rows[row].cells[1].textContent + '\n';
      }


    //let cleanCode = table_text.replace(/(\r\n|\n|\r)/gm, "");

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



