

codeBlocks = document.querySelectorAll(".highlight");

const code = codeBlocks[1].innerHTML;
const lines = code.split("\n");

//extract preamble code that will wrap table (<pre><code> elements)
const preamble = lines[0].match(/^[^>]*>([^>]*>)/)[0];

//remove preamble code from first element of lines and add copy button to the endof this element
lines[0] = lines[0].replace(preamble, '') + '<button class="copy-button" data-button="Copy"></button>';
table_end = lines[lines.length - 1]; //save last element
lines.splice(lines.length - 1, 1); //delete last element

//make sure empty lines have <br> element so that empty lines are included whenever code is copied
for (br = 0; br < lines.length; br++) {
    if(lines[br].includes('</span></span><span class="line"><span class="cl">')){
       lines[br] = lines[br] + "<br>";
    }
}

let codeTable = document.createElement("table");
codeTable.setAttribute('id', "codeTable");

//add rows to table by adding each element of lines
for (let t = 0; t < lines.length; t++) {
  let row = codeTable.insertRow(-1);

  let newCell1 = row.insertCell(0); //insert line number
  let newCell2 = row.insertCell(1);
  let newCell3 = row.insertCell(2);

  newCell1.innerHTML = "<span class= 'line-number' data-number='" + (t+1)  + "'" + "id = '" + (t+1) + "'></span>";
  newCell2.innerHTML = lines[t];
  newCell3.innerHTML = "";
}

//add hide/Expand button
codeTable.rows[0].cells[2].innerHTML = '<button id ="collapseButton" data-button = "Hide"></button>';

//compile complete table
complete_codeTable = preamble + codeTable.outerHTML + table_end;
document.querySelectorAll(".highlight")[1].outerHTML = complete_codeTable;


// Get the table by its id
var table = document.getElementById("codeTable");
//var codeChunk = document.querySelectorAll(".highlight-wrapper, [class*='-code']:not(.fa-code)");

// Get the button by its id
var isCollapsed = true;

// Save the contents of first and second columns before collapsing the table
var firstColContent = [];
var secondColContent = [];

for (var i = 0; i < table.rows.length; i++) {
  firstColContent.push(table.rows[i].cells[0].innerHTML);
  secondColContent.push(table.rows[i].cells[1].innerHTML);
}



var button = document.getElementById("collapseButton");

console.log(button.getAttribute("data-button"));
// Add event listener to the button
button.addEventListener("click", function() {

  if (isCollapsed) {

  //collapse first column such that line number range is shown
  var first_row_num = table.rows[0].cells[0].innerHTML;
  var last_row_num = table.rows[table.rows.length-1].cells[0].innerHTML;

  table.rows[0].cells[0].innerHTML = first_row_num + "–" + last_row_num;

    for (var x = 0; x < table.length; x++) {

      //table.rows[0].cells[0].innerHTML = firstRow + " – " + lastR
      table.rows[x].cells[1].innerHTML = ""; //replace all values of second column with an emptys tring
      table.rows[x].cells[1].style.width = secondColWidth + "px"; //set width of second column to original width
    }


    // replace contents of entire table begging at the second second row with nothing
    for (var j = 1; j < table.rows.length; j++) {
        table.rows[j].style.display = "none";
    }

    //for the first row of the second column, replace contents with empty line
    table.rows[0].cells[1].innerHTML = '</span></span><span class="line"><span class="cl"><br>';
    table.rows[0].style.height = '20px';

         button.setAttribute("data-button",  'Expand');
        isCollapsed = false;
    }

        else {
         table.rows[0].style.height = '14px';
        table.rows[0].cells[0].innerHTML = '';

    // Expand the table and show the contents of the second column again
        for (var t = 0; t < table.rows.length; t++) {
            table.rows[t].cells[1].innerHTML = secondColContent[t];
            table.rows[t].cells[0].innerHTML = firstColContent[t];
            table.rows[t].style.display = "";
        }

        button.setAttribute("data-button",  'Hide');
        isCollapsed = true;
    }
});








