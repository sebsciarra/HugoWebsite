
// Get the table by its id
var table = document.getElementById("myTable");
var codeChunk = document.querySelectorAll(".highlight-wrapper, [class*='-code']:not(.fa-code)");


// Get the button by its id
var button = document.getElementById("collapseButton");
var isCollapsed = true;


// Save the contents of first and second columns before collapsing the table
var firstColContent = [];
var secondColContent = [];

for (var i = 0; i < table.rows.length; i++) {
  firstColContent.push(table.rows[i].cells[0].innerHTML);
  secondColContent.push(table.rows[i].cells[1].innerHTML);
}




// Save the original width and background color of the second column
var secondCol = document.querySelectorAll(".second-col");
var secondColWidth;
var secondColBackground;

for (var i = 0; i < secondCol.length; i++) {
  secondColWidth = secondCol[i].offsetWidth;
  secondColBackground = window.getComputedStyle(secondCol[i]).getPropertyValue("background-color");
}



// Add event listener to the button
button.addEventListener("click", function() {

  if (isCollapsed) {

    var firstRow = table.rows[0].cells[0].innerHTML;
var lastRow = table.rows[table.rows.length-1].cells[0].innerHTML;

table.rows[0].cells[0].innerHTML = firstRow + "–" + lastRow;


    //for each row in the table
    for (var i = 0; i < table.rows.length; i++) {

      //table.rows[0].cells[0].innerHTML = firstRow + " – " + lastR
      table.rows[i].cells[1].innerHTML = ""; //replace all values of second column with an emptystring
      table.rows[i].cells[1].style.width = secondColWidth + "px"; //set width of second column to original width
      table.rows[i].cells[1].style.backgroundColor = secondColBackground;//set background color of second column to original color
    }

    // replace contents of entire second row with nothing
    for (var j = 1; j < table.rows.length; j++) {
        table.rows[j].style.display = "none";
    }

        //firstRow.style.display = "";
        //lastRow.style.display = "none";
        button.innerHTML = "Expand";
        isCollapsed = false;
    }

        else {

        table.rows[0].cells[0].innerHTML = '';

    // Expand the table and show the contents of the second column again
        for (var t = 0; t < table.rows.length; t++) {
            table.rows[t].cells[1].innerHTML = secondColContent[t];
            table.rows[t].cells[0].innerHTML = firstColContent[t];
            table.rows[t].style.display = "";
        }

        button.innerHTML = "Hide";
        isCollapsed = true;
    }
  });







