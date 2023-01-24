
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

        codeTable.rows[0].cells[0].innerHTML = firstRow + " – " + last_row_num
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








//COPY BUTTON
// https://aaronluna.dev/blog/add-copy-button-to-code-blocks-hugo-chroma/
function createCopyButton(highlightDiv) {
  const button = document.createElement("button");
  button.className = "copy-code-button";
  button.type = "button";
  button.innerText = "Copy";
  button.addEventListener("click", () => copyCodeToClipboard(button, highlightDiv));
  addCopyButtonToDom(button, highlightDiv);
}


async function copyCodeToClipboard(button, highlightDiv) {

  // make sure non-breakble characters are not copied
  const codeToCopy = highlightDiv.querySelector(":last-child > .chroma > code").innerText.replace(/\u00A0/g,' ');

  try {
    result = await navigator.permissions.query({ name: "clipboard-write" });
    if (result.state == "granted" || result.state == "prompt") {
      await navigator.clipboard.writeText(codeToCopy);
    } else {
      copyCodeBlockExecCommand(codeToCopy, highlightDiv);
    }
  } catch (_) {
    copyCodeBlockExecCommand(codeToCopy, highlightDiv);
  }
  finally {
    codeWasCopied(button);
  }
}


function copyCodeBlockExecCommand(codeToCopy, highlightDiv) {
  const textArea = document.createElement("textArea");
  textArea.contentEditable = 'true'
  textArea.readOnly = 'false'
  textArea.className = "copyable-text-area";
  textArea.value = codeToCopy;
  highlightDiv.insertBefore(textArea, highlightDiv.firstChild);
  const range = document.createRange()
  range.selectNodeContents(textArea)
  const sel = window.getSelection()
  sel.removeAllRanges()
  sel.addRange(range)
  textArea.setSelectionRange(0, 999999)
  document.execCommand("copy");
  highlightDiv.removeChild(textArea);
}

function codeWasCopied(button) {
  button.blur();
  button.innerText = "Copied!";
  setTimeout(function() {
    button.innerText = "Copy";
  }, 2000);
}

function addCopyButtonToDom(button, highlightDiv) {
  highlightDiv.insertBefore(button, highlightDiv.firstChild);
  const wrapper = document.createElement("div");
  wrapper.className = "highlight-wrapper";
  highlightDiv.parentNode.insertBefore(wrapper, highlightDiv);
  wrapper.appendChild(highlightDiv);
}


document.querySelectorAll(".highlight")
  .forEach(highlightDiv =>  createCopyButton(highlightDiv));

