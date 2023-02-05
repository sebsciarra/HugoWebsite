const cope_buttons = document.querySelectorAll(".copy-code-button");
cope_buttons.forEach(function(copyBtn) {
  copyBtn.addEventListener("click", function(event) {

    const table_lines = copyBtn.closest('table').innerText.split('\n');

    //remove second line because it is an empty line character that takes place of the button
    table_lines.splice(1, 1);
    table_text = table_lines.join('\n');

    var originalText = copyBtn.dataset.button;

    copyBtn.dataset.button = "Copied!";

    setTimeout(function() {
    copyBtn.dataset.button = originalText;
  }, 750);

    navigator.clipboard.writeText(table_text).then(function() {
      console.log("Copied to clipboard");

    });
  });
});
async function copyCodeToClipboard(button, highlightDiv) {
  const codeToCopy = highlightDiv.querySelector(":last-child > .chroma > code").innerText;
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
//  highlightDiv.querySelector('table').rows[0].cells[1].innerHTML += '<div class = "highlight-wrapper"><button class="copy-code-button" type="button">Copy//</button> </div>';
}

document.querySelectorAll(".highlight")
  .forEach(highlightDiv => createCopyButton(highlightDiv));

