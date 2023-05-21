//set class name of appendix h1 elements to 'appendix'
const h1Elements = document.querySelectorAll('h1');

for (let i = 0; i < h1Elements.length; i++) {
  const h1Element = h1Elements[i];
  if (h1Element.innerHTML.startsWith('Appendix')) {
     h1Element.classList.add('appendix');
  }
  else if (h1Element.innerHTML.startsWith('References')) {
    h1Element.classList.add('references');
  }
}
// Get all h1 elements on the page
const blog_sections = document.querySelectorAll('h1:not(.blog_name):not(.appendix):not(.references)');

for (let i = 0; i < h1Elements.length; i++) {
  const h1Element = h1Elements[i];
  const spanElement = document.createElement('span');
  spanElement.style.display = 'none';
  spanElement.innerHTML = '\\(\\nextSection\\)';
  h1Element.insertAdjacentElement('afterend', spanElement);
}




// Get the <h1> element with innerText that starts with 'Appendix A'
const appendixHeader = document.querySelector("h1.appendix");

if (appendixHeader && appendixHeader.innerText.startsWith("Appendix A")) {

  const div = document.createElement("div");
  div.style.display = "none";
  div.innerHTML = "\\(\\setSection{A}\\)";
  appendixHeader.after(div);

}



