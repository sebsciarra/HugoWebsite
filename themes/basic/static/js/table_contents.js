
const contentDiv = document.querySelector('.content');
const section_titles = contentDiv.querySelectorAll('h1, h2, h3, h4, h5, h6');

const tocList = document.createElement('ul');

//create table of contents
for (let i = 0; i < section_titles.length; i++) {

  const heading = section_titles[i];

  //extract id attribute of heading so it can later be inputted into the href attribute of each <a> element
  heading_id = heading.getAttribute('id');

  const listItem = document.createElement('li');
  const link = document.createElement('a');
  link.setAttribute('href', '#' + heading_id); //allows ToC elements to act as hyperlinks

   link.setAttribute('header-level', heading.tagName.toLowerCase());

  link.textContent = heading.textContent;
  listItem.appendChild(link);
  tocList.appendChild(listItem);
}

const tocContainer = document.querySelector('.tableContents');
tocContainer.appendChild(tocList);



// Get all the table of contents links and their corresponding sections
const links = document.querySelectorAll('.tableContents a');
const sections = document.querySelectorAll('.content h1[id], .content h2[id], .content h3[id], .content h4[id], .content h5[id], .content h6[id]');



// Add click event listeners to the table of contents links
for (let i = 0; i < links.length; i++) {

  links[i].addEventListener('click', function (event) {
    event.preventDefault(); // Prevent the default link behavior
    const targetId = this.getAttribute('href'); // Get the is of current link
    const targetSection = document.querySelector(targetId); // Get the target section element
    const targetOffset = targetSection.offsetTop; // Get the target section offset top

    window.scrollTo({
      top: targetOffset,
      behavior: 'smooth' // Scroll smoothly to the target section
    });
  });
}

// Add scroll event listener to the window to highlight the active table of contents link
window.addEventListener('scroll', function () {
  let currentSection = '';
  sections.forEach(section => {
    const sectionTop = section.offsetTop;
    const sectionHeight = section.offsetHeight;
    if (pageYOffset >= sectionTop - sectionHeight / 3) {
      currentSection = section.getAttribute('id');
    }
  });
  links.forEach(link => {
    link.classList.remove('active');
    const href = link.getAttribute('href');
    if (href === `#${currentSection}`) {
      link.classList.add('active');
    }
  });
});






