
// EXTERNAL LINKS OPEN NEW TAB
const external_links = document.querySelectorAll('a[href^="https"], a[href^="bit"]');  //a[href^="/"]
external_links.forEach(link => link.setAttribute("target", "_blank"))
external_links.forEach(link => link.setAttribute("rel", "noopener noreferrer")) //prevents malicious use of links
