const toggleButton = document.getElementsByClassName('toggle-button')[0];
const navbar = document.getElementsByClassName('navbar')[0];

toggleButton.addEventsListener('click', () => {
  navbar.classList.toggle('active');
}
)

