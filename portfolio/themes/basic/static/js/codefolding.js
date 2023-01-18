
window.initializeCodeFolding = function(show) {

  // handlers for show-all and hide all
  $("#rmd-show-all-code").click(function() {
    $('div.r-code-collapse').each(function() {
      $(this).collapse('show');
    });
  });
  $("#rmd-hide-all-code").click(function() {
    $('div.r-code-collapse').each(function() {
      $(this).collapse('hide');
    });
  });

  // index for unique code element ids
  var currentIndex = 1;

  // select all R code blocks
 var rCodeBlocks = $('div.highlight, code.r-code');
  rCodeBlocks.each(function() {

    // create a collapsable div to wrap the code in
    var div = $('<div class="collapse r-code-collapse"></div>');

    var id = 'rcode-643E0F36' + currentIndex++;
    div.attr('id', id);
    $(this).before(div);
    $(this).detach().appendTo(div);

    if (show) {
      $('div.r-code-collapse').each(function() {
        $(this).collapse('show');
      });
    }

    // add a show code button right above
    var showCodeText = $('<span>' + (show ? 'Hide' : 'Expand') + '</span>');
    var showCodeButton = $('<button type="button" class="btn btn-light btn-sm code-folding-btn pull-right"></button>');
    showCodeButton.append(showCodeText);
    showCodeButton
        .attr('data-toggle', 'collapse')
        .attr('data-target', '#' + id)
        .attr('aria-expanded', show)
        .attr('aria-controls', id);

    var buttonRow = $('<div class="row"></div>');
    var buttonCol = $('<div class="col-md-12"></div>');

    buttonCol.append(showCodeButton);
    buttonRow.append(buttonCol);






    //accord_buttons.forEach((btn) => {
    //  btn.addEventListener('click', ()=> {
    //
    //    const panel = btn.nextElementSibling; // selects element that followers the button (in this case, the body)
    //    panel.classList.toggle('active');
    //    btn.classList.toggle('active');
    //    });
    //});



    div.before(buttonRow);

    // update state of button on show/hide
    div.on('hidden.bs.collapse', function () {
      showCodeText.text('Expand');
    });
    div.on('show.bs.collapse', function () {
      showCodeText.text('Hide');
    });
  });

}

  // Get the table by its id
  var table = document.getElementById("myTable");
  var firstRow = table.rows[0];
  var lastRow = table.rows[table.rows.length-1];
  // Hide all rows except the first and last rows of the first column
  for (var i = 1; i < table.rows.length - 1; i++) {
    table.rows[i].style.display = "none";
  }
  firstRow.style.display = "";
  lastRow.style.display = "";


