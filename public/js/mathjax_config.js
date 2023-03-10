
//see https://github.com/mathjax/MathJax/issues/3013
window.MathJax = {
   options: {
    enableMenu: true,          // set to false to disable the menu
    menuOptions: {
      settings: {
        displayOverflow: 'linebreak'
        }
      }
   },
  section: {
    n: -1,
    useLetters: false,
    letters: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  },

  loader: {load: ['[tex]/tagformat', '[tex]/mathtools', 'output/chtml']},
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']], //allow inline math
    displayMath: [['$$','$$']],
    tagSide: 'right', //location of equation numbers
    tags: 'all',
    packages: {'[+]': ['tagformat', 'sections', 'autoload-all', 'mathtools']},
    tagformat: {
      number: (n) => {
        const section = MathJax.config.section;
        return (section.useLetters ? section.letters[section.n] : section.n) + '.' + n;
      }
    }
  },

  chtml: {
   mtextInheritFont: true,         // font to use for mtext, if not inheriting (empty means use MathJax fonts)
   displayOverflow: 'linebreak'
  },

  linebreaks: {                  // options for when overflow is linebreak
      inline: true,                   // true for browser-based breaking of inline equations
      width: '100%',                  // a fixed size or a percentage of the container width
      lineleading: .2,                // the default lineleading in em units
      LinebreakVisitor: null,         // The LinebreakVisitor to use
  },



  startup: {
    ready() {
      const Configuration = MathJax._.input.tex.Configuration.Configuration;
      const CommandMap = MathJax._.input.tex.SymbolMap.CommandMap;
      new CommandMap('sections', {
        nextSection: 'NextSection',
        setSection: 'SetSection',
      }, {
        NextSection(parser, name) {
          MathJax.config.section.n++;
          parser.tags.counter = parser.tags.allCounter = 0;
        },
        SetSection(parser, name) {
          const section = MathJax.config.section;
          const c = parser.GetArgument(name);
          const n = section.letters.indexOf(c);
          if (n >= 0) {
            section.n = n;
            section.useLetters = true;
          } else {
            section.n = parseInt(c);
            section.useLetters = false;
          }
        }
      });
      Configuration.create(
        'sections', {handler: {macro: ['sections']}}
      );
      MathJax.startup.defaultReady();
    }
  }
};

