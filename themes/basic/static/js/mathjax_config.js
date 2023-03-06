
//see https://github.com/mathjax/MathJax/issues/3013
window.MathJax = {
  section: {
    n: -1,
    useLetters: false,
    letters: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  },
  loader: {load: ['[tex]/tagformat']},
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']], //allow inline math
    displayMath: [['$$','$$']],
    tagSide: 'right', //location of equation numbers
    tags: 'all',
    packages: {'[+]': ['tagformat', 'sections', 'autoload-all']},
    tagformat: {
      number: (n) => {
        const section = MathJax.config.section;
        return (section.useLetters ? section.letters[section.n] : section.n) + '.' + n;
      }
    }
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
