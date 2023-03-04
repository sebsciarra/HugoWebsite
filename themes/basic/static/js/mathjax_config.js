

window.MathJax = {
  section: -1,
   loader: {load: ['[tex]/tagformat']},
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']], //allow inline math
    displayMath: [['$$','$$']],
    tagSide: 'right', //location of equation numbers
    tags: 'all',
    packages: {'[+]': ['tagformat', 'sections', 'autoload-all']},
    tagformat: {
      number: (n) => MathJax.config.section + '.' + n
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
          MathJax.config.section++;
          parser.tags.counter = parser.tags.allCounter = 0;
        },
        SetSection(parser, name) {
          const n = parser.GetArgument(name);
          MathJax.config.section = parseInt(n);
        }
      });
      Configuration.create(
        'sections', {handler: {macro: ['sections']}}
      );
      MathJax.startup.defaultReady();
    }
  }
};


