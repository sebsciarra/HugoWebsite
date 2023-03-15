
//see https://github.com/mathjax/MathJax/issues/3013


window.MathJax = {
  section: {
    n: -1,
    useLetters: false,
    letters: "AABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
    }},

  chtml: {
   mtextInheritFont: true,         // font to use for mtext, if not inheriting (empty means use MathJax fonts)
   displayOverflow: 'linebreak'
  },

  linebreaks: {                  // options for when overflow is linebreak
      inline: true,                   // true for browser-based breaking of inline equations
      width: '100%',                  // a fixed size or a percentage of the container width
      lineleading: 2,                // the default lineleading in em units
      LinebreakVisitor: null,         // The LinebreakVisitor to use
  },



  startup: {
    ready() {
                  const {CommonWrapper} = MathJax._.output.common.Wrapper;
      const {LineBBox} = MathJax._.output.common.LineBBox;

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
        },
      });
      Object.assign(CommonWrapper.prototype, {
        invalidateBBox(bubble = true) {
          if (this.bboxComputed || this._breakCount >= 0) {
            this.bboxComputed = false;
            this.lineBBox = [];
            this._breakCount = -1;
            if (this.parent && bubble) {
              this.parent.invalidateBBox();
            }
          }
        },
        _getLineBBox: CommonWrapper.prototype.getLineBBox,
        getLineBBox(i) {
          if (!this.lineBBox[i] && !this.breakCount) {
            const obox = this.getOuterBBox();
            this.lineBBox[i] = LineBBox.from(obox, this.linebreakOptions.lineleading);
          }
          return this._getLineBBox(i);
        }
      });

      Configuration.create(
        'sections', {handler: {macro: ['sections']}}
      );
      MathJax.startup.defaultReady();
    }
  }
};

