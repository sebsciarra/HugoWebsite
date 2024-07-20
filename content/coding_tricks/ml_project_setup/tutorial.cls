
%----------------------------------------------------------------------------------------
%	CLASS DEFINITION AND PARAMETERS
%----------------------------------------------------------------------------------------
%Class definition
%required to create a class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tutorial}[2023/08/25] %info in square brackets specifies release date

\providecommand{\baseclass}{article} %define \baseclass command

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\ProcessOptions\relax
\LoadClass{\baseclass}


%----------------------------------------------------------------------------------------
%	REQUIRED PACKAGES
%----------------------------------------------------------------------------------------
\RequirePackage{xurl} %ensures urls do not go outside page margins
\RequirePackage{titlecaps} %capitalizes first letter of words (needed to toc entries when using \addcontentsline)
\RequirePackage{hyperref} %capitalizes first letter of words (needed to toc entries when using \addcontentsline)
\RequirePackage{titlesec}
\RequirePackage{fancyhdr} %needed for header and footer formatting
\RequirePackage{etoolbox}
\RequirePackage{multicol} %needed for multicolumn layout in abstract papge
\RequirePackage[]{footmisc}  %removes indentation in footnotes
\RequirePackage{soul}

\RequirePackage[labelfont = {bf, up}]{caption}
\RequirePackage[flushleft]{threeparttable}
\RequirePackage{anyfontsize} %needed for tables to take on Helvetica font
\RequirePackage{textcomp} %for copyright symbol on title page
\RequirePackage{longtable}
\RequirePackage{makecell}
\RequirePackage[section]{placeins}
\RequirePackage{array}
\RequirePackage{multirow}
\RequirePackage{wrapfig}
\RequirePackage{colortbl}
\RequirePackage{pdflscape}
\RequirePackage{tabu}
\RequirePackage[normalem]{ulem}
\RequirePackage{amsthm}
\RequirePackage{amsmath} %needed for argmax
\RequirePackage{bm}  %thicker bold in math
\RequirePackage{upgreek}  %required for non-italic greek letters
\RequirePackage{longtable,booktabs,setspace}
\RequirePackage{xargs} %needed to create environments with multiple optional arguments (e.g., apaFigure)
\RequirePackage{enumitem} %needed to remove spacing between list items
\RequirePackage{geometry}
\RequirePackage[absolute]{textpos} %allows everything below title on titlepage to be positioned regardless of how long the title is
\RequirePackage{xparse}
\RequirePackage{fvextra}
\RequirePackage{framed}
\RequirePackage{float} %cannot use floatrow when using caption (or else formatting of figures and tables will be messed up)
\RequirePackage{pdflscape}
\RequirePackage{graphicx}
\RequirePackage{everypage}

\setlist{nolistsep} %remove spacings between list items
%place footnoterule definition into  new command, then modify spacing between footnoterule and footnotetext with \vskip10pt
\let\newfootnoterule\footnoterule
\def\footnoterule{\newfootnoterule \vskip10pt}

%see https://gist.github.com/Manuel-S/f285795979f42f65a144bf87ac52013b for all possible syntax highlighting customizations
\AtBeginEnvironment{Highlighting}{
  \setstretch{1}\fontsize{10pt}{10pt}
  \renewcommand{\CommentTok}[1]{\textcolor[RGB]{94, 90, 90}{#1}}%#comment colour
  \renewcommand{\DocumentationTok}[1]{\textcolor[RGB]{94, 90, 90}{#1}} %multiline comments colour
  \renewcommand{\NormalTok}[1]{\textcolor[RGB]{0, 24, 45}{#1}} %variable name color
  \renewcommand{\OtherTok}[1]{\fontsize{8pt}{8pt}\textcolor[RGB]{0, 24, 45}{#1}}
  \renewcommand{\AttributeTok}[1]{\textcolor[RGB]{0, 24, 45}{#1}} %attribute color
  \renewcommand{\FunctionTok}[1]{\textcolor[RGB]{6, 120, 225}{#1}} %function names
  \renewcommand{\StringTok}[1]{\textcolor[RGB]{212, 93, 196}{#1}} %strings
  \renewcommand{\ConstantTok}[1]{\textcolor[RGB]{212, 93, 196}{#1}} %constants such as NA, NULL
}

%vertical space above and below chunk
%\setlength{\topsep}{10pt}
%\AfterEndEnvironment{Shaded}{\vspace{10pt}}


%necessary to define tighlist
\def\tightlist{}


