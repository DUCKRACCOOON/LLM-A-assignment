"""Export the current Korean report as plain text, preserving source spacing.

This supports the macros used by our report, not arbitrary LaTeX documents.
Tables and captions are retained; image pixels and layout instructions are omitted.
"""
import re
from .experiment import ROOT


def main():
    source = (ROOT / 'report/LLM_Astar_Reproduction_Report.tex').read_text(encoding='utf-8')
    title = re.search(r'\\title\{(.*?)\}\s*\\author', source, re.S).group(1)
    text = title + '\n\n' + source.split(r'\begin{document}', 1)[1]
    text = re.sub(r'\\bibitem\[[^\]]*\]\{[^}]*\}', '\n', text)
    text = re.sub(r'\\begin\{thebibliography\}\{[^}]*\}', '\nReferences\n', text)
    text = re.sub(r'\\begin\{tabular\}\{[^}]*\}', '\n', text)
    text = re.sub(r'\\begin\{minipage\}\{[^}]*\}', '', text)
    text = re.sub(r'\\includegraphics\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'\\setlength\{[^}]*\}\{[^}]*\}', '', text)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = text.replace(r'\citet{meng2025}', 'Meng et al. (2025)')
    text = text.replace(r'\citep{openai_mini,openai_gpt35}', '(OpenAI, 2026a, 2026b)')
    for label, number in [('tab:reproduction', '1'), ('tab:ablation', '2'), ('fig:cases', '2')]:
        text = text.replace('\\ref{' + label + '}', number)
    text = text.replace(r'100\exp\left(\frac{1}{N}\sum_{i=1}^{N}\log\frac{m_i}{m_i^{A*}}\right).',
                        '100 × exp((1/N) × Σ log(m_i / m_i^(A*))).')
    text = re.sub(r'\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?', '\n', text)
    text = re.sub(r'\\(?:section|subsection|paragraph|caption)\{([^{}]*)\}', r'\n\1\n', text)
    text = re.sub(r'\\(?:texttt|emph|url)\{([^{}]*)\}', r'\1', text)
    for macro in ['maketitle', 'centering', 'small', 'noindent', 'appendix', 'protect',
                  'toprule', 'midrule', 'bottomrule']:
        text = text.replace('\\' + macro, '')
    text = text.replace(r'\item ', '\n• ')
    for old, new in [(r'\%', '%'), (r'\|', '‖'), (r'\times', '×'), (r'\ldots', '…'),
                     (r'\\', '\n'), ('$', ''), ('~', ' '), (' & ', ' | ')]:
        text = text.replace(old, new)
    unhandled = re.search(r'\\[A-Za-z]+', text)
    if unhandled:
        raise ValueError('Unhandled macro: ' + unhandled[0])
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'
    destination = ROOT / 'output/LLM_Astar_Report_KO.txt'
    destination.write_text(text, encoding='utf-8')
    print(f'Exported {len(text):,} characters with source Korean spacing: {destination}')


if __name__ == '__main__':
    main()
