"""Export the final HTML report to Markdown without changing frozen outputs."""
import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'results/improvements_final'


class Node:
    def __init__(self, tag, attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []


class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node('root')
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'meta', 'img', 'br', 'link', 'hr', 'input'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def descendants(node, tag):
    if isinstance(node, str):
        return []
    return ([node] if node.tag == tag else []) + [
        item for child in node.children for item in descendants(child, tag)
    ]


def render(node):
    if isinstance(node, str):
        return re.sub(r'\s+', ' ', node).replace('*', '\\*')
    tag = node.tag
    if tag in {'head', 'style'} or node.attrs.get('class') == 'label':
        return ''
    if tag == 'img':
        filename = {'본실험 지표 비교': 'comparison.png',
                    '지도 확대에 따른 연산량·저장량 증가': 'scaling.png'}[node.attrs['alt']]
        path = ROOT / 'results/improvements/report' / filename
        assert base64.b64decode(node.attrs['src'].split(',', 1)[1]) == path.read_bytes()
        return f"\n\n![{node.attrs['alt']}]({path.as_posix()})\n\n"
    if tag == 'table':
        rows = []
        for tr in descendants(node, 'tr'):
            cells = [child for child in tr.children if isinstance(child, Node) and child.tag in {'th', 'td'}]
            rows.append([render(cell).strip().replace('|', '\\|').replace('\n', ' ') for cell in cells])
        assert rows and all(len(r) == len(rows[0]) for r in rows)
        rows.insert(1, ['---'] * len(rows[0]))
        return '\n\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows) + '\n\n'
    content = ''.join(render(child) for child in node.children).strip()
    if tag in {'h1', 'h2', 'h3'}:
        return '\n\n' + '#' * int(tag[1]) + ' ' + content + '\n\n'
    if tag in {'strong', 'b'}:
        return '**' + content + '** '
    if tag == 'code':
        return '`' + content + '`'
    if tag == 'br':
        return '\n\n'
    if tag == 'a':
        path = (DIRECTORY / node.attrs['href']).resolve()
        assert path.is_file()
        return f'[{content}]({path.as_posix()})'
    if tag == 'li':
        return '- ' + content + '\n'
    if tag in {'p', 'div', 'ul', 'main'}:
        return '\n\n' + content + '\n\n'
    return content


def main():
    provenance = json.loads((DIRECTORY / 'provenance.json').read_text(encoding='utf-8'))
    assert provenance['complete'] and provenance['verified_rows'] == 28270
    parser = Parser()
    parser.feed((DIRECTORY / 'report.html').read_text(encoding='utf-8'))
    text = render(descendants(parser.root, 'main')[0])
    text = re.sub(r'[ \t]+\n', '\n', text).replace('** ,', '**,')
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    heading, rest = text.split('\n\n', 1)
    assessment = '''실험 완료일: 2026-10-14 · 상태: 전체 실행 및 서버·로컬 검증 완료

> **결과 판단:** 기본 LLM-A* 대비 관찰된 개선 효과가 뚜렷한 후보는 **1번(경유점 정제)**과 **3번(조건부 호출)**입니다. 1번은 탐색 연산량 감소, 3번은 연산량과 API 호출 수 감소가 핵심입니다. 2번은 출력 유효율이 소폭 높아졌지만, 동일 호출 횟수의 대조군 대비 일관된 탐색 개선을 보이지 않았습니다.
>
> 이 보고서의 “개선 효과”는 이번 데이터에서 관찰한 차이를 뜻합니다. **통계적 유의성 검정이나 신뢰구간 산출을 수행하지 않았으므로, 통계적으로 유의하다고 결론 내릴 수는 없습니다.** 또한 세 개선안 모두 GPT 없는 g+2h 대조군보다 본실험의 집계 연산량이 적지는 않았습니다.
'''
    text = heading + '\n\n' + assessment + '\n' + rest + '\n'
    destination = DIRECTORY / 'report.md'
    destination.write_text(text, encoding='utf-8')
    assert text.count('![') == 2 and 'data:image' not in text
    assert all(term in text for term in ['28,270', '8,814', '+12.50%', '+49.91%', '50.20%', '실행 순서', '통계적 유의성'])
    print(json.dumps({'output': str(destination), 'bytes': destination.stat().st_size,
                      'tables': len(descendants(parser.root, 'table')), 'figures': 2}, ensure_ascii=True))


if __name__ == '__main__':
    main()
