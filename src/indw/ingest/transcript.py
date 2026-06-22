from __future__ import annotations

from dataclasses import dataclass, field


def _tag_pair(name: str) -> tuple[str, str]:
    return f'<|{name}|>', f'<|/{name}|>'


@dataclass
class Block:
    name: str
    content: str
    open_tag: str
    close_tag: str

    def render(self) -> str:
        body = self.content.strip()
        if not body:
            return ''
        return f'{self.open_tag}\n{body}\n{self.close_tag}\n'


@dataclass
class TranscriptBuilder:
    blocks: list[Block] = field(default_factory=list)
    append_eos: bool = True
    eos_token: str = '<|endoftext|>'

    def add(self, name: str, content: str) -> TranscriptBuilder:
        open_tag, close_tag = _tag_pair(name)
        self.blocks.append(Block(
            name=name,
            content=content,
            open_tag=open_tag,
            close_tag=close_tag,
        ))
        return self

    def build(self, *, hide_names: frozenset[str] | None = None) -> str:
        hidden = hide_names or frozenset()
        parts: list[str] = []
        for block in self.blocks:
            if block.name in hidden:
                continue
            rendered = block.render()
            if rendered:
                parts.append(rendered)
        text = ''.join(parts)
        if self.append_eos and text and not text.rstrip().endswith(self.eos_token):
            text = text.rstrip() + '\n' + self.eos_token + '\n'
        return text


def wrap_block(name: str, content: str) -> str:
    return TranscriptBuilder(append_eos=False).add(name, content).build().strip()
