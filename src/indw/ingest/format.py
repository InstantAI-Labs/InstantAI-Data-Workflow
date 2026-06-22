from __future__ import annotations

from typing import Any, Callable, Optional

from indw.ingest.transcript import TranscriptBuilder, wrap_block


def plain_text(row: dict[str, Any], field: str = 'text') -> str:
    val = row.get(field, '')
    return val.strip() if isinstance(val, str) else ''


def format_conversation(row: dict[str, Any]) -> str:
    messages = row.get('message_tree') or row.get('messages') or []
    if isinstance(messages, dict):
        messages = [messages]
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role', 'user')
        content = msg.get('text') or msg.get('content', '')
        if content:
            lines.append(f'{role}: {content.strip()}')
    if lines:
        return '\n'.join(lines)
    for key in ('text', 'content', 'instruction', 'response'):
        if row.get(key):
            return str(row[key]).strip()
    return ''


def format_ultrachat(row: dict[str, Any]) -> str:
    data = row.get('data', row.get('messages', []))
    if isinstance(data, list):
        parts = []
        for turn in data:
            if isinstance(turn, str):
                parts.append(turn.strip())
            elif isinstance(turn, dict):
                parts.append((turn.get('content') or turn.get('text', '')).strip())
        return '\n'.join((p for p in parts if p))
    return plain_text(row, 'text')


def format_oasst(row: dict[str, Any]) -> str:
    text = (row.get('text') or row.get('content') or '').strip()
    if not text:
        return ''
    role = row.get('role', 'user')
    return f'{role}: {text}'


def format_code(row: dict[str, Any], field: str = 'code') -> str:
    code = plain_text(row, field)
    lang = row.get('language') or row.get('lang', '')
    if lang and code:
        return f'# language: {lang}\n{code}'
    return code


def format_frontier_chat(
    row: dict[str, Any],
    *,
    system: str = '',
    include_thoughts: bool = False,
) -> str:
    sys_text = system or (row.get('system') or row.get('system_prompt') or '')
    data = row.get('data', row.get('messages', []))
    if not isinstance(data, list):
        plain = format_ultrachat(row)
        if not plain:
            return ''
        b = TranscriptBuilder()
        if sys_text:
            b.add('system', str(sys_text).strip())
        b.add('user', plain)
        return b.build()
    b = TranscriptBuilder()
    if sys_text:
        b.add('system', str(sys_text).strip())
    pending_thoughts: Optional[str] = None
    if include_thoughts and row.get('thoughts'):
        pending_thoughts = str(row['thoughts']).strip()
    role = 'user'
    for turn in data:
        if isinstance(turn, str):
            content = turn.strip()
        elif isinstance(turn, dict):
            content = (turn.get('content') or turn.get('text') or '').strip()
            role = turn.get('role', role)
        else:
            continue
        if not content:
            continue
        tag = 'assistant' if role in ('assistant', 'gpt', 'bot') else 'user'
        if tag == 'assistant' and pending_thoughts:
            b.add('thoughts', pending_thoughts)
            pending_thoughts = None
        b.add(tag, content)
        role = 'assistant' if tag == 'user' else 'user'
    return b.build()


def format_ultrachat_frontier(
    row: dict[str, Any],
    *,
    system: str = '',
    include_thoughts: bool = False,
) -> str:
    return format_frontier_chat(row, system=system, include_thoughts=include_thoughts)


def format_code_frontier(row: dict[str, Any], field: str = 'code') -> str:
    code = format_code(row, field)
    if not code:
        return ''
    lang = row.get('language') or row.get('lang', '')
    inner = f'# language: {lang}\n{code}' if lang else code
    return wrap_block('code', inner) + '\n<|endoftext|>\n'


def format_instruction_qa(row: dict[str, Any]) -> str:
    system = (row.get('system_prompt') or row.get('system') or '').strip()
    instruction = (row.get('instruction') or row.get('prompt') or row.get('question') or '').strip()
    context = (row.get('input') or row.get('context') or '').strip()
    output = (row.get('output') or row.get('response') or row.get('answer') or '').strip()
    if not output:
        return instruction or context or system
    question_parts = [part for part in (system, instruction, context) if part]
    question = '\n\n'.join(question_parts).strip()
    if not question:
        return output
    return f'Question: {question}\n\nAnswer: {output}'


def format_stack_exchange(row: dict[str, Any]) -> str:
    question = (row.get('question') or row.get('title') or '').strip()
    answers = row.get('answers') or row.get('answer') or []
    answer = ''
    if isinstance(answers, str):
        answer = answers.strip()
    elif isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, str):
            answer = first.strip()
        elif isinstance(first, dict):
            answer = (first.get('text') or first.get('answer') or '').strip()
    if question and answer:
        return f'Question: {question}\n\nAnswer: {answer}'
    return question or answer


FORMATTERS: dict[str, Callable[[dict[str, Any]], str]] = {
    'plain': lambda r: plain_text(r, 'text'),
    'conversation': format_conversation,
    'ultrachat': format_ultrachat,
    'oasst': format_oasst,
    'code': lambda r: format_code(r, 'code'),
    'frontier': format_frontier_chat,
    'ultrachat_frontier': format_ultrachat_frontier,
    'code_frontier': lambda r: format_code_frontier(r, 'code'),
    'stack_exchange': format_stack_exchange,
    'instruction_qa': format_instruction_qa,
}
