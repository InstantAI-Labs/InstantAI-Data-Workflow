from __future__ import annotations

import re

_CHARS_PER_TOKEN = 3.8

_REPEATED_WORD = re.compile(r'\b(\w+)(?:\s+\1){2,}\b', re.I)
_REPEATED_SYM = re.compile(r'([!?.,;:\-_=*#])\1{4,}')
_GARBAGE_UNICODE = re.compile(r'[\uFFFD\u0000-\u0008]{2,}')
_RANDOM_PUNCT = re.compile(r'(?:(?:[!?.,]){5,}|(?:\s[^\w\s]){8,})')
_HTML_TAG = re.compile(r'<[a-z][^>]*>', re.I)
_COOKIE = re.compile(r'(?i)cookie(?:s)?\s+(?:policy|consent|banner)|accept\s+(?:all\s+)?cookies')
_NAV = re.compile(
    r'(?i)(?:home|about|contact|privacy|terms)\s*[|›>»/\\]\s*(?:home|about|contact|products|blog|shop)',
)
_FORUM = re.compile(r'(?i)\b(?:posted by|last reply|view topic|re:\s|upvoted|downvoted|karma)\b')
_SEO = re.compile(r'(?i)\b(?:click here|best \d+ ways|ultimate guide|you won\'?t believe)\b')
_AI_SLOP = re.compile(
    r"(?i)\b(?:as an ai|delve into|it's important to note|rich tapestry|multifaceted|holistic approach)\b"
)
_OCR = re.compile(r'(?i)\b(?:lorem ipsum|xxx+|abc abc)\b')
_WORD_SALAD = re.compile(r'(?i)\b(?:the the|and and|of of|in in)\b')
