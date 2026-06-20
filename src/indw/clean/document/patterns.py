from __future__ import annotations

import re

_UI_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'add\s+comment|add\s+reply|edit|share|report|save|follow|login|sign\s*up|sign\s*in|'
    r'register\s*(?:\||›|>|»|/|$)|'
    r'advertisement|sponsored(?:\s+content)?|cookie(?:s)?\s+(?:policy|notice|consent)|'
    r'accept\s+(?:all\s+)?cookies|'
    r'skip\s+to\s+(?:main\s+)?content|subscribe|newsletter|follow\s+us|'
    r'join\s+our\s+discord|download\s+(?:the\s+)?app|install\s+(?:the\s+)?app|'
    r'share\s+this\s+article|related\s+posts?|previous\s+article|next\s+article|'
    r'social\s+media|share\s+on\s+(?:facebook|twitter|x|linkedin)|'
    r'upvote|downvote|vote|reply|comment|permalink|load\s+more\s+comments|show\s+more|'
    r'share\s+report\s+save|'
    r'previous|next|home\s*[|›>»/]|breadcrumb|you\s+are\s+here|navigation|menu|sidebar|footer|header|'
    r'add\s+to\s+cart|view\s+cart|shopping\s+cart|checkout|buy\s+now|shop\s+now|'
    r'email\s+(?:address|signup|sign[\s-]?up)|join\s+our\s+mailing\s+list|'
    r'affiliate\s+disclosure|paid\s+partnership|commission\s+may\s+be\s+earned|'
    r'session[_\s-]?id|tracking[_\s-]?id|utm_[a-z]+|analytics|gtag|'
    r'product\s+recommendations?|customers?\s+also\s+bought|frequently\s+bought\s+together'
    r')\s*[|:]?\s*$'
)

_METADATA_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'\d+\s+(?:years?|months?|weeks?|days?|hours?|minutes?)\s+ago|'
    r'(?:edited|posted|updated|asked|answered|published|modified)\s*:?\s*\d+|'
    r'(?:last\s+)?(?:modified|updated|edited)\s*:|reading\s+time\s*:|'
    r'tags?\s*:|categories\s*:|author\s+bio\b|'
    r'\d+(?:\.\d+)?[km]?\s+(?:views?|upvotes?|downvotes?|points?|replies?|comments?|likes?)|'
    r'(?:joined|posts|likes|views|replies|reputation|user\s*id|username)\s*:\s*\S+|'
    r'thread\s+starter\s*:|'
    r'originally\s+posted\s+by\s+\w+|'
    r'jump\s+to\s*:\s*navigation|'
    r'score:\s*\d+|reputation:\s*\d+|member\s+since|joined\s+\d+|'
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\s+'\d{2}\s+at\s+\d|"
    r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\s+at\s+\d|'
    r'page\s+\d+\s+of\s+\d+|©\s*\d{4}|all\s+rights\s+reserved|'
    r'@\w+\s+\d+\s+(?:years?|months?|days?)\s+ago|'
    r'u/\w+\s*[·•|]\s*\d+\s+(?:years?|months?|days?)\s+ago'
    r')\s*[|.!]?\s*$'
)

_ACK_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'\+1|thanks?|thank\s+you|thx|ty|this\s+worked|great\s+answer|helpful|accepted|'
    r'nice|awesome|perfect|solved|works\s+for\s+me|exactly\s+what\s+i\s+needed'
    r')\s*[!.]?\s*$'
)

_REPLY_ACK_LINE = re.compile(
    r'(?i)^\s*(?:comment|reply|meta)\s*:\s*(?:'
    r'\+1|thanks?|thank\s+you|thx|ty|nice|helpful|this\s+worked|great|awesome|perfect'
    r')\b'
)

_MODERATOR_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'\[?(?:moderator|mod|admin|automoderator)\]?|'
    r'this\s+(?:post|comment|thread)\s+(?:has\s+been|was)\s+(?:removed|deleted|locked)|'
    r'locked\s+by\s+moderator|removed\s+by\s+moderator'
    r')'
)

_THREAD_MARKER = re.compile(
    r'(?i)^\s*(?:question|answer|accepted\s+answer|best\s+answer|comment|reply|'
    r'user|assistant|original\s+post|op)\s*:\s*'
)

_QA_QUESTION = re.compile(r'(?i)^\s*(?:question|q)\s*:\s*(.+)$')
_QA_ANSWER = re.compile(r'(?i)^\s*(?:answer|accepted\s+answer|best\s+answer|a)\s*:\s*(.+)$')

_HTML_SCRIPT_STYLE = re.compile(r'<(script|style|noscript)[^>]*>.*?</\1>', re.I | re.S)
_HTML_TAG = re.compile(r'<[^>]+>')
_HTML_BREAK = re.compile(r'<br\s*/?>', re.I)

_HTML_ENTITIES = {
    '&nbsp;': ' ',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&apos;': "'",
}

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_CODE_FENCE = re.compile(r'```[\s\S]*?```|`[^`\n]+`', re.M)

_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_INVISIBLE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]')
_PIPE_NAV = re.compile(r'^\s*[^|]{0,40}(?:\s*\|\s*[^|]{0,40}){2,}\s*$')
_REPEAT_PUNCT = re.compile(r'([!?.,;:])\1{2,}')
_MULTI_BLANK = re.compile(r'\n{3,}')
