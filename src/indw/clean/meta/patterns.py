from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from indw.clean.artifact.registry import line_is_artifact
from indw.clean.document.patterns import _CODE_FENCE, _METADATA_LINE, _UI_LINE, _WORD

_COPYRIGHT_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'copyright\s*(?:©|\(c\)|\(C\))?|'
    r'©\s*\d{4}|'
    r'(?:\(c\)|\(C\))\s*\d{4}|'
    r'all\s+rights\s+reserved|'
    r'proprietary\s+(?:and\s+)?confidential|'
    r'unauthorized\s+(?:reproduction|distribution|copying)|'
    r'permission\s+(?:is\s+)?(?:not\s+)?granted\s+to\s+(?:copy|reproduce|distribute)|'
    r'redistribution\s+(?:and\s+)?(?:use\s+)?(?:in\s+source\s+and\s+binary\s+forms)?|'
    r'this\s+(?:program|software|work)\s+is\s+(?:free|licensed|copyrighted)|'
    r'spdx-license-identifier\s*:|'
    r'creative\s+commons|cc[\s-]?by(?:[\s-]?nc)?(?:[\s-]?sa)?|'
    r'without\s+any\s+warranty|'
    r'(?:no\s+)?warranty\s+(?:is\s+provided|disclaimer)|'
    r'fitness\s+for\s+a\s+particular\s+purpose'
    r')\b.*$'
)
_COPYRIGHT_BLOCK = re.compile(
    r'(?is)(?:'
    r'(?:^|\n)\s*(?:copyright|©|\(c\))\s*.*?(?:all\s+rights\s+reserved\.?)?\s*(?:\n|$)|'
    r'(?:^|\n)\s*permission\s+is\s+hereby\s+granted.*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*the\s+mit\s+license.*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*apache\s+license.*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*gnu\s+(?:general|lesser)\s+public\s+license.*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*mozilla\s+public\s+license.*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*bsd\s+license.*?(?:\n\n|\Z)|'
    r'(?:^|\n)\s*redistribution\s+and\s+use\s+in\s+source\s+and\s+binary\s+forms.*?(?:\n\n|\Z)'
    r')',
)
_ADA_LICENSE_BLOCK = re.compile(
    r'(?is)^(?:'
    r'[-=]{10,}\s*\n'
    r'(?:\s*--[^\n]*\n){2,}'
    r')+',
)
_ADA_COMMENT_META = re.compile(
    r'(?i)^\s*--\s*(?:'
    r'copyright|©|\(c\)|all\s+rights\s+reserved|'
    r'(?:gnu|apache|mit|bsd|lgpl|mpl)\s+license|'
    r'licensed\s+under|spdx-license-identifier|'
    r'redistribution|permission\s+is\s+hereby|'
    r'generated\s+(?:automatically|by)|do\s+not\s+edit|'
    r'\$revision\s*:|written\s+by\s*\(|author\s*:|version\s*:|'
    r'free\s+software\s+foundation'
    r')\b.*$'
)
_VENDOR_NOTICE_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'generated\s+by\s+\S+|'
    r'author\s*:\s*\S+|'
    r'version\s*:\s*[\d.]+|'
    r'\$revision\s*:\s*[\d.]+|'
    r'spdx-license-identifier\s*:\s*\S+'
    r')\s*$'
)
_LICENSE_LINE = re.compile(
    r'(?i)\b(?:'
    r'mit\s+license|apache\s+license|gnu\s+(?:general|lesser)\s+public\s+license|'
    r'bsd\s+license|mozilla\s+public\s+license|lgpl|creative\s+commons|'
    r'permission\s+is\s+hereby\s+granted|redistribution\s+and\s+use|'
    r'use,?\s+copy,?\s+modify|use,?\s+duplicate,?\s+release|'
    r'permission\s+notice\s+appear|'
    r'the\s+software\s+is\s+provided\s+"as\s+is"|'
    r'released\s+technical\s+data|government\s+makes\s+no\s+express|'
    r'without\s+any\s+warranty|licensed\s+under\s+the'
    r')\b'
)
_TRAILING_FORUM_BLOCK = re.compile(
    r'(?:\s*(?:[#•·]\s*)*Forum\s+Statistics.*)$',
    re.I | re.S,
)
_AI_PROMPT_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'you\s+are\s+an?\s+(?:ai|artificial\s+intelligence|language\s+model|helpful)\s+assistant|'
    r'as\s+an?\s+(?:ai|language\s+model)\b|'
    r'think\s+step[\s-]by[\s-]step|'
    r'(?:^|\s)(?:system|assistant|user)\s*:\s*|'
    r'<\|(?:system|assistant|user)\|>|'
    r'\[INST\]|\[/INST\]|<<SYS>>|'
    r'human\s*:\s*assistant\s*:'
    r').*$'
)
_FORUM_STATS_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'(?:joined|posts|likes|views|replies|reputation|user\s*id|username)\s*:\s*\S+|'
    r'total\s+(?:topics|posts|replies|views|members)\s*[\d,]+|'
    r'•\s+total\s+(?:topics|posts|replies|views)\s*[\d,]+|'
    r'thread\s+starter\s*:|'
    r'signature\s*:'
    r').*$'
)
_FORUM_TIMESTAMP = (
    r'(?:'
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\s+(?:at\s+\d{1,2}:\d{2}|\s+'\d{2}\s+at\s+\d{1,2}:\d{2})|"
    r'\d{1,2}\s+at\s+\d{1,2}:\d{2}'
    r')'
)
_FORUM_BULLET_COMMENT = re.compile(
    rf'(?i)•\s+[^•]+?(?:–\s*[\w.-]+\s+)?{_FORUM_TIMESTAMP}'
)
_FORUM_QUOTE_PREFIX = re.compile(
    r'(?i)\b(?:hello\s+\w+\s+)?originally\s+posted\s+by\s+\w+\s+'
)
_FORUM_ANSWER_TRANSITION = re.compile(
    r'(?i)(?<!\w)(?:'
    r'take|plot|draw|gradient|intercept|therefore|thus|the\s+solution|'
    r'adopting|note\s+that|here\s+we|in\s+conclusion|answer\s*:|'
    r'solution\s*:|step\s+\d'
    r')\b'
)
_WIKI_NAV_INLINE = re.compile(r'(?i)\bjump\s+to\s*:\s*navigation\s*,\s*search\b')
_FORUM_USERNAME_GREETING = re.compile(r'(?i)^\s*hello\s+\w+\s*$')
_LEGAL_FOOTER = re.compile(
    r'(?i)\b(?:'
    r'terms\s+of\s+(?:service|use)|privacy\s+policy|cookie\s+policy|'
    r'legal\s+disclaimer|disclaimer\s*:|'
    r'by\s+using\s+this\s+(?:site|service|website)'
    r')\b'
)
_EDITORIAL_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'last\s+(?:edited|modified|updated|reviewed)\s*(?:on|at|:)?|'
    r'updated\s+on\s+\d|modified\s+on\s+\d|'
    r'(?:created|published|posted)\s+(?:on|at|by)\s+|'
    r'(?:written|authored|edited)\s+by\s+|'
    r'by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s*[|•·]\s*(?:\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))|'
    r'revision\s+history|version\s+history|change\s*log|changelog|'
    r'edit\s+history|contributors?\s*:|'
    r'published\s*:?\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+\s+\d{1,2},?\s+\d{4})|'
    r'(?:first\s+)?published\s+(?:in\s+)?\d{4}|'
    r'reading\s+time\s*:|tags?\s*:|categories\s*:|author\s+bio\b'
    r')\b.*$'
)
_BOILERPLATE_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'(?:(?:home|about|contact|products?|services?|blog|shop|support|faq)\s*[|›>»/\\]\s*){2,}|'
    r'(?:home|about|contact)\s*[|›>»/\\]|'
    r'breadcrumb|you\s+are\s+here\s*:|'
    r'(?:accept|manage)\s+(?:all\s+)?cookies|cookie\s+(?:settings|preferences)|'
    r'this\s+(?:site|website)\s+uses\s+cookies|'
    r'privacy\s+policy|terms\s+of\s+(?:service|use)|'
    r'skip\s+to\s+(?:main\s+)?content|'
    r'subscribe\s+to\s+(?:our\s+)?newsletter|sign\s+up\s+for\s+updates|'
    r'join\s+our\s+discord|download\s+(?:the\s+)?app|install\s+(?:the\s+)?app|'
    r'related\s+posts?|related\s+articles?|previous\s+article|next\s+article|'
    r'network\s+with\s+us|follow\s+us\b|'
    r'follow\s+us\s+on\s+(?:facebook|twitter|linkedin|instagram)|'
    r'share\s+(?:this|on)\s+(?:facebook|twitter|linkedin)|share\s+this\s+article|'
    r'advertisement|sponsored\s+content|promoted\s+content|'
    r'enable\s+javascript|javascript\s+is\s+(?:disabled|required)'
    r')\b.*$'
)
_REPO_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'(?:file|directory|path)\s*:\s*/|'
    r'(?:\./|\.\./)[\w./-]+|'
    r'commit\s+[0-9a-f]{7,40}|'
    r'git\s+(?:clone|hash|commit|sha)\b|'
    r'(?:build|ci)\s+(?:log|status|passing|failed)|'
    r'generated\s+(?:automatically|by\s+\w+)|'
    r'do\s+not\s+edit(?:\s+this\s+file)?|'
    r'auto[\s-]?generated|'
    r'\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|'
    r'(?:travis|circleci|github\s+actions|codecov|coveralls)\s+'
    r')\b.*$'
)
_README_EMPTY_HEADER = re.compile(
    r'(?im)^\s*#+\s*(?:readme|license|contributing|changelog|code\s+of\s+conduct)\s*$'
)
_HEADER_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'page\s+\d+\s+of\s+\d+|'
    r'(?:^|\s)-\s*\d+\s*-?\s*$|'
    r'isbn[\s:-]*[\d\-xX]{10,17}|'
    r'doi\s*:\s*10\.\d{4,}/\S+|'
    r'arxiv\s*:\s*\d{4}\.\d{4,}|'
    r'(?:table\s+of\s+contents|contents)\s*$|'
    r'(?:retrieved|accessed|archived)\s+(?:from|on)\s+|'
    r'archive\s+(?:identifier|url|org)\s*:|'
    r'citation\s*:\s*|'
    r'bibliographic\s+information|'
    r'article\s+id\s*:|'
    r'reproduction\s+date\s*:|'
    r'world\s+heritage\s+encyclopedia|'
    r'jsdisabledcontent|'
    r'my\s+account\s*[|›>»/\\]\s*register|'
    r'(?:title|author|language|subject)\s*:\s*\S'
    r')\b.*$'
)
_TOC_ONLY_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'(?:chapter|section|part)\s+\d+.*\.{2,}\s*\d+\s*$|'
    r'\d+\.\s+\S.{3,60}\s+\.{3,}\s*\d+\s*$'
    r')$'
)
_EMAIL_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'from\s*:|to\s*:|cc\s*:|bcc\s*:|subject\s*:|date\s*:|'
    r'message[\s-]?id\s*:|in-reply-to\s*:|references\s*:|'
    r'return[\s-]?path\s*:|received\s*:|'
    r'list[\s-]?id\s*:|mailing[\s-]?list|'
    r'unsubscribe|list-unsubscribe|'
    r'sent\s+from\s+my\s+(?:iphone|ipad|android|mobile)|'
    r'--\s*$|'
    r'>\s|'
    r'on\s+\w{3,9}\s+\d{1,2},?\s+\d{4}.+wrote\s*:\s*$'
    r').*$'
)
_FORUM_LINE = re.compile(
    r'(?i)^\s*(?:'
    r'posted\s+by\s+|original\s+poster\s*:|'
    r'originally\s+posted\s+by\s+\w+|'
    r'view\s+(?:topic|thread|profile)|'
    r'reply\s+with\s+quote|quote\s+from|'
    r'jump\s+to\s+(?:first|last)\s+(?:unread\s+)?post|'
    r'jump\s+to\s*:\s*navigation|'
    r'thread\s+(?:tools|starter)|'
    r'forum\s+navigation|'
    r'forum\s+statistics|'
    r'share\s+on\s+other\s+sites|'
    r'recommended\s+posts?|'
    r'(?:\d+\s+)?answers?\s+\d+\s+views?|'
    r'this\s+topic\s+is\s+\d+\s+days?\s+old|'
    r'please\s+post\s+a\s+new\s+topic|'
    r'•\s+\d+\s*•\s+\d+\s*•'
    r')\b.*$'
)
_FRONT_MATTER = re.compile(r'^\s*---\s*\n.*?\n---\s*\n', re.S)
_CODE_COPYRIGHT = re.compile(
    r'(?i)^\s*(?:#|//|/\*|\*)\s*(?:'
    r'copyright|©|\(c\)|all\s+rights\s+reserved|'
    r'spdx-license-identifier|licensed\s+under|'
    r'@copyright|@license|'
    r'without\s+any\s+warranty|fitness\s+for\s+a\s+particular\s+purpose'
    r')\b'
)
_CODE_GENERATED = re.compile(
    r'(?i)^\s*(?:#|//|/\*|\*)\s*(?:'
    r'generated\s+(?:automatically|by)|auto[\s-]?generated|'
    r'do\s+not\s+edit|machine\s+generated|'
    r'@generated|@file\s+generated|'
    r'ide\s+metadata|build\s+metadata'
    r')\b'
)
_CODE_AUTHOR_META = re.compile(
    r'(?i)^\s*(?:#|//|/\*|\*)\s*(?:'
    r'@author\b|@version\b|@date\b|@modified\b|'
    r'author\s*:|file\s*:|filename\s*:|'
    r'revision\s*:|rcsid\s*:|cvs\s+id\s*:'
    r')\b'
)
_CODE_LICENSE_BLOCK = re.compile(
    r'(?is)/\*[!]?\s*(?:'
    r'copyright|license|permission\s+is\s+hereby|mit\s+license|apache\s+license|'
    r'gnu\s+(?:general|lesser)\s+public'
    r').*?\*/'
)
_INFORMATIVE_COMMENT = re.compile(
    r'(?i)(?:'
    r'\b(?:explain|because|note\s+that|important|warning|caution|example|usage|'
    r'parameter|returns?|raises?|see\s+also|algorithm|complexity|todo\s*:\s*\w{4,})|'
    r'@\w+\s+\S|'
    r'"""[\s\S]{20,}"""|'
    r"'''[\s\S]{20,}'''"
    r')'
)
_INLINE_LICENSE_CHUNK = re.compile(
    r'(?is)(?:'
    r'[-=]{8,}\s*|'
    r'--\s*(?:'
    r'copyright|©|\(c\)|all\s+rights\s+reserved|'
    r'(?:gnu|apache|mit|bsd|lgpl|mpl|zlib)\s+license|'
    r'permission\s+is\s+hereby|redistribution|'
    r'spdx-license-identifier|generated\s+(?:automatically|by)|'
    r'free\s+software\s+foundation|gnat\s+library|gnarl|'
    r'licensed\s+under|do\s+not\s+edit|'
    r'written\s+by|author\s*:|version\s*:|'
    r'\$revision\s*:|see\s+license\s+for\s+details'
    r')[^-]*'
    r')'
)
_COLLAPSED_CODE_START = re.compile(
    r'(?i)\b(?:'
    r'pragma|package(?:\s+body)?|with|use|procedure|function|type|subtype|'
    r'#include|#define|def\s+|class\s+|namespace\s+|import\s+|from\s+'
    r')\b'
)
_SOCIAL_PROMO_PREFIX = re.compile(
    r'(?i)\b(?:'
    r'our\s+discord|join\s+our\s+discord|discord\s+hit|'
    r'join\s+here|meet\s+students|network\s+with\s+us|'
    r'join\s+us\s+on\s+facebook|get\s+the\s+latest\s+news\s+and\s+updates|'
    r'follow\s+us\s+on\s+facebook|ask\s+top\s+educators|'
    r'discord\s+server|\d+k?\s+members!?|'
    r'celebrat(?:e|ing)\s+\d+\s*(?:k|thousand)\s+members'
    r')\b'
)
_ENCYCLOPEDIA_CHROME = re.compile(
    r'(?i)(?:'
    r'#jsdisabledcontent|'
    r'article\s+id\s*:\s*\w+|'
    r'reproduction\s+date\s*:|'
    r'world\s+heritage\s+encyclopedia|'
    r'my\s+account\s*[|›>»/\\]\s*register\s*[|›>»/\\]\s*help'
    r')'
)
_AI_TRAINING_BODY = re.compile(
    r'(?i)\b(?:'
    r'user\s+will\s+you\s+give\s+you\s+a\s+(?:task|question)|'
    r'your\s+task\s+is\s+to\s+(?:answer|complete|generate)|'
    r'question:\s*you\s+are\s+an?\s+(?:ai|helpful)'
    r')\b'
)
_INSTRUCTION_LABEL = re.compile(
    r'(?im)^\s*(?P<label>'
    r'question|answer|instruction|task|prompt|user|assistant|system|human|'
    r'response|output|explanation|justification|rationale|premise|hypothesis|'
    r'context|article|summary|input|query|reply|select\s+from|'
    r'choose(?:\s+the\s+correct\s+answer)?|additional\s+answer|'
    r'q|a'
    r')\s*:\s*(?P<tail>.*)$'
)
_INSTRUCTION_SCAFFOLD = re.compile(
    r'(?i)\b(?:'
    r'you\s+are\s+an?\s+(?:ai|artificial\s+intelligence|helpful|language\s+model)\s*(?:assistant)?|'
    r'complete\s+the\s+following\s+task|your\s+(?:goal|task)\s+is\s+to|'
    r'while\s+performing\s+the\s+task|answer\s+the\s+following\s+question|'
    r'generate\s+a\s+summary|summarize\s+the\s+following(?:\s+(?:text|article))?|'
    r'summarize\s+this\s+(?:text|article|passage)?|'
    r'provide\s+a\s+(?:detailed|long)\s+answer'
    r')\b'
)
_COT_MARKERS = re.compile(
    r'(?i)\b(?:'
    r'think\s+step[\s-]?by[\s-]?step|let\'?s\s+think(?:\s+step\s+by\s+step)?|'
    r'chain\s+of\s+thought|reasoning\s*:|scratchpad\s*:|'
    r'internal\s+reasoning|hidden\s+reasoning|intermediate\s+reasoning'
    r')\b'
)
_META_LABELS = frozenset({
    'instruction', 'task', 'prompt', 'user', 'assistant', 'system', 'human',
    'select', 'choose', 'query',
})
_KNOWLEDGE_LABELS = frozenset({
    'body', 'premise', 'hypothesis', 'article', 'context', 'summary',
    'answer', 'response', 'output', 'explanation', 'justification', 'rationale',
    'question',
})
