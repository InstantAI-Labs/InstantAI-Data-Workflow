from __future__ import annotations

from typing import Iterator

from indw.schedule.core import _InterleavedSources


class IngestCoordinator:
    def __init__(self, interleaved: _InterleavedSources):
        self._interleaved = interleaved
        self._seq = 0

    def __iter__(self) -> Iterator[tuple[str, int, str, int]]:
        for src_name, _path, line_no, line in self._interleaved:
            seq = self._seq
            self._seq += 1
            yield src_name, line_no, line, seq

    def close(self) -> None:
        self._interleaved.close()
