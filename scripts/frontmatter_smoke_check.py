from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.emit_obsidian import smoke_check_note_render


def main() -> int:
    template_path = Path("templates/obsidian_note.md")
    smoke_check_note_render(template_path=template_path)
    print(f"Frontmatter smoke check passed for {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
