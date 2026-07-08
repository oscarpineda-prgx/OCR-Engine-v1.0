from __future__ import annotations

import argparse
import json

from app.api.routes.batches import _process_single_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated single-document processor")
    parser.add_argument("--doc-id", required=True, type=int)
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--source-type", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = _process_single_document(args.doc_id, args.file_path, args.source_type)
    # Keep worker output ASCII-safe so Windows console encodings cannot fail
    # after extraction succeeds and before the parent process receives JSON.
    print(json.dumps(result.__dict__, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
