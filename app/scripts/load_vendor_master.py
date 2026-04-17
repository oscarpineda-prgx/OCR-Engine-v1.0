from __future__ import annotations

import argparse

from app.db import models  # noqa: F401
from app.db.session import Base, SessionLocal, engine
from app.services.vendor_master.loader import refresh_vendor_master_from_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Load vendor master Excel into SQLite table vendor_master.")
    parser.add_argument(
        "--excel",
        required=True,
        help="Path to the vendor master Excel file.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows instead of replacing existing vendor_master content.",
    )
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        stats = refresh_vendor_master_from_excel(
            db=db,
            excel_path=args.excel,
            replace=not args.append,
        )
    finally:
        db.close()

    print(stats)


if __name__ == "__main__":
    main()

