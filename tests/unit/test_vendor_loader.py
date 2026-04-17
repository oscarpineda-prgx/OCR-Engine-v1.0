"""Unit tests for app/services/vendor_master/loader.py."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import VendorMaster
from app.db.session import Base
from app.services.vendor_master.loader import (
    bootstrap_vendor_master_if_empty,
    refresh_vendor_master_from_excel,
)

# ---------------------------------------------------------------------------
# Test database setup
# ---------------------------------------------------------------------------

TEST_DB_PATH = "test_vendor_loader.db"
TEST_DB_URL = f"sqlite:///{TEST_DB_PATH}"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# ---------------------------------------------------------------------------
# Module-level setup / teardown
# ---------------------------------------------------------------------------


def setup_module():
    """Create all tables before any test in this module runs."""
    Base.metadata.create_all(bind=engine)


def teardown_module():
    """Drop all tables and remove the test database file after all tests."""
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Truncate vendor_master between tests so each starts with a clean slate."""
    yield
    db = TestSession()
    try:
        db.query(VendorMaster).delete()
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_EXCEL_PATH = "/tmp/fake_vendor_master.xlsx"


def _make_vendor_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame that mimics a vendor master Excel file."""
    return pd.DataFrame(rows, dtype=str)


def _all_vendors(db) -> list[VendorMaster]:
    return db.query(VendorMaster).all()


# ===========================================================================
# 1. refresh_vendor_master_from_excel
# ===========================================================================


class TestRefreshVendorMasterFromExcel:
    """Tests for refresh_vendor_master_from_excel."""

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_inserts_records_correctly(self, mock_read_excel):
        """Records from the DataFrame are inserted into vendor_master."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Acme Corp SA de CV", "RFC": "ACO850101XYZ"},
            {"Vendor Name": "Beta Industries", "RFC": "BIN900202ABC"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                result = refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            assert result["rows_inserted"] == 2
            vendors = _all_vendors(db)
            assert len(vendors) == 2
            names = {v.vendor_name for v in vendors}
            assert "Acme Corp SA de CV" in names
            assert "Beta Industries" in names
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_vendor_name_normalized_is_populated(self, mock_read_excel):
        """vendor_name_normalized is set for each inserted record."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Acme Corp SA de CV", "RFC": "ACO850101XYZ"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            vendor = _all_vendors(db)[0]
            assert vendor.vendor_name_normalized is not None
            assert len(vendor.vendor_name_normalized) > 0
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_rfc_normalized_is_populated(self, mock_read_excel):
        """rfc_normalized is set for each inserted record that has an RFC."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Acme Corp", "RFC": "ACO-850101-XYZ"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            vendor = _all_vendors(db)[0]
            assert vendor.rfc_normalized is not None
            assert len(vendor.rfc_normalized) > 0
            # Normalized RFC should not contain dashes
            assert "-" not in vendor.rfc_normalized
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_duplicate_name_rfc_pairs_are_skipped(self, mock_read_excel):
        """Rows with the same (normalized_name, normalized_rfc) are deduplicated."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Acme Corp", "RFC": "ACO850101XYZ"},
            {"Vendor Name": "Acme Corp", "RFC": "ACO850101XYZ"},
            {"Vendor Name": "Acme Corp", "RFC": "ACO850101XYZ"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                result = refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            assert result["rows_inserted"] == 1
            assert result["rows_skipped"] == 2
            assert len(_all_vendors(db)) == 1
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_empty_dataframe_raises_value_error(self, mock_read_excel):
        """An empty DataFrame raises a ValueError."""
        mock_read_excel.return_value = pd.DataFrame()

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(ValueError, match="empty"):
                    refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            assert len(_all_vendors(db)) == 0
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_missing_columns_raises_value_error(self, mock_read_excel):
        """A DataFrame with unrecognized columns raises a ValueError."""
        mock_read_excel.return_value = _make_vendor_df([
            {"UnknownCol1": "foo", "UnknownCol2": "bar"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(ValueError, match="Could not detect"):
                    refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_corrupt_file_raises_exception(self, mock_read_excel):
        """A corrupt/unreadable Excel file propagates the exception."""
        mock_read_excel.side_effect = Exception("File is corrupt")

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(Exception, match="corrupt"):
                    refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_replace_deletes_existing_records_before_insert(self, mock_read_excel):
        """With replace=True (default), existing records are cleared first."""
        db = TestSession()
        try:
            # Seed the table with a pre-existing record.
            db.add(VendorMaster(
                vendor_name="Old Vendor",
                rfc="OLD000000AAA",
                vendor_name_normalized="OLD VENDOR",
                rfc_normalized="OLD000000AAA",
                source_file="old.xlsx",
            ))
            db.commit()
            assert len(_all_vendors(db)) == 1

            mock_read_excel.return_value = _make_vendor_df([
                {"Vendor Name": "New Vendor", "RFC": "NEW111111BBB"},
            ])

            with patch.object(Path, "exists", return_value=True):
                result = refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH, replace=True)

            vendors = _all_vendors(db)
            assert len(vendors) == 1
            assert vendors[0].vendor_name == "New Vendor"
            assert result["replace"] is True
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_replace_false_keeps_existing_records(self, mock_read_excel):
        """With replace=False, existing records are preserved."""
        db = TestSession()
        try:
            db.add(VendorMaster(
                vendor_name="Old Vendor",
                rfc="OLD000000AAA",
                vendor_name_normalized="OLD VENDOR",
                rfc_normalized="OLD000000AAA",
                source_file="old.xlsx",
            ))
            db.commit()
            assert len(_all_vendors(db)) == 1

            mock_read_excel.return_value = _make_vendor_df([
                {"Vendor Name": "New Vendor", "RFC": "NEW111111BBB"},
            ])

            with patch.object(Path, "exists", return_value=True):
                result = refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH, replace=False)

            vendors = _all_vendors(db)
            assert len(vendors) == 2
            assert result["replace"] is False
        finally:
            db.close()

    def test_file_not_found_raises(self):
        """Passing a nonexistent path raises FileNotFoundError."""
        db = TestSession()
        try:
            with pytest.raises(FileNotFoundError):
                refresh_vendor_master_from_excel(db, "/nonexistent/path/vendors.xlsx")
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_result_dict_contains_expected_keys(self, mock_read_excel):
        """The returned dict has all expected summary keys."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Acme Corp", "RFC": "ACO850101XYZ"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                result = refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            expected_keys = {
                "source_file", "replace", "rows_read",
                "rows_inserted", "rows_skipped", "vendor_col", "rfc_col",
            }
            assert expected_keys == set(result.keys())
            assert result["rows_read"] == 1
            assert result["source_file"] == "fake_vendor_master.xlsx"
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_rows_with_nan_name_and_rfc_are_skipped(self, mock_read_excel):
        """Rows where both vendor name and RFC are NaN/empty are skipped."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "nan", "RFC": "nan"},
            {"Vendor Name": "", "RFC": ""},
            {"Vendor Name": "Valid Vendor", "RFC": "VAL123456AAA"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                result = refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            assert result["rows_inserted"] == 1
            assert result["rows_skipped"] == 2
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    def test_source_file_stored_on_records(self, mock_read_excel):
        """Each inserted record stores the source filename."""
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Acme Corp", "RFC": "ACO850101XYZ"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                refresh_vendor_master_from_excel(db, FAKE_EXCEL_PATH)

            vendor = _all_vendors(db)[0]
            assert vendor.source_file == "fake_vendor_master.xlsx"
        finally:
            db.close()


# ===========================================================================
# 2. bootstrap_vendor_master_if_empty
# ===========================================================================


class TestBootstrapVendorMasterIfEmpty:
    """Tests for bootstrap_vendor_master_if_empty."""

    def test_skips_when_table_has_records(self):
        """When vendor_master already has rows, bootstrap does nothing."""
        db = TestSession()
        try:
            db.add(VendorMaster(
                vendor_name="Existing Vendor",
                rfc="EXI000000AAA",
                vendor_name_normalized="EXISTING VENDOR",
                rfc_normalized="EXI000000AAA",
                source_file="existing.xlsx",
            ))
            db.commit()

            result = bootstrap_vendor_master_if_empty(db)

            assert result["loaded"] is False
            assert result["reason"] == "already_loaded"
        finally:
            db.close()

    @patch("app.services.vendor_master.loader._best_vendor_master_file", return_value=None)
    def test_no_file_found_returns_file_not_found(self, mock_best_file):
        """When no Excel file is found, returns file_not_found without crashing."""
        db = TestSession()
        try:
            result = bootstrap_vendor_master_if_empty(db)

            assert result["loaded"] is False
            assert result["reason"] == "file_not_found"
        finally:
            db.close()

    @patch("app.services.vendor_master.loader.pd.read_excel")
    @patch("app.services.vendor_master.loader._best_vendor_master_file")
    def test_loads_when_table_is_empty(self, mock_best_file, mock_read_excel):
        """When the table is empty and a file exists, it loads the vendors."""
        fake_path = Path("/tmp/Vendor Master BD.xlsx")
        mock_best_file.return_value = fake_path
        mock_read_excel.return_value = _make_vendor_df([
            {"Vendor Name": "Bootstrap Vendor", "RFC": "BOO123456ZZZ"},
        ])

        db = TestSession()
        try:
            with patch.object(Path, "exists", return_value=True):
                result = bootstrap_vendor_master_if_empty(db)

            assert result["loaded"] is True
            assert result["rows_inserted"] == 1
            vendors = _all_vendors(db)
            assert len(vendors) == 1
            assert vendors[0].vendor_name == "Bootstrap Vendor"
        finally:
            db.close()

    @patch("app.services.vendor_master.loader._best_vendor_master_file")
    def test_does_not_crash_on_missing_excel(self, mock_best_file):
        """When _best_vendor_master_file returns None, bootstrap exits gracefully."""
        mock_best_file.return_value = None

        db = TestSession()
        try:
            result = bootstrap_vendor_master_if_empty(db)
            assert result["loaded"] is False
            # Table remains empty
            assert len(_all_vendors(db)) == 0
        finally:
            db.close()
