from app.services.vendor_master.matcher import (
    VendorMasterEntry,
    VendorMasterResolver,
    canonicalize_rfc,
    core_vendor_name,
    normalize_vendor_name_for_match,
)


def test_name_normalization_removes_noise_prefixes():
    raw = "confirma su acuerdo con GRUPO CATANA SA DE CV a quien en lo sucesivo"
    assert normalize_vendor_name_for_match(raw) == "GRUPO CATANA SA DE CV"


def test_core_name_removes_legal_tokens():
    normalized = normalize_vendor_name_for_match("GRUPO CATANA SA DE CV")
    assert core_vendor_name(normalized) == "GRUPO CATANA"


def test_fill_name_from_rfc():
    resolver = VendorMasterResolver(
        [
            VendorMasterEntry(
                vendor_name="GRUPO CATANA SA DE CV",
                rfc="GCA001122ABC",
                vendor_name_normalized="GRUPO CATANA SA DE CV",
                vendor_name_core="GRUPO CATANA",
            )
        ]
    )
    rfc, name = resolver.fill_missing_fields("GCA-001122-ABC", None)
    assert rfc == "GCA001122ABC"
    assert name == "GRUPO CATANA SA DE CV"


def test_fill_rfc_from_fuzzy_name():
    resolver = VendorMasterResolver(
        [
            VendorMasterEntry(
                vendor_name="GRUPO CATANA SA DE CV",
                rfc="GCA001122ABC",
                vendor_name_normalized="GRUPO CATANA SA DE CV",
                vendor_name_core="GRUPO CATANA",
            )
        ]
    )
    rfc, name = resolver.fill_missing_fields(None, "GRUPO CATANA SA")
    assert rfc == "GCA001122ABC"
    assert name == "GRUPO CATANA SA"


def test_canonicalize_foreign_tax_id_keeps_hyphenated_format():
    assert canonicalize_rfc("75-2218815") == "75-2218815"

