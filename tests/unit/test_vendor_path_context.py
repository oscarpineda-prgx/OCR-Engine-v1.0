from app.services.vendor_master.matcher import (
    VendorMasterEntry,
    VendorMasterResolver,
    core_vendor_name,
    normalize_vendor_name_for_match,
)
from app.services.vendor_master.path_context import (
    resolve_vendor_from_path,
    vendor_name_candidates_from_path,
)


def _entry(vendor_name: str, rfc: str) -> VendorMasterEntry:
    normalized = normalize_vendor_name_for_match(vendor_name)
    return VendorMasterEntry(
        vendor_name=vendor_name,
        rfc=rfc,
        vendor_name_normalized=normalized,
        vendor_name_core=core_vendor_name(normalized),
    )


def test_resolve_vendor_from_parent_folder_name():
    resolver = VendorMasterResolver(
        [
            _entry("BACARDI Y COMPANIA SA DE CV", "BAC850101ABC"),
        ]
    )

    match = resolve_vendor_from_path(
        r"\\amer.prgx.com\images\OxxoMex\Link 2026\PROCESO ALTAS\2015\BACARDI Y COMPANIA SA DE\0000000300018_292728_14333_300315_2.pdf",
        resolver,
    )

    assert match is not None
    assert match.rfc == "BAC850101ABC"
    assert match.nombre_proveedor == "BACARDI Y COMPANIA SA DE CV"
    assert match.candidate == "BACARDI Y COMPANIA SA DE"
    assert match.strategy.startswith("path_name:")


def test_rejects_generic_link2026_path_segments():
    candidates = vendor_name_candidates_from_path(
        r"\\amer.prgx.com\images\OxxoMex\Link 2026\PROCESO ALTAS\2015\000123.pdf"
    )

    assert candidates == []


def test_resolve_vendor_from_rfc_path_segment():
    resolver = VendorMasterResolver(
        [
            _entry("PROVEEDOR RFC SA DE CV", "ABC010101AB1"),
        ]
    )

    match = resolve_vendor_from_path(
        r"\\amer.prgx.com\images\OxxoMex\Link 2026\ABC010101AB1\archivo.pdf",
        resolver,
    )

    assert match is not None
    assert match.rfc == "ABC010101AB1"
    assert match.nombre_proveedor == "PROVEEDOR RFC SA DE CV"
    assert match.strategy == "path_rfc"
