from vdriftbench.category_map import CATEGORY_MACRO_MAP, to_macro


def test_all_37_raw_categories_are_mapped():
    assert len(CATEGORY_MACRO_MAP) == 37


def test_known_category_maps_to_expected_macro():
    assert to_macro("历史(高威胁)") == "历史类"
    assert to_macro("科学/伪科学") == "科学与超现实类"


def test_unmapped_category_falls_back_without_crashing():
    assert to_macro("某个从未见过的新category") == "综合与极端攻击类"


def test_macro_categories_count_is_eight():
    assert len(set(CATEGORY_MACRO_MAP.values())) == 8
