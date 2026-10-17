from scripts.diff_utils import invert_unified_diff


def test_invert_swe_smith_bug_patch_restores_removed_initialization() -> None:
    bug_patch = """diff --git a/src/x.py b/src/x.py
index a5b6092..72d902a 100644
--- a/src/x.py
+++ b/src/x.py
@@ -10,7 +10,6 @@ def make_names_unique(database):
-    converter = LongNamesConverter()
 
     for node in database.nodes:
         name = converter.convert(node.name)
"""
    repair = invert_unified_diff(bug_patch)
    assert "index 72d902a..a5b6092 100644" in repair
    assert "--- a/src/x.py\n+++ b/src/x.py" in repair
    assert "@@ -10,6 +10,7 @@" in repair
    assert "+    converter = LongNamesConverter()" in repair
    assert "-    converter = LongNamesConverter()" not in repair


def test_invert_unified_diff_is_an_involution_for_text_patch() -> None:
    patch = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
-old
+new
 keep
"""
    assert invert_unified_diff(invert_unified_diff(patch)) == patch


def test_invert_unified_diff_canonicalizes_rename_direction() -> None:
    patch = """diff --git a/old.py b/new.py
similarity index 95%
rename from old.py
rename to new.py
--- a/old.py
+++ b/new.py
@@ -1 +1 @@
-old
+new
"""
    repair = invert_unified_diff(patch)
    assert "diff --git a/new.py b/old.py" in repair
    assert "rename from new.py\nrename to old.py" in repair
    assert "--- a/new.py\n+++ b/old.py" in repair
    assert invert_unified_diff(repair) == patch


def test_invert_unified_diff_rejects_binary_patch() -> None:
    patch = """diff --git a/a.png b/a.png
GIT binary patch
literal 1
A
"""
    try:
        invert_unified_diff(patch)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("binary patch was not rejected")
