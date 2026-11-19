"""Package cache identity on explicit synthetic filesystem inventories."""

import hashlib
import json
import os
import pytest

from ensemble_dwd import _cache_identity as identity


@pytest.fixture
def roots(tmp_path):
    result = {name: (tmp_path / ("root " + name)).resolve()
              for name in ("ensemble_dwd", "dwd")}
    for root in result.values():
        root.mkdir()
        (root / "__init__.py").write_bytes(b"synthetic package source")
    return result


def put(root, relative, data=b"synthetic artifact"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def assert_inventory(roots, expected_keys):
    """Use declared fixture paths and os.stat, without a second traversal."""
    expected = {}
    for key in expected_keys:
        package, _, relative = key.partition("/")
        path = roots[package] / relative
        info = os.stat(path)
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        expected[key] = (path, stamp)
    observed = identity._artifact_files(roots)
    assert observed == expected
    assert list(observed) == expected_keys
    return observed


# Expected keys and order are declared explicitly, independently of the package
# traversal/filter code. Inputs include ignored entries and mixed-case suffixes.
@pytest.mark.parametrize("created,expected_dwd", [
    (
        ("z.py", "a.py", "core.c", "core.h"),
        ["__init__.py", "a.py", "core.c", "core.h", "z.py"],
    ),
    (
        ("z/last.py", "a/z.py", "a/a.py", "a/deep/item.h", "looks.py/inside.py"),
        ["__init__.py", "a/a.py", "a/z.py", "a/deep/item.h", "looks.py/inside.py", "z/last.py"],
    ),
    (
        ("note.txt", "data.json", "binary.dat", "no_extension", ".py", ".so",
         "only_notes/note.txt", "__pycache__/excluded.py",
         "nested/__pycache__/excluded.so", "nested/kept.py"),
        ["__init__.py", "nested/kept.py"],
    ),
    (
        ("a.PY", "c.C", "h.H", "native.PYD", "native.DLL", "native.dylib",
         "module.cpython-312-x86_64-linux-gnu.so", "lib.so.2.0", ".so.7",
         "strange.so.part.py", "not.software", "directory.so.names/leaf.py"),
        [".so.7", "__init__.py", "a.PY", "c.C", "h.H", "lib.so.2.0",
         "module.cpython-312-x86_64-linux-gnu.so", "native.DLL", "native.PYD",
         "native.dylib", "strange.so.part.py", "directory.so.names/leaf.py"],
    ),
    (
        ("space folder/space file.py", "omega_\u03a9/caf\u00e9.py"),
        ["__init__.py", "omega_\u03a9/caf\u00e9.py", "space folder/space file.py"],
    ),
], ids=("flat", "nested", "irrelevant", "native_names", "unicode_and_spaces"))
def test_exact_keys_paths_stamps_and_order(roots, created, expected_dwd):
    for name in created:
        put(roots["dwd"], name)
    (roots["ensemble_dwd"] / "empty").mkdir()
    expected = ["dwd/" + name for name in expected_dwd] + ["ensemble_dwd/__init__.py"]
    assert_inventory(roots, expected)


@pytest.mark.skipif(os.name != "posix", reason="These literal basenames are POSIX-specific")
def test_posix_tab_and_backslash_basenames_keep_lexical_keys(roots):
    for name in ("tab\tname.py", "back\\slash.py", "nested/back\\slash.so.2"):
        put(roots["dwd"], name)
    assert_inventory(roots, [
        "dwd/__init__.py", "dwd/back\\slash.py", "dwd/tab\tname.py",
        "dwd/nested/back\\slash.so.2", "ensemble_dwd/__init__.py",
    ])


def test_snapshot_content_and_native_digests_match_fixture_bytes(roots):
    names = ("source.py", "nested/helper.c", "nested/helper.h", "native.so.3", "native.pyd")
    for name in names:
        put(roots["dwd"], name, data=("fixture: " + name).encode())
    keys = ["dwd/__init__.py", "dwd/native.pyd", "dwd/native.so.3", "dwd/source.py",
            "dwd/nested/helper.c", "dwd/nested/helper.h", "ensemble_dwd/__init__.py"]
    expected_files = assert_inventory(roots, keys)
    digests = {key: hashlib.sha256(path.read_bytes()).hexdigest()
               for key, (path, _) in expected_files.items()}
    expected_digest = hashlib.sha256(json.dumps(
        digests, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    snapshot = identity._ArtifactSnapshot(roots, anchor=True)
    assert snapshot.check() == expected_digest
    assert snapshot.files == expected_files
    assert snapshot.digests == digests
    assert snapshot.native_digests() == {
        "dwd/native.pyd": digests["dwd/native.pyd"],
        "dwd/native.so.3": digests["dwd/native.so.3"],
    }
    assert snapshot.check() == expected_digest


@pytest.mark.parametrize("change", ("add_source", "remove_source", "replace_native", "rename_native", "edit_source"))
def test_real_inventory_changes_refuse_and_stay_failed(roots, change):
    source = put(roots["dwd"], "nested/source.py", b"old source")
    native = put(roots["dwd"], "nested/native.so.2", b"old native")
    snapshot = identity._ArtifactSnapshot(roots)
    snapshot.check()
    if change == "add_source":
        put(roots["dwd"], "new.py", b"new source")
    elif change == "remove_source":
        source.unlink()
    elif change == "replace_native":
        replacement = put(roots["dwd"], "replacement.tmp", b"replacement native bytes")
        os.replace(replacement, native)
    elif change == "rename_native":
        native.rename(native.with_name("renamed.so.2"))
    else:
        source.write_bytes(b"changed source with different size")
    with pytest.raises(identity._CacheIdentityError, match="changed"):
        snapshot.check()
    with pytest.raises(identity._CacheIdentityError, match="changed"):
        snapshot.check()


def test_empty_irrelevant_and_excluded_folders_do_not_change_identity(roots):
    put(roots["dwd"], "nested/source.py")
    snapshot = identity._ArtifactSnapshot(roots)
    original = snapshot.check()
    (roots["dwd"] / "empty").mkdir()
    put(roots["dwd"], "only_notes/readme.txt")
    put(roots["dwd"], "__pycache__/ignored.py")
    assert_inventory(roots, ["dwd/__init__.py", "dwd/nested/source.py", "ensemble_dwd/__init__.py"])
    assert snapshot.check() == original


def test_import_anchor_detects_prior_source_mutation(roots):
    source = put(roots["dwd"], "source.py", b"before anchor")
    snapshot = identity._ArtifactSnapshot(roots, anchor=True)
    source.write_bytes(b"after anchor with changed size")
    with pytest.raises(identity._CacheIdentityError, match="since cache module import"):
        snapshot.check()


def test_change_during_first_hash_refuses(roots, monkeypatch):
    target = put(roots["dwd"], "source.py", b"first bytes")
    read = identity._file_digest

    def changed(path):
        digest = read(path)
        if path == target:
            path.write_bytes(b"changed after read, before second observation")
        return digest

    monkeypatch.setattr(identity, "_file_digest", changed)
    snapshot = identity._ArtifactSnapshot(roots)
    with pytest.raises(identity._CacheIdentityError, match="during identification"):
        snapshot.check()
    assert snapshot.digest is None


def test_selected_file_stat_failure_is_not_hidden(roots, monkeypatch):
    target = put(roots["dwd"], "selected.py")
    original = identity._stamp

    def failed(path):
        if path == target:
            raise PermissionError("selected artifact denied")
        return original(path)

    monkeypatch.setattr(identity, "_stamp", failed)
    with pytest.raises(PermissionError, match="selected artifact denied"):
        identity._artifact_files(roots)
    snapshot = identity._ArtifactSnapshot(roots)
    with pytest.raises(identity._CacheIdentityError, match="unreadable"):
        snapshot.check()


def test_directory_listing_errors_propagate_through_onerror(roots, monkeypatch):
    def failed_walk(root, *, onerror):
        onerror(PermissionError("directory listing denied"))
        yield  # This function must be a generator, like os.walk.

    monkeypatch.setattr(identity.os, "walk", failed_walk)
    with pytest.raises(PermissionError, match="directory listing denied"):
        identity._artifact_files(roots)


@pytest.mark.parametrize("failure", ("missing_root", "missing_init"))
def test_required_package_identity_refuses(roots, failure):
    if failure == "missing_root":
        (roots["dwd"] / "__init__.py").unlink()
        roots["dwd"].rmdir()
    else:
        (roots["dwd"] / "__init__.py").unlink()
    with pytest.raises(identity._CacheIdentityError):
        identity._ArtifactSnapshot(roots).check()


def make_symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip("This host does not permit this real symlink fixture: " + str(error))


@pytest.mark.parametrize("kind", ("file", "directory", "broken_file"))
def test_lexical_symlink_inventory_behavior_where_permitted(roots, tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "directory":
        put(outside, "not_traversed.py")
        make_symlink_or_skip(roots["dwd"] / "linked", outside, directory=True)
        assert_inventory(roots, ["dwd/__init__.py", "ensemble_dwd/__init__.py"])
    elif kind == "broken_file":
        make_symlink_or_skip(roots["dwd"] / "broken.py", outside / "missing.py")
        with pytest.raises(identity._CacheIdentityError):
            identity._ArtifactSnapshot(roots).check()
    else:
        target = put(outside, "source.py", b"first target")
        link = roots["dwd"] / "alias.py"
        make_symlink_or_skip(link, target)
        observed = assert_inventory(roots, ["dwd/__init__.py", "dwd/alias.py", "ensemble_dwd/__init__.py"])
        assert observed["dwd/alias.py"][0] == link
        snapshot = identity._ArtifactSnapshot(roots)
        snapshot.check()
        target.write_bytes(b"new target with a different size")
        with pytest.raises(identity._CacheIdentityError, match="changed"):
            snapshot.check()
