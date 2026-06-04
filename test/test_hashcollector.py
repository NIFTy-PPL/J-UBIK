import json

import pytest

import jubik.hashcollector as hc


def test_get_git_hash_from_local_package_editable_success(tmp_path, monkeypatch):
    package_name = "mypkg"
    package_dir = tmp_path / package_name
    package_dir.mkdir()

    source_dir = tmp_path / "source"
    source_pkg_dir = source_dir / package_name
    source_pkg_dir.mkdir(parents=True)
    (source_dir / ".git").mkdir()

    editable_file = package_dir / f"{package_name}_editable.py"
    editable_file.write_text(f"MAPPING = {{'{package_name}': '{source_pkg_dir}'}}\n")

    monkeypatch.setattr(hc.resources, "files", lambda name: package_dir)

    def fake_check_output(cmd, cwd=None):
        assert cmd == ["git", "rev-parse", "HEAD"]
        assert cwd == str(source_dir)
        return b"abc123\n"

    monkeypatch.setattr(hc.subprocess, "check_output", fake_check_output)

    res = hc._get_git_hash_from_local_package(package_name)
    assert res == "abc123"


def test_get_git_hash_from_local_package_falls_back_to_git_path(tmp_path, monkeypatch):
    package_name = "mypkg"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    git_repo = tmp_path / "repo"
    git_repo.mkdir()

    monkeypatch.setattr(hc.resources, "files", lambda name: package_dir)
    monkeypatch.setattr(hc.os, "listdir", lambda path: [])

    def fake_check_output(cmd, cwd=None):
        assert cmd == ["git", "rev-parse", "HEAD"]
        assert cwd == str(git_repo)
        return b"def456\n"

    monkeypatch.setattr(hc.subprocess, "check_output", fake_check_output)

    res = hc._get_git_hash_from_local_package(package_name, git_path=str(git_repo))
    assert res == "def456"


def test_get_git_hash_from_local_package_package_missing_raises(monkeypatch):
    def missing_package(_):
        raise ModuleNotFoundError

    monkeypatch.setattr(hc.resources, "files", missing_package)

    with pytest.raises(FileNotFoundError, match="Package 'missing' not found"):
        hc._get_git_hash_from_local_package("missing")


def test_get_git_hash_from_local_package_mapping_missing_key_raises(tmp_path, monkeypatch):
    package_name = "mypkg"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    editable_file = package_dir / f"{package_name}_editable.py"
    editable_file.write_text("MAPPING = {'otherpkg': '/tmp/source'}\n")

    monkeypatch.setattr(hc.resources, "files", lambda name: package_dir)

    with pytest.raises(KeyError, match="not in mapping"):
        hc._get_git_hash_from_local_package(package_name)


def test_get_git_hash_from_local_package_bad_mapping_literal_raises(tmp_path, monkeypatch):
    package_name = "mypkg"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    editable_file = package_dir / f"{package_name}_editable.py"
    editable_file.write_text("MAPPING = not_a_literal\n")

    monkeypatch.setattr(hc.resources, "files", lambda name: package_dir)

    with pytest.raises(ValueError, match="Could not evaluate the value of 'MAPPING'"):
        hc._get_git_hash_from_local_package(package_name)


def test_save_local_packages_hashes_to_txt_writes_json(tmp_path, monkeypatch):
    def fake_get_hash(package_name, git_path=None):
        return f"hash-{package_name}-{git_path}"

    monkeypatch.setattr(hc, "_get_git_hash_from_local_package", fake_get_hash)

    filename = tmp_path / "hashes.json"
    hc.save_local_packages_hashes_to_txt(
        packages_names=["pkg_a", "pkg_b"],
        filename=str(filename),
        paths_to_git=["/repo/a", None],
        verbose=False,
    )

    payload = json.loads(filename.read_text())
    assert payload == {
        "pkg_a": "hash-pkg_a-/repo/a",
        "pkg_b": "hash-pkg_b-None",
    }


def test_save_local_packages_hashes_to_txt_propagates_errors(tmp_path, monkeypatch):
    def fake_get_hash(package_name, git_path=None):
        raise FileNotFoundError(f"missing {package_name}")

    monkeypatch.setattr(hc, "_get_git_hash_from_local_package", fake_get_hash)

    with pytest.raises(FileNotFoundError, match="missing pkg_a"):
        hc.save_local_packages_hashes_to_txt(
            packages_names=["pkg_a"],
            filename=str(tmp_path / "hashes.json"),
            paths_to_git=None,
            verbose=False,
        )
