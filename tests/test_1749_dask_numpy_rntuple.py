# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE
from __future__ import annotations

import numpy as np
import pytest

import uproot

ak = pytest.importorskip("awkward")
pytest.importorskip("dask.array")


def _write(path, offset=0, x_dtype=np.float64):
    data = {
        "x": (np.arange(10) + offset).astype(x_dtype),
        "i": np.arange(10, dtype=np.int32) + offset,
        "a": np.arange(30.0).reshape(10, 3) + offset,
        "j": ak.Array([[1.0 + offset, 2.0], []] * 5),
    }
    with uproot.recreate(path) as output:
        output["ntuple"] = data  # writes an RNTuple


def _assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if expected.dtype == object:
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, e)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_numpy_rntuple(tmp_path, open_files):
    path = tmp_path / "rntuple.root"
    _write(path)
    ntuple = uproot.open(path)["ntuple"]

    arrays = uproot.dask(
        f"{path}:ntuple", library="np", open_files=open_files, steps_per_file=3
    )
    assert sorted(arrays) == ["a", "i", "j", "x"]
    for key, array in arrays.items():
        expected = ntuple[key].array(library="np")
        assert array.dtype == expected.dtype
        assert array.shape[1:] == expected.shape[1:]
        if open_files:
            assert array.shape == expected.shape
        _assert_same(array.compute(), expected)

    np.testing.assert_array_equal(
        arrays["a"].sum(axis=1).compute(), ntuple["a"].array(library="np").sum(axis=1)
    )


@pytest.mark.parametrize("step_size", [3, "1 kB"])
def test_dask_numpy_rntuple_multiple_files(tmp_path, step_size):
    files = {}
    for index in range(2):
        path = tmp_path / f"rntuple-{index}.root"
        _write(path, offset=index)
        files[path] = "ntuple"

    arrays = uproot.dask(files, library="np", step_size=step_size)
    expected = np.concatenate([np.arange(10.0), np.arange(10.0) + 1])
    np.testing.assert_array_equal(arrays["x"].compute(), expected)
    assert arrays["a"].compute().shape == (20, 3)


@pytest.mark.parametrize("open_files", [True, False])
def test_dask_numpy_rntuple_inconsistent_dtypes(tmp_path, open_files):
    first = tmp_path / "first.root"
    different = tmp_path / "different.root"
    _write(first)
    _write(different, x_dtype=np.float32)
    files = {first: "ntuple", different: "ntuple"}

    if open_files:
        with pytest.raises(
            ValueError, match=r"inconsistent NumPy dtype.*'x'.*different\.root"
        ):
            uproot.dask(files, library="np", filter_name="x", open_files=True)
    else:
        array = uproot.dask(files, library="np", filter_name="x", open_files=False)
        with pytest.raises(
            ValueError, match=r"inconsistent NumPy dtype.*'x'.*different\.root"
        ):
            array["x"].compute()
