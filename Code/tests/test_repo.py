import numpy as np

from Code.adapters.Repositorio import Repo


def test_save_load_models(tmp_path):
    repo = Repo(tmp_path)
    repo.ensure_layout()

    centroids = np.random.rand(3, 2).astype(np.float32)
    repo.save_kmeans("toy", centroids)
    data = repo.load_kmeans("toy")
    assert "C" in data
    np.testing.assert_array_equal(data["C"], centroids)
    assert data["C"].dtype == np.float32

    X = np.random.rand(4, 5).astype(np.float32)
    y = np.array(["a", "b", "c", "d"])
    repo.save_knn("tiny", X, y)
    knn_data = repo.load_knn("tiny")
    np.testing.assert_array_equal(knn_data["X"], X)
    assert knn_data["X"].dtype == np.float32
    assert knn_data["y"].dtype.kind in {"U", "S"}
    np.testing.assert_array_equal(knn_data["y"], y)
