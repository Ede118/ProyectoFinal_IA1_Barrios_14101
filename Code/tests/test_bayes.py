import numpy as np

from Code.Statistics.BayesAgent import BayesAgent


def test_posterior_and_decide():
    agent = BayesAgent(K=3, C=4)
    pi = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    likelihood = np.array(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.3, 0.3, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    counts = np.array([3, 0, 1, 0], dtype=np.int64)

    posterior = agent.posterior(pi, likelihood, counts)
    assert posterior.shape == (3,)
    assert np.isclose(posterior.sum(), 1.0, atol=1e-6)

    decision = agent.decide(posterior, labels=["H1", "H2", "H3"])
    assert decision == "H2"

    tie_decision = agent.decide(np.array([0.5, 0.5, 0.0], dtype=np.float32), labels=["A", "B", "C"])
    assert tie_decision == "A"
