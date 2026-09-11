"""The public analysis uses the paper's direction and subject-level aggregation."""
import numpy as np
import pytest

from evals.neuroprobe.paper_results import paired_p, subject_means


@pytest.mark.parametrize('difference,expected', [(1., 1 / 64), (-1., 1.), (0., 1.)])
def test_exact_one_sided_test(difference, expected):
    assert paired_p(np.full(6, difference), np.zeros(6)) == expected


def test_subject_mean_combines_sessions_and_tasks():
    grid = {('a', 'btbank7_0'): .5, ('b', 'btbank7_0'): .7,
            ('a', 'btbank7_1'): .7, ('b', 'btbank7_1'): .9,
            ('a', 'btbank10_0'): .4, ('b', 'btbank10_0'): .6}
    assert subject_means(grid) == {7: .7, 10: .5}
