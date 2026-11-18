"""
Simple tests for functions in metrics.py

Each test builds a tiny DataFrame (already sorted by predicted score, which is
what both functions assume) and compares the returned value against a number
worked out by hand. This checks that the functions run without errors AND that
the math is correct.

Run with:   pytest -v test_metrics.py
Or plainly: python test_metrics.py
"""

import math
import pandas as pd
import pytest
from metrics import enrichment_factor, apk, precision_k, f1_k, avg_rank, best_rank, top_recall, new_ndcg, diversity


def make_df(labels):
    """Helper: build a ranked DataFrame from a list of 0/1 activity labels.
    Index 0 = top-ranked prediction."""
    return pd.DataFrame({"activity_binary": labels})


# enrichment_factor()


def test_ef_perfect_ranking():
    # 100 variants, 10 active, and all 10 actives sit at the very top.
    # top 10% = 10 rows, all 10 are hits.
    # EF = (hits_top/n_top) / (n_active/n_total) = (10/10) / (10/100) = 10.0
    labels = [1] * 10 + [0] * 90
    df = make_df(labels)
    assert enrichment_factor(df, k=10) == pytest.approx(10.0)


def test_ef_random_like():
    # 100 compounds, 20 active. Top 10 rows contain exactly 2 actives.
    # EF = (2/10) / (20/100) = 0.2 / 0.2 = 1.0  -> no enrichment over random.
    labels = ([1, 1] + [0] * 8) + ([1] * 18 + [0] * 72)
    df = make_df(labels)
    assert df["activity_binary"].sum() == 20  # sanity check on the fixture
    assert enrichment_factor(df, k=10) == pytest.approx(1.0)


def test_ef_no_hits_in_top_returns_zero():
    # No actives in the selected top fraction -> function returns 0.0
    labels = [0] * 10 + [1] * 90
    df = make_df(labels)
    assert enrichment_factor(df, k=10) == 0.0


def test_ef_missing_column_raises():
    df = pd.DataFrame({"something_else": [1, 0, 1]})
    with pytest.raises(RuntimeError):
        enrichment_factor(df, k=10)


def test_ef_none_dataframe_raises():
    with pytest.raises(RuntimeError):
        enrichment_factor(None, k=10)


# apk()  (Average Precision at k)

def test_apk_all_top_k_active():
    # Top 3 are all active and there are exactly 3 actives in total.
    # precisions at each hit: 1/1, 2/2, 3/3 = 1 + 1 + 1 = 3
    # AP@k = 3 / (total actives = 3) = 1.0
    df = make_df([1, 1, 1, 0, 0])
    assert apk(df, k=3) == pytest.approx(1.0)


def test_apk_hand_computed_mixed_ranking():
    # ranking = [1,0,1,0,1], k=5, total actives = 3
    # hits at positions 1, 3, 5:
    #   pos 1 -> 1/1
    #   pos 3 -> 2/3
    #   pos 5 -> 3/5
    # sum = 1 + 0.6666... + 0.6 = 2.26666...
    # AP@k = 2.26666... / 3 = 0.755555...
    df = make_df([1, 0, 1, 0, 1])
    expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert apk(df, k=5) == pytest.approx(expected)


def test_apk_k_smaller_than_list():
    # Only the top k=2 rows are considered.
    # ranking top2 = [1, 0]; hit at pos 1 -> 1/1 = 1; sum = 1
    # total actives in whole df = 2  ->  AP@k = 1/2 = 0.5
    df = make_df([1, 0, 1, 0])
    assert apk(df, k=2) == pytest.approx(0.5)


def test_apk_no_actives_returns_zero():
    df = make_df([0, 0, 0, 0])
    assert apk(df, k=4) == 0.0


def test_apk_missing_column_raises():
    df = pd.DataFrame({"score": [0.9, 0.5, 0.1]})
    with pytest.raises(RuntimeError):
        apk(df, k=3)


def test_apk_none_dataframe_raises():
    with pytest.raises(RuntimeError):
        apk(None, k=3)




def make_round_df(activity, threshold_hit, variants=None):
    """Helper: build a round DataFrame as load_dataset would return it.
    Rows are already sorted by predicted score (index 0 = top prediction);
    activity_binary and true_rank are derived from the activity values."""
    df = pd.DataFrame({"activity": activity})
    df["activity_binary"] = (df["activity"] >= threshold_hit).astype(int)
    df["true_rank"] = df["activity"].rank(ascending=False, method="first").astype(int)
    if variants is None:
        variants = [f"A{i + 1}G" for i in range(len(df))]
    df["variant"] = variants
    return df
# precision_k()

def test_precision_perfect_ranking():
    # top 3 are all hits -> 3/3
    df = make_round_df([0.9, 0.8, 0.5, 0.2, 0.1], threshold_hit=0.5)
    assert precision_k(df, k=3) == pytest.approx(1.0)


def test_precision_mixed_ranking():
    # top 3 binary = [1, 0, 1] -> 2/3
    df = make_round_df([0.9, 0.2, 0.8, 0.1, 0.5], threshold_hit=0.5)
    assert precision_k(df, k=3) == pytest.approx(2 / 3)


def test_precision_no_hits_in_top_returns_zero():
    df = make_round_df([0.1, 0.2, 0.9, 0.8], threshold_hit=0.5)
    assert precision_k(df, k=2) == 0.0


# top_recall()  (current definition: hits in top k / hits in the whole pool)

def test_top_recall_mixed_ranking():
    # 3 hits in total, 2 of them in the top 3 -> 2/3
    df = make_round_df([0.9, 0.2, 0.8, 0.1, 0.5], threshold_hit=0.5)
    assert top_recall(df, k=3) == pytest.approx(2 / 3)


def test_top_recall_k_covers_whole_pool():
    # k = len(df): every hit is in the top k -> 1.0
    df = make_round_df([0.9, 0.2, 0.8, 0.1, 0.5], threshold_hit=0.5)
    assert top_recall(df, k=5) == pytest.approx(1.0)


# f1_k()

def test_f1_hand_computed():
    # k=2: precision = 1/2 (top 2 binary = [1, 0]); recall = 1/3 (1 of 3 hits)
    # F1 = 2 * (1/2 * 1/3) / (1/2 + 1/3) = (1/3) / (5/6) = 2/5
    df = make_round_df([0.9, 0.2, 0.8, 0.1, 0.5], threshold_hit=0.5)
    assert f1_k(df, k=2) == pytest.approx(0.4)


def test_f1_no_hits_in_top_returns_zero():
    # precision = 0 and recall = 0 -> guarded 0.0, no division by zero
    df = make_round_df([0.1, 0.2, 0.9, 0.8], threshold_hit=0.5)
    assert f1_k(df, k=2) == 0.0


# avg_rank()

def test_avg_rank_perfect_ranking():
    # top 3 true ranks = [1, 2, 3] -> mean 2.0 = (k+1)/2
    df = make_round_df([0.9, 0.8, 0.5, 0.2, 0.1], threshold_hit=0.5)
    assert avg_rank(df, k=3) == pytest.approx(2.0)


def test_avg_rank_mixed_ranking():
    # top 3 true ranks = [1, 4, 2] -> 7/3
    df = make_round_df([0.9, 0.2, 0.8, 0.1, 0.5], threshold_hit=0.5)
    assert avg_rank(df, k=3) == pytest.approx(7 / 3)


# best_rank()

def test_best_rank_winner_in_top_k():
    # top 3 true ranks = [1, 4, 2] -> 1
    df = make_round_df([0.9, 0.2, 0.8, 0.1, 0.5], threshold_hit=0.5)
    assert best_rank(df, k=3) == 1


def test_best_rank_winner_outside_top_k():
    # true ranks in prediction order = [3, 2, 1]; top 2 -> min(3, 2) = 2
    df = make_round_df([0.2, 0.5, 0.9], threshold_hit=0.5)
    assert best_rank(df, k=2) == 2


# diversity()  (number of distinct mutated positions in the top k)

def test_diversity_repeated_position():
    # top 3 variants A1G, A1C, A2G -> positions {1, 2} -> 2
    df = make_round_df([0.9, 0.8, 0.7, 0.6], threshold_hit=0.5,
                       variants=["A1G", "A1C", "A2G", "A3G"])
    assert diversity(df, k=3) == 2
    assert diversity(df, k=4) == 3


def test_diversity_multi_mutant():
    # A1G_A2C counts two positions -> {1, 2, 3} -> 3
    df = make_round_df([0.9, 0.8], threshold_hit=0.5, variants=["A1G_A2C", "A3G"])
    assert diversity(df, k=2) == 3


# new_ndcg()

def test_new_ndcg_perfect_ranking():
    # prediction order == ideal order -> DCG == IDCG -> 1.0
    df = make_round_df([0.9, 0.8, 0.5, 0.2, 0.1], threshold_hit=0.5)
    assert new_ndcg(df, threshold_hit=0.5, k=3) == pytest.approx(1.0)


def test_new_ndcg_reversed_ranking():
    # min-max scaled activities: 0.1->0, 0.2->0.125, 0.5->0.5, 0.8->0.875, 0.9->1
    # gains below the threshold (0.1, 0.2) are zeroed
    # ideal top 3 gains = [1, 0.875, 0.5]; predicted top 3 gains = [0, 0, 0.5]
    df = make_round_df([0.1, 0.2, 0.5, 0.8, 0.9], threshold_hit=0.5)
    idcg = 1 / math.log2(2) + 0.875 / math.log2(3) + 0.5 / math.log2(4)
    dcg = 0.5 / math.log2(4)
    assert new_ndcg(df, threshold_hit=0.5, k=3) == pytest.approx(dcg / idcg)



# Allow running without pytest installed as a runner.
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
