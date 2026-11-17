"""First-k label reconstruction must survive different DuckDB aggregate input orders."""
import itertools

import duckdb
import numpy as np
import pandas as pd
import pytest

from pmgraph.firstk import check_population_labels, label_bets
from pmgraph.labels import bets_from_sides, skill_test


def _boundary_sides(order):
    qty = np.array([.1, .2, .8, .3, .6])
    sign = np.array([1., 1., 1., -1., -1.])
    return pd.DataFrame(dict(wallet=0, market_id=31, event_id="e31",
                             ts=pd.Timestamp("2025-07-03", tz="UTC"), sgn=sign[list(order)],
                             qty=qty[list(order)], price=.5, w=1))


def _bets(sides):
    with duckdb.connect() as con:
        con.register("s", sides)
        return bets_from_sides(con, "SELECT * FROM s", pd.DataFrame({"b": [0], "pi": [.5]}))


def test_position_at_direction_threshold_is_kept_for_every_fill_order():
    # Long shares 1.1, short shares .9: mathematical net/gross direction is exactly .1.
    for order in itertools.permutations(range(5)):
        b = _bets(_boundary_sides(order))
        assert len(b) == 1, f"a true boundary bet was dropped for order {order}"
        np.testing.assert_allclose(b.A, [.1], atol=1e-14)


def test_position_below_direction_threshold_is_still_dropped():
    sides = _boundary_sides(range(5))
    sides.loc[4, "qty"] += 1e-10
    assert _bets(sides).empty


def _first_k_labels(order):
    # One observed market, 30 ordinary future bets and one boundary future bet. Either 30 or
    # 31 bets qualifies, but dropping the boundary bet changes the saved T and fails step 12.
    other = pd.DataFrame(dict(wallet=0, market_id=np.arange(31),
                              event_id=[f"e{i}" for i in range(31)],
                              ts=pd.date_range("2025-07-01", periods=31, freq="h", tz="UTC"),
                              sgn=1., qty=1., price=.5, w=np.arange(31) % 2))
    sides = pd.concat([other, _boundary_sides(order)], ignore_index=True).rename(columns={"wallet": "nid"})
    sides["long_"] = sides.sgn > 0
    with duckdb.connect() as con:
        con.register("sides", sides)
        con.execute("CREATE TEMP TABLE ls AS SELECT * FROM sides")
        con.execute("CREATE TEMP TABLE obs AS SELECT nid, market_id, ts, 1 rn FROM ls WHERE market_id=0")
        con.execute("CREATE TEMP TABLE gm(nid BIGINT, market_id BIGINT)")
        b = label_bets(con, 1, pd.DataFrame({"b": [0], "pi": [.5]}))
        return skill_test(b)


def test_first_k_reconstructed_labels_match_saved_labels_across_fill_orders():
    saved = _first_k_labels([0, 1, 2, 3, 4])
    rebuilt = _first_k_labels([0, 4, 3, 1, 2])
    assert saved.n_bets.iloc[0] == rebuilt.n_bets.iloc[0] == 31
    np.testing.assert_allclose(rebuilt["T"], saved["T"], atol=1e-9)


def test_label_consistency_passes_for_reordered_wallets_and_small_roundoff(tmp_path):
    saved = pd.DataFrame({"T": [.1, -.2], "n_bets": [30, 40]}, index=[5, 8])
    rebuilt = saved.iloc[::-1].copy()
    rebuilt["T"] += 1e-12
    report = tmp_path / "mismatch.csv"
    check_population_labels(rebuilt, saved, 1, report)
    assert not report.exists()


def test_label_consistency_reports_missing_values_counts_and_large_changes(tmp_path):
    saved = pd.DataFrame({"T": [.1, -.2, .0], "n_bets": [30, 40, 50]}, index=[5, 8, 9])
    rebuilt = pd.DataFrame({"T": [.11, -.2], "n_bets": [30, 39]}, index=[5, 8])
    report = tmp_path / "mismatch.csv"
    with pytest.raises(AssertionError, match="3/3 wallets differ.*missing=1.*bet_count_mismatches=2"):
        check_population_labels(rebuilt, saved, 1, report)
    rows = pd.read_csv(report).set_index("nid")
    assert set(rows.index) == {5, 8, 9}
    assert rows.loc[9, "missing_rebuilt"]
    assert rows.loc[8, "n_bets_rebuilt"] == 39
    assert rows.loc[5, "abs_delta_T"] == pytest.approx(.01)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_label_consistency_rejects_nonfinite_values_even_when_both_match(tmp_path, value):
    saved = pd.DataFrame({"T": [value], "n_bets": [30]}, index=[5])
    with pytest.raises(AssertionError, match="1/1 wallets differ"):
        check_population_labels(saved, saved, 1, tmp_path / "mismatch.csv")
