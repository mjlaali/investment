import pytest
import dateutil
from investment.questrade.tax import _fill_gaps, _compute_acb


@pytest.mark.parametrize(
    'days,rates,expected_days,expected_rates',
    [
        (
            [],
            [],
            [],
            []
        ),
        (
            [dateutil.parser.parse('2020-01-01')],
            [1.0],
            [dateutil.parser.parse('2020-01-01')],
            [1.0]
        ),
        (
                [
                    dateutil.parser.parse('2020-01-01'),
                    dateutil.parser.parse('2020-01-04')
                ],
                [1.0, 3.0],
                [
                    dateutil.parser.parse('2020-01-01'),
                    dateutil.parser.parse('2020-01-02'),
                    dateutil.parser.parse('2020-01-03'),
                    dateutil.parser.parse('2020-01-04'),
                ],
                [1.0, 2.0, 2.0, 3.0]
        )

    ]
)
def test_fill_gap(days, rates, expected_days, expected_rates):
    out_days, out_rates = _fill_gaps(days, rates)
    assert out_days == expected_days
    assert out_rates == expected_rates


@pytest.mark.parametrize(
    'quantities, amounts, commissions, expected_avg_price, expected_capital_gain',
    [
        (
            [1, -1], [-100, 105], [0, 0], [100, 0], [0, 5]
        ),
        (
            [1, -1], [-100, 105], [0, -1], [100, 0], [0, 4]
        ),
        (
                [100, 20, -50, -70], [-100, -20, 50, 100], [-10, -15, -20, -10],
                [1.10, 1.21, 1.21, 0], [0, 0, -30.42, 5.42]
        )
    ]
)
def test_compute_acb(quantities, amounts, commissions, expected_avg_price, expected_capital_gain):
    avg_price, capital_gains, _ = _compute_acb(quantities, amounts, commissions)
    assert avg_price == pytest.approx(expected_avg_price, rel=1e-1)
    assert capital_gains == pytest.approx(expected_capital_gain, rel=1e-1)
    assert sum(amounts) + sum(commissions) == pytest.approx(sum(capital_gains))
