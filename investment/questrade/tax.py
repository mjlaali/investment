import glob
import sys
from pprint import pprint

import pandas as pd
import os
import dateutil
import json
import numpy as np
from datetime import timedelta

SETTLEMENT_DATE = 'Settlement Date'

ACCOUNT_TYPE = 'Account Type'
RRSP_ACCOUNT_TYPE = 'Individual RRSP'
TFSA_ACCOUNT_TYPE = 'Individual TFSA'

ACTIVITY_TYPE = 'Activity Type'
WITHDRAWALS = 'Withdrawals'
DEPOSIT = 'Deposits'
INTEREST = 'Interest'
TRANSFER = 'Transfers'

CAD_GROSS = 'CAD Gross'
CAD_COMMISSION = 'CAD Commission'
QUANTITY = 'Quantity'

ACTION = 'Action'
CONTRIBUTION = 'CON'

EXCHANGE_RATE = 'CAD2USD'

NORMALIZED_SYMBOL = 'Normalized Symbol'

ACCOUNT_NUMBER = 'Account #'


def _extract_cad_amount(row):
    if row['Net Amount'] != 0:
        amount = row['Net Amount']
    else:
        description = row['Description']
        start_tag = 'BOOK VALUE $'
        try:
            sign = np.sign(row['Quantity'])
            amount = float(description[description.find(start_tag) + len(start_tag):].replace(',', '')) * sign
        except ValueError as e:
            pprint(row.to_dict())
            raise ValueError(description) from e

    if row['Currency'] == 'USD':
        amount *= row[EXCHANGE_RATE]
    return amount


symbol_map = {
    '8200999': 'Interest',
    'DLR': 'DLR.TO',
    'H038778': 'DLR.TO',
    'H062990': 'QQQ',
    'VCN': 'VCN.TO',
    'VGRO': 'VGRO.TO',
    'V007563': 'VOO',
    'XAW': 'XAW.TO',
    'ZAG': 'ZAG.TO',
    np.nan: ''
}

normalized_symbol = {
    'DLR.TO', 'QQQ', 'VCN.TO', 'VEQT.TO', 'VGRO.TO', 'VOO', 'XAW.TO', 'ZAG.TO', None
}


def _normalize_symbol(symbol):
    if symbol in normalized_symbol:
        return symbol
    assert symbol in symbol_map, f'Please add {symbol} into symbol_map'
    return symbol_map[symbol]


def _fill_gaps(days, rates):
    new_days = []
    new_rates = []

    for day, rate in zip(days, rates):
        if len(new_days) > 0:
            last_date = new_days[-1]
            gap = (day - last_date).days
            if gap > 1:
                avg_rate = (new_rates[-1] + rate) / 2

                for num_days in range(1, gap):
                    new_days.append(last_date + timedelta(days=num_days))
                    new_rates.append(avg_rate)
            else:
                assert gap == 1
        new_days.append(day)
        new_rates.append(rate)
    return new_days, new_rates


def read_bank_of_canada_exchange_rate(exchange_rate_file):
    exchange_rate = pd.read_csv(exchange_rate_file)

    exchange_rate['date'] = exchange_rate['date'].apply(lambda x: dateutil.parser.parse(x).date())
    days, rates = _fill_gaps(exchange_rate['date'], exchange_rate['FXUSDCAD'])

    return pd.DataFrame(
        data={
            SETTLEMENT_DATE: days,
            EXCHANGE_RATE: rates
        })


def _compute_acb(quantities, amounts, commissions):
    avg_share_price = []
    capital_gains = []
    acb = []
    total_shares = 0
    total_acb = 0
    for num_shares, amount, commission in zip(quantities, amounts, commissions):
        assert commission <= 0
        # either buy or sell
        if not (num_shares > 0 > amount) and not (amount > 0 > num_shares):
            print(
                f'num_shares={num_shares} and amount={amount} are not compatible, is this a transfer or an error?',
                file=sys.stderr
            )
            amount *= -1
        if num_shares > 0:
            total_shares += num_shares
            total_acb -= amount + commission
            capital_gains.append(0)
            acb.append(0)
        elif num_shares < 0:
            value = num_shares * total_acb / total_shares
            total_acb += value
            total_shares += num_shares
            acb.append(-(value + commission))
            capital_gains.append(amount + value + commission)
        else:
            raise ValueError()
        if total_shares > 0:
            avg_share_price.append(total_acb / total_shares)
        else:
            avg_share_price.append(0)
    return avg_share_price, capital_gains, acb


class Questrade:
    @staticmethod
    def from_files(data_dir, exchange_rate_file):
        transactions = None
        for fpath in glob.glob(f'{data_dir}/*.xlsx'):
            activities = pd.read_excel(fpath)
            if transactions is None:
                transactions = activities
            else:
                transactions = pd.concat([transactions, activities])

        exchange_rate = read_bank_of_canada_exchange_rate(exchange_rate_file)
        return Questrade(transactions, exchange_rate)

    def __init__(self, transactions, exchange_rate):
        for date_column in ['Transaction Date', 'Settlement Date']:
            transactions[date_column] = transactions[date_column].apply(lambda x: dateutil.parser.parse(x).date())

        self._transactions = transactions.sort_values(SETTLEMENT_DATE)

        self._normalize_currency_to_cad(exchange_rate)
        self._set_contribution_cad_amount()

    def _normalize_currency_to_cad(self, exchange_rate):
        transactions = self._transactions
        assert set(transactions['Currency'].unique().tolist()) == {'USD', 'CAD'}
        transactions = transactions.merge(exchange_rate, on=SETTLEMENT_DATE, how='left')
        assert not transactions[EXCHANGE_RATE].isnull().any(), transactions[transactions[EXCHANGE_RATE].isnull()]

        transactions[CAD_GROSS] = transactions['Net Amount']
        transactions[CAD_COMMISSION] = transactions['Commission']
        transactions.loc[transactions['Currency'] == 'USD',
                         CAD_GROSS] = transactions['Gross Amount'] * transactions[EXCHANGE_RATE]
        transactions.loc[transactions['Currency'] == 'USD',
                         CAD_COMMISSION] = transactions['Commission'] * transactions[EXCHANGE_RATE]
        self._transactions = transactions

    def _set_contribution_cad_amount(self):
        transactions = self._transactions
        zero_contribution = (transactions[ACTION] == CONTRIBUTION) & (
                transactions[CAD_GROSS] == 0)
        transactions.loc[zero_contribution, CAD_GROSS] = \
            transactions[zero_contribution].apply(_extract_cad_amount, axis=1)

    @staticmethod
    def select_transaction_in(transactions, year) -> pd.DataFrame:
        start_of_year = dateutil.parser.parse(f'{year}-01-01').date()
        end_of_year = dateutil.parser.parse(f'{year + 1}-01-01').date()
        return transactions[
            (start_of_year <= transactions[SETTLEMENT_DATE]) & (transactions[SETTLEMENT_DATE] < end_of_year)
            ]

    @property
    def transaction(self):
        return self._transactions

    @property
    def rrsp_transactions(self):
        return self._transactions[self._transactions[ACCOUNT_TYPE] == RRSP_ACCOUNT_TYPE]

    @property
    def non_registered_transactions(self):
        return self._transactions[
            (self._transactions[ACCOUNT_TYPE] != RRSP_ACCOUNT_TYPE) &
            (self._transactions[ACCOUNT_TYPE] != TFSA_ACCOUNT_TYPE)
            ]

    def rrsp_contribution(self, year):
        rrsp_transactions = self.rrsp_transactions
        rrsp_transactions = Questrade.select_transaction_in(rrsp_transactions, year)
        print(rrsp_transactions.to_string())
        rrsp_deposit = rrsp_transactions[rrsp_transactions[ACTIVITY_TYPE] == DEPOSIT]
        return rrsp_deposit[CAD_GROSS].sum()

    def compute_interest(self, year):
        transactions = self.non_registered_transactions
        transactions = Questrade.select_transaction_in(transactions, year)
        interests = transactions[transactions[ACTIVITY_TYPE] == INTEREST]
        return (
            ((interests['Currency'] == 'USD').astype(int) * interests['Net Amount'] * interests[EXCHANGE_RATE]).sum() +
            ((interests['Currency'] == 'CAD').astype(int) * interests['Net Amount']).sum()
        )



    def compute_capital_gain(self):
        transactions = self.non_registered_transactions.copy()
        actions = set(map(str, transactions[ACTION].unique()))
        assert actions.issubset({
            'BRW',  # journal
            'Buy',
            'CON',  # Contribute
            'DEP',  # Deposit
            'DIV',  # Dividend
            'FCH',  # Inter account transfer
            'INT',  # Interest
            'Sell',
            'TF6',  # Transfer TFSA
            'TSF',
            'nan'
        })
        transactions[NORMALIZED_SYMBOL] = transactions['Symbol'].apply(_normalize_symbol)
        transactions['Value'] = transactions[CAD_GROSS] + transactions[CAD_COMMISSION]
        trades = transactions[(transactions[QUANTITY] != 0) & (transactions[ACTION] != 'BRW')].copy()
        acb_by_share = {}
        for share, share_trades in trades.groupby(NORMALIZED_SYMBOL):
            avg_acb, capital_gain, acb = _compute_acb(
                share_trades[QUANTITY], share_trades[CAD_GROSS], share_trades[CAD_COMMISSION]
            )
            share_trades = share_trades.copy()
            share_trades['Shares'] = share_trades[QUANTITY].cumsum()
            share_trades['Avg ACB'] = avg_acb
            share_trades['ACB'] = acb
            share_trades['Capital Gain'] = capital_gain
            share_trades['Net Value'] = share_trades[CAD_GROSS] + share_trades[CAD_COMMISSION]
            acb_by_share[share] = share_trades[
                [
                    SETTLEMENT_DATE,
                    ACTION,
                    NORMALIZED_SYMBOL,
                    'Quantity',
                    'Price',
                    'Gross Amount',
                    'Commission',
                    'Currency',
                    'Account #',
                    EXCHANGE_RATE,
                    'Avg ACB',
                    'Net Value',
                    'ACB',
                    'Capital Gain',
                    'Shares',
                    'Description'
                ]
            ]
        return acb_by_share

    def account_contributions(self):
        accounts_to_contributions = {}
        for account_number in self._transactions[ACCOUNT_NUMBER].unique():
            account_transactions = self._transactions[self._transactions[ACCOUNT_NUMBER] == account_number]
            contributions = account_transactions[
                account_transactions[ACTIVITY_TYPE].isin({DEPOSIT, WITHDRAWALS, TRANSFER})
            ][CAD_GROSS].sum()
            accounts_to_contributions[int(account_number)] = contributions
        return accounts_to_contributions


def main():
    for name in {'majid'}:
        questrade = Questrade.from_files(f'data/{name}/questrade', 'data/FX_RATES_DAILY-sd-2017-01-03.csv')

        acb_shares = questrade.compute_capital_gain()

        for year in range(2019, 2021):
            export_dir = f'data/{name}/{year}'

            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            with open(f'{export_dir}/summary.txt', 'w') as fout:
                fout.write(f'RRSP Contribution at {year} = {questrade.rrsp_contribution(year)}\n')
                fout.write(f'Interest you paid in {year} = {questrade.compute_interest(year)}\n')
                fout.write(
                    f'Account contributions = {json.dumps(questrade.account_contributions(), indent=2, sort_keys=True)}\n'
                )

            for share, trades in acb_shares.items():
                Questrade.select_transaction_in(trades, year).to_csv(f'{export_dir}/{share}.csv')


if __name__ == '__main__':
    main()
