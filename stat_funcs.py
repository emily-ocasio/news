"""
Compute statistics
"""
import sqlite3
import pandas as pd
from pandas.io.formats.style import Styler

conn = sqlite3.connect('newarticles.db')

CAT_COLS = ('CNTYFIPS',
            'Ori',
            'State',
            'Agency',
            'Agentype',
            'Source',
            'Solved',
            'StateName',
            'Month',
            'ActionType',
            'Situation',
            'Homicide',
            'VicRace',
            'VicSex',
            'VicEthnic',
            'OffSex',
            'OffRace',
            'OffEthnic',
            'Weapon',
            'Relationship',
            'Circumstance',
            'Subcircum',
            'MSA'
            )

CAT_STRS = ('ID',
            'Victim')


def shr_df() -> pd.DataFrame:
    """
    Returns dataframe representing homicides
        with assignment victim names and counts
    """
    types = ({column: 'category' for column in CAT_COLS}
             | {column: 'string'for column in CAT_STRS}
             | {'YearMonth': pd.PeriodDtype(freq='M')})
    df = pd.read_sql_query("SELECT * FROM assignments",
                           conn,
                           index_col='index',
                           dtype=types)  # type: ignore
    combined = combine_columns(df, ('VicRace', 'OffRace'))
    df[combined.name] = combined
    df['OneOrLess'] = df['AssignCount'] < 2
    return df


def breakdown_by_value(series: pd.Series, as_pct=True,
                       sort_by_value=True) -> pd.DataFrame:
    """
    Input: pandas Series
    Return: DataFrame with counts and percentages of values of the series
    """
    name = series.name
    breakdown = series.value_counts()
    df = pd.DataFrame(breakdown)
    pct_name = f"{name}_pct"
    df[pct_name] = series.value_counts(normalize=True)
    df = df.sort_index() if sort_by_value else df
    return df.style.format({pct_name: '{:.2%}'}) if as_pct else df


def victim_race_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: Assignment DataFrame
    Returns DataFrame with percentages for each victim race
    """
    return breakdown_by_value(df['VicRace'], sort_by_value=False)


def offender_race_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame
    Returns DataFrame with percentages for each offender race
    """
    return breakdown_by_value(df['OffRace'], sort_by_value=False)


def combine_columns(df: pd.DataFrame, columns: tuple[str, str]) -> pd.Series:
    """
    Input: DataFrame
    Returns Series of combination of column categories
    """
    def agg_func(seq):
        return "_".join(seq)
    return pd.Series(
        df[list(columns)].agg(agg_func, axis=1),
        name='_'.join(columns))


def prt(styl: Styler) -> None:
    """
    Changes styler (html) display back to string for including in text doc
    """
    print(styl.data.to_string())


def remove_infrequent(
        df: pd.DataFrame, filter_cols=('VicRace', 'OffRace')) -> pd.DataFrame:
    """
    Filter out rows where combination of Victim Race
        and Offender Race results in less than 1% (1800/100 = 18 cases)
    """
    min_val = int(len(df)/100)
    pivot = df.pivot_table(index=filter_cols, aggfunc=len, values='Victim')
    df_filter = (pivot[pivot['Victim'] > min_val]
                    .reset_index()[list(filter_cols)])
    return df.merge(df_filter, on = list(filter_cols))
