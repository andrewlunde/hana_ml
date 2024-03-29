"""
This module contains PAL wrapper for white noise test algorithms.

The following function is available:

    * :func:`white_noise_test`
"""

#pylint:disable=line-too-long
import logging
import uuid
from hdbcli import dbapi

from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    arg,
    try_drop,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

def white_noise_test(conn_context, data, endog=None, key=None, lag=None, probability=None, thread_ratio=None):#pylint:disable=too-many-arguments, too-few-public-methods
    r"""
    This algorithm is used to identify whether a time series is a white noise series.
    If white noise exists in the raw time series, the algorithm returns the value of 1. If not, the value of 0 will be returned.

    Parameters
    ----------

    conn_context : ConnectionContext
        Connection to the HANA system.

    data : DataFrame
        Input data. At least two columns, one is ID column, the other is raw data.

    endog : str, optional

        The column of series to be tested.

        Defaults to the second column.

    key : str, optional
        The ID column.

        Defaults to the first column.
    lag : int, optional
        Specifies the lag autocorrelation coefficient that the statistic will be based on.
        It corresponds to the degree of freedom of chi-square distribution.

        Defaults to half of the sample size (n/2).

    probability : float, optional
        The confidence level used for chi-square distribution.
        The value is 1 - a, where a is the significance level.

        Defaults to 0.9.

    thread_ratio : float, optional
        The ratio of available threads.

        - 0: single thread
        - 0~1: percentage
        - Others: heuristically determined

        Defaults to -1.

    Returns
    -------
    stats_tbl : DataFrame
        Statistics for time series, structured as follows:

        - STAT_NAME: Name of the statistics of the series.
        - STAT_VALUE: include following values
                    - WN: 1 for white noise, 0 for not white noise
                    - Q: Q statistics defined as above
                    - chi^2: chi-square distribution

    Attributes
    ----------
    None

    Examples
    --------

    Time series data:

    >>> df.collect()
        TIME_STAMP    SERIES
    0      1           1356.00
    1      2           826.00
    2      3           1586.00
    ......
    10     11          2218.00
    11     12          2400.00

    Perform white_noise_test function:

    >>> stats = white_noise_test(cc, data, endog='SERIES', lag=3,
                                  probability=0.9, thread_ratio=0.2)

    Outputs:

    >>> stats.collect()
        STATS_NAME    STAT_VALUE
    0    WN            1.000000
    1    Q             5.576053
    2    chi^2         6.251389
    """
    #pylint: disable=too-many-arguments

    lag = arg('method', lag, int)
    probability = arg('probability', probability, float)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    key = arg('key', key, str)
    endog = arg('endog', endog, str)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    stats_tbl = '#PAL_WHITE_NOISE_TEST_STATS_TBL_{}_{}'.format(id, unique_id)

    cols = data.columns
    if len(cols) < 2:
        msg = ("Input data should contain at least 2 columns: " +
               "one for ID, another for raw data.")
        logger.error(msg)
        raise ValueError(msg)

    if endog is not None and endog not in cols:
        msg = ('The endog should be selected from columns of data!')
        logger.error(msg)
        raise ValueError(msg)

    if key is not None and key not in cols:
        msg = ('The key should be selected from columns of data!')
        logger.error(msg)
        raise ValueError(msg)

    if key is None:
        key = cols[0]

    if endog is None:
        endog = cols[1]

    if key == endog:
        msg = ('The key and endog cannot be same!')
        logger.error(msg)
        raise ValueError(msg)

    data = data[[key] + [endog]]

    param_rows = [('LAG', lag, None, None),
                  ('PROBABILITY', None, probability, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]

    try:
        param_t = ParameterTable(name='#PARAM_TBL')
        call_pal_auto(conn_context,
                      'PAL_WN_TEST',
                      data,
                      param_t.with_data(param_rows),
                      stats_tbl)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, '#PARAM_TBL')
        try_drop(conn_context, stats_tbl)
        raise

    #pylint: disable=attribute-defined-outside-init, unused-variable
    return conn_context.table(stats_tbl)
