"""
This module contains Python wrappers for PAL trend test algorithm.

The following function is available:

    * :func:`trend_test`
"""

# pylint:disable=line-too-long, too-many-arguments, too-few-public-methods, unused-argument, too-many-locals
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

def trend_test(conn_context, data, endog=None, key=None, method=None, alpha=None):
    r"""
    Trend test is able to identify whether a time series has an upward or downward trend or not, and calculate the de-trended time series.

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

    method : {'mk', 'difference-sign'}, optional

        The method used to identify trend:

            'mk': MK test.

            'difference-sign': Difference-sign test.

        Defaults to 'mk'.

    alpha : float, optional

        Significance value. .
        The value range is (0, 0.5).
        Defaults to 0.05.

    Returns
    -------

    stats_tbl : DataFrame
        Statistics for time series, structured as follows:

        - STAT_NAME: includes TREND (-1 for downward trend, 0 for no trend, and 1 for upward trend),
          S ('difference-sign': the number S of times that difference of adjacent data is positive;
          'MK': the number of positive pairs minus the negative pairs),
          P-VALUE: The p-value of the observed S.
        - STAT_VALUE: value of stats above.

    detrended_tbl : DataFrame
        detrended_tbl, structured as follows:

        - ID : Time stamp that is monotonically increasing sorted.
        - DETRENDED_SERIES: The corresponding de-trended time series. The first value absents if trend presents.

    Examples
    --------

    Time series data:

    >>> df.collect()
        TIME_STAMP    SERIES
    0      1           1500
    1      2           1510
    2      3           1550
    ......
    10     11          1708
    11     12          1715

    Perform trend_test function:

    >>> stats, detrended = trend_test(cc, data, endog= 'SERIES', method= 'mk', alpha=0.05)

    Outputs:

    >>> stats.collect()
        STATS_NAME    STAT_VALUE
    0    TREND            1
    1    S                60
    2    P-VALUE         0.0000267...

    >>> detrended.collect()
         ID    DETRENDED_SEARIES
    1    2     10
    2    3     40
    ......
    11   11    3
    12   12    7

    """
    method_map = {'mk':1, 'difference-sign':2}

    method = arg('method', method, method_map)
    alpha = arg('alpha', alpha, float)
    key = arg('key', key, str)
    endog = arg('endog', endog, str)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = ['STATS', 'DETRENDED',]
    outputs = ['#PAL_TREND_TEST_{}_TBL_{}_{}'.format(name, id, unique_id) for name in outputs]
    stats_tbl, detrended_tbl = outputs

    param_rows = [('METHOD', method, None, None),
                  ('ALPHA', None, alpha, None)]

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

    try:
        param_t = ParameterTable(name='#PARAM_TBL')
        call_pal_auto(conn_context,
                      'PAL_TREND_TEST',
                      data,
                      param_t.with_data(param_rows),
                      *outputs)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, '#PARAM_TBL')
        try_drop(conn_context, stats_tbl)
        try_drop(conn_context, detrended_tbl)
        raise

    #pylint: disable=attribute-defined-outside-init, unused-variable
    return conn_context.table(stats_tbl), conn_context.table(detrended_tbl)
