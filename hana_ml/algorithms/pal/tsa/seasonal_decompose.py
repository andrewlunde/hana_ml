"""
This module contains Python wrappers for PAL seasonality test algorithm.

The following function is available:

    * :func:`seasonal_decompose`
"""

# pylint:disable=line-too-long
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

def seasonal_decompose(conn_context, data, endog=None, key=None, alpha=None, thread_ratio=None, model=None):#pylint:disable=too-many-arguments, too-few-public-methods, unused-argument, too-many-locals
    """
    Seasonal_decompose is to decompose a time series into three components: trend, seasonality and random noise.

    Parameters
    ----------

    conn_context : ConnectionContext
        Connection to the HANA system.

    data : DataFrame
        Input data. At least two columns, one is ID column, the other is raw data.

    endog : str, optional
        The column of series to be decompsed.
        Defaults to the second column.

    key : str, optional
        The ID column.
        Defaults to the first column.

    alpha : float, optional
        The criterion for the autocorrelation coefficient.
        The value range is (0, 1). A larger value indicates stricter requirement for seasonality.

        Defaults to 0.2.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread, \
        and 1 indicates up to all available threads. Values between 0 and 1 \
        will use that percentage of available threads. Values outside this \
        range tell PAL to heuristically determine the number of threads to use.

        Defaults to heuristically determined.

    Returns
    -------

    stats_tbl : DataFrame
        Statistics for time series, structured as follows:

        - STAT_NAME: includes type (additive or multiplicative), period (number of seasonality)
          acf (autocorrelation coefficient).
        - STAT_VALUE: value of stats above.

    decompose_tbl : DataFrame
        Seasonal decomposition, structured as follows:

            - ID : index/time stamp.
            - SEASONAL: seasonality component.
            - TREND: trend component.
            - RANDOM: white noise component.

    Examples
    --------

    Time series data:

        TIME_STAMP    SERIES
    0      1            10.0
    1      2            7.0
    2      3            17.0
    ......
    34     35           0.0
    35     36           0.0

    Perform seasonal_decompse function:

    >>> stats, decompose = seasonal_decompose(cc, data, engod='SERIES', alpha=0.2, thread_ratio=0.5)

    Outputs:

    >>> stats.collect()
        STATS_NAME    STAT_VALUE
    0    type        multiplicative
    1    period      4
    2    acf         0.501947

    >>> decompose.collect()
         ID    SEASONAL    TREND        RANDOM
    0    1     1.252660    10.125       0.788445
    1    2     0.349952    14.000       1.428769
    2    3     0.748851    16.875       1.345271
    ......
    34   35    0.748851    32.125       0.000000
    35   36    1.648537    3.125        0.000000

    """

    alpha = arg('alpha', alpha, float)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    key = arg('key', key, str)
    endog = arg('endog', endog, str)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = ['STATS', 'DECOMPOSE',]
    outputs = ['#PAL_SEASONAL_DECOMPOSE_{}_TBL_{}_{}'.format(name, id, unique_id) for name in outputs]
    stats_tbl, decompose_tbl = outputs

    param_rows = [('ALPHA', None, alpha, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]

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
                      'PAL_SEASONALITY_TEST',
                      data,
                      param_t.with_data(param_rows),
                      *outputs)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, '#PARAM_TBL')
        try_drop(conn_context, stats_tbl)
        try_drop(conn_context, decompose_tbl)
        raise

    #pylint: disable=attribute-defined-outside-init, unused-variable
    return conn_context.table(stats_tbl), conn_context.table(decompose_tbl)
