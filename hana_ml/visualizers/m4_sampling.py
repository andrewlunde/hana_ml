"""M4 algorithm for sampling query"""
import logging
from hdbcli import dbapi
from hana_ml import dataframe

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

def get_min_index(hana_df):
    """Get Minimum Timestamp of Time Series Data

    Parameters
    ----------
    hana_df : DataFrame
        Time series data whose 1st column is index and 2nd one is value.

    Returns
    -------
    datetime
        Return the minimum timestamp.
    """
    if hana_df.count() == 0 or len(hana_df.columns) < 2:
        raise Exception("Empty dataframe or the number of columns is less than 2")
    try:
        return hana_df.connection_context.sql('SELECT ' + 'MIN(' + '\"' + hana_df.columns[0] \
    + '\"' + ')' + ' FROM(' + hana_df.select_statement + ')').collect().iat[0, 0]
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise

def get_max_index(hana_df):
    """Get Maximum Timestamp of Time Series Data

    Parameters
    ----------
    hana_df : DataFrame
        Time series data whose 1st column is index and 2nd one is value.

    Returns
    -------
    datetime
        Return the maximum timestamp.
    """
    if hana_df.count() == 0 or len(hana_df.columns) < 2:
        raise Exception("Empty dataframe or the number of columns is less than 2")
    try:
        return hana_df.connection_context.sql('SELECT ' + 'MAX(' + '\"' + hana_df.columns[0] \
    + '\"' + ')' + ' FROM(' + hana_df.select_statement + ')').collect().iat[0, 0]
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise

def m4_sampling(hana_df, width):
    """M4 algorithm for big data visualization

    Parameters
    ----------
    hana_df : DataFrame
        Data to be sampled. Time seires data whose 1st column is index and 2nd one is value.
    width : int
        Sampling Rate. It is an indicator of how many pixels being in the picture.

    Returns
    -------
    DataFrame
        Return the sampled dataframe.
    """
    index = hana_df.columns[0]
    value = hana_df.columns[1]
    try:
        start_time = get_min_index(hana_df).strftime("%Y-%m-%d %H:%M:%S")
        end_time = get_max_index(hana_df).strftime("%Y-%m-%d %H:%M:%S")
        to_ts_start_time = 'TO_TIMESTAMP(' + "'" + start_time + "'" + ',' \
            + "'" + 'YYYY-MM-DD HH24:MI:SS' + "'" + ')'
        to_ts_end_time = 'TO_TIMESTAMP(' + "'" + end_time + "'" + ',' \
            + "'" + 'YYYY-MM-DD HH24:MI:SS' + "'" + ')'
        add_key_sql_statement = 'SELECT round('\
            + str(width)\
            + '*SECONDS_BETWEEN(' + index + ',' + to_ts_start_time + ')/(1e-10 + SECONDS_BETWEEN('\
            + to_ts_end_time + ',' + to_ts_start_time + '))) k,' + index + ',' + value \
            + ' FROM ' + '(' + hana_df.select_statement + ')'
        min_max_groupby_sql_statement = 'SELECT k, min(' + value + ')  v_min, max(' \
            + value + ') v_max, min(' + index + ') t_min, max(' + index + ') t_max FROM' \
            + '(' + add_key_sql_statement + ') GROUP BY k'
        join_sql_statement = 'SELECT T.' + index + ', T.' + value + ' FROM ' + '(' \
            + add_key_sql_statement + ') T ' + 'INNER JOIN '\
            + '(' + min_max_groupby_sql_statement + ') Q ' \
            + 'ON T.k = Q.k ' \
            + 'AND (T.' + value + ' = Q.v_min OR T.' + value + ' = Q.v_max OR T.' + index \
            + ' = Q.t_min OR T.' + index + ' = Q.t_max)'
    except Exception as err:
        logger.exception(str(err))
        raise
    try:
        df_sample = dataframe.DataFrame(hana_df.connection_context, join_sql_statement)
        return df_sample
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise
