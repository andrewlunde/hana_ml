"""
This module contains PAL wrappers for metrics to assess the quality of
model outputs.

The following functions are available:

    * :func:`accuracy_score`
    * :func:`auc`
    * :func:`confusion_matrix`
    * :func:`multiclass_auc`
    * :func:`r2_score`
"""
import logging
import uuid
from hdbcli import dbapi

from hana_ml.dataframe import quotename
from .pal_base import (
    Table,
    ParameterTable,
    INTEGER,
    DOUBLE,
    NVARCHAR,
    parse_one_dtype,
    execute_logged,
    create,
    call_pal_auto,
    require_pal_usable,
    try_drop,
    arg
)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

def confusion_matrix(conn_context, data, key, label_true=None, label_pred=None,#pylint: disable=too-many-arguments, too-many-locals, too-many-statements
                     beta=None, native=True):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.


    Parameters
    ----------

    conn_context : ConnectionContext

        Database connection object.

    data : DataFrame

        DataFrame containing the data.

    key : str

        Name of the ID column.

    label_true : str, optional

        Name of the original label column.

        If not given, defaults to the second columm.

    label_pred : str, optional

        Name of the the predicted label column.
        If not given, defaults to the third columm.

    beta : float, optional

        Parameter used to compute the F-Beta score.

        Default value: 1

    native : bool, optional

        Indicates whether to use native sql statements for confusion matrix
        calculation.

        Default value: True

    Returns
    -------

    confusion_matrix_df : DataFrame

        Confusion matrix, structured as follows:

          - Original label, with same name and data type as it is in data.
          - Predicted label, with same name and data type as it is in data.
          - Count, type INTEGER, the number of data points with the
            corresponding combination of predicted and original label.


        The dataframe is sorted by (original label, predicted label) in descending
        order.

    classification_report_df : DataFrame

        Structured as follows:

          - Class, type NVARCHAR(100), class name
          - Recall, type DOUBLE, the recall of each class
          - Precision, type DOUBLE, the precision of each class
          - F_MEASURE, type DOUBLE, the F_measure of each class
          - SUPPORT, type INTEGER, the support - sample number in each class

    Examples
    --------
    Data contains the original label and predict label:

    >>> df.collect()
       ID  ORIGINAL  PREDICT
    0   1         1        1
    1   2         1        1
    2   3         1        1
    3   4         1        2
    4   5         1        1
    5   6         2        2
    6   7         2        1
    7   8         2        2
    8   9         2        2
    9  10         2        2

    Calculate the confusion matrix

    >>> cm, cr = confusion_matrix(connection_context, df, 'ID', 'ORIGINAL',
    ...                           'PREDICT')
    >>> cm.collect()
       ORIGINAL  PREDICT  COUNT
    0         1        1      4
    1         1        2      1
    2         2        1      1
    3         2        2      4
    >>> cr.collect()
      CLASS  RECALL  PRECISION  F_MEASURE  SUPPORT
    0     1     0.8        0.8        0.8        5
    1     2     0.8        0.8        0.8        5
    """
    if not native:
        require_pal_usable(conn_context)

    key = arg('key', key, str, required=True)
    label_true = arg('label_true', label_true, str)
    label_pred = arg('label_pred', label_pred, str)
    beta = arg('beta', beta, float)
    native = arg('native', native, bool)
    if label_true is None:
        label_true = data.columns[1]
    if label_pred is None:
        label_pred = data.columns[2]
    if data.dtypes([label_true])[0][1] != data.dtypes([label_pred])[0][1]:
        err_msg = ('The data type of the original_label column and predict_label ' +
                   'column must be the same.')
        logger.error(err_msg)
        raise TypeError(err_msg)
    data = data[[key, label_true, label_pred]]
    unique_id = str(uuid.uuid1())
    unique_id = unique_id.replace('-', '_').upper()
    cm_tbl_spec = [parse_one_dtype(data.dtypes([label_true])[0]),
                   parse_one_dtype(data.dtypes([label_pred])[0]),
                   ("COUNT", INTEGER)
                  ]
    cr_tbl_spec = [('CLASS', NVARCHAR(100)),
                   ('RECALL', DOUBLE),
                   ('PRECISION', DOUBLE),
                   ('F_MEASURE', DOUBLE),
                   ('SUPPORT', INTEGER)
                  ]
    if native:
        cm_tbl = '#CM_{}'.format(unique_id)
        not_in_sql = '(SELECT {0}, {1} FROM ({2}))'.format(quotename(label_true),
                                                           quotename(label_pred),
                                                           data.select_statement)
        col_names = [quotename(label_true), quotename(label_pred)]
        union_sqls = [('(SELECT DISTINCT T1.{0} AS {1}, T2.{2} AS {3}, ' +
                       '0 AS COUNT FROM ({4}) T1, ({4}) T2 ' +
                       'WHERE (T1.{0}, T2.{2}) NOT IN {5})').format(col1_before,
                                                                    quotename(label_true),
                                                                    col2_before,
                                                                    quotename(label_pred),
                                                                    data.select_statement,
                                                                    not_in_sql)
                      for col1_before in col_names for col2_before in col_names]
        union_sqls.append(('(SELECT {0}, {1}, COUNT(*) AS COUNT FROM ({2}) dt GROUP BY {0}, {1})' +
                           ' ORDER BY {0}, {1} INTO {3}').format(quotename(label_true),
                                                                 quotename(label_pred),
                                                                 data.select_statement,
                                                                 cm_tbl))
        cm_insert_sql = ' UNION '.join(union_sqls)
        #creat the intermediate sql for precision, recall and f-score calculation
        mid_sql = ('SELECT A.{0} AS CLASS, CASE WHEN A.UP = 0 OR B.P_DOWN = 0 ' +
                   'THEN 0 ELSE A.UP/B.P_DOWN END AS PRECISION,' +
                   'CASE WHEN A.UP = 0 OR C.R_DOWN = 0 THEN 0 ELSE A.UP/C.R_DOWN END AS RECALL,' +
                   'C.R_DOWN AS SUPPORT FROM ' +
                   '(SELECT {0}, SUM(COUNT) AS UP FROM {2} WHERE {0} = {1} GROUP BY {0}) A,' +
                   '(SELECT {1}, SUM(COUNT) AS P_DOWN FROM {2} GROUP BY {1}) B, ' +
                   '(SELECT {0}, SUM(COUNT) R_DOWN FROM {2} GROUP BY {0}) C ' +
                   'WHERE A.{0} = B.{1} AND A.{0} = C.{0}').format(quotename(label_true),
                                                                   quotename(label_pred),
                                                                   cm_tbl)
        cr_tbl = '#CR_{}'.format(unique_id)
        beta_used = 1
        if beta is not None:
            beta_used = beta
        #based on test, when denominator of the f-measure equation is 0, then f-measure is 0
        cr_insert_sql = ('SELECT CLASS, RECALL, PRECISION, ' +
                         'CASE WHEN ({0} * {0} * PRECISION + RECALL) = 0 THEN 0 ELSE ' +
                         '(1 + {0}*{0}) * PRECISION * RECALL / ' +
                         '({0} * {0} * PRECISION + RECALL) END AS F_MEASURE, ' +
                         'SUPPORT FROM ({1}) INTO {2};').format(beta_used,
                                                                mid_sql,
                                                                cr_tbl)
        try:
            #create table first with fixed sql type and use select into to fill in the data
            create(conn_context, Table(cm_tbl, cm_tbl_spec))
            create(conn_context, Table(cr_tbl, cr_tbl_spec))
            with conn_context.connection.cursor() as cur:
                for sql in (cm_insert_sql, cr_insert_sql):
                    execute_logged(cur, sql)
                cm_df = conn_context.table(cm_tbl)
                cm_df_sorted = cm_df.sort([cm_df.columns[0], cm_df.columns[1]], desc=True)
                return cm_df_sorted, conn_context.table(cr_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            raise
    #Through PAL
    tables = cm_tbl, cr_tbl = ['#CM_{}_TBL_{}'.format(name, unique_id)
                               for name in ['MATRIX', 'CLASSIFICATION_REPORT']]
    param_rows = [('BETA', None, beta, None)]
    try:
        #materialize(conn_context, data_tbl, data)
        #if beta is not None:
        #    params = [('BETA', None, beta, None)]
        #    create(conn_context, ParameterTable(param_tbl).with_data(params))
        #else:
        #    create(conn_context, ParameterTable(param_tbl))
        #create(conn_context, Table(cm_tbl, cm_tbl_spec))
        #create(conn_context, Table(cr_tbl, cr_tbl_spec))
        call_pal_auto(conn_context,
                      'PAL_CONFUSION_MATRIX',
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_er:
        logger.exception("Fit failure, the error message: %s", db_er, exc_info=True)
        try_drop(conn_context, tables)
        raise
    cm_df = conn_context.table(cm_tbl)
    return cm_df.sort([cm_df.columns[0], cm_df.columns[1]], desc=True), conn_context.table(cr_tbl)

def auc(conn_context, data, positive_label=None):
    """
    Compute area under curve (AUC) to evaluate the performance
    of binary-class classification algorithms.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data : DataFrame

        Input data, structured as follows:

            - ID column.
            - True class of the data point.
            - Classifier-computed probability that the data point belongs to the positive class.

    positive_label : str, optional

        If original label is not 0 or 1, specifies the
        label value which will be mapped to 1.

    Returns
    -------

    auc : float

        The area under the receiver operating characteristic curve.

    roc : DataFrame

        False positive rate and true positive rate, structured as follows:

            - ID column, type INTEGER.
            - FPR, type DOUBLE, representing false positive rate.
            - TPR, type DOUBLE, representing true positive rate.

    Examples
    --------

    Input data:

    >>> df.collect()
       ID  ORIGINAL  PREDICT
    0   1         0     0.07
    1   2         0     0.01
    2   3         0     0.85
    3   4         0     0.30
    4   5         0     0.50
    5   6         1     0.50
    6   7         1     0.20
    7   8         1     0.80
    8   9         1     0.20
    9  10         1     0.95

    Compute Area Under Curve:

    >>> auc, roc = auc(cc, df)

    Ideal output:

    >>> print(auc)
     0.66

    >>> roc.collect()
       ID  FPR  TPR
    0   0  1.0  1.0
    1   1  0.8  1.0
    2   2  0.6  1.0
    3   3  0.6  0.6
    4   4  0.4  0.6
    5   5  0.2  0.4
    6   6  0.2  0.2
    7   7  0.0  0.2
    8   8  0.0  0.0
    """
    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    positive_label = arg('positive_label', positive_label, str)

    tables = ['AUC', 'ROC']
    tables = ['#PAL_AUC_{}_TBL_{}'.format(name, unique_id)
              for name in tables]
    auc_tbl, roc_tbl = tables

    param_rows = [
        ('POSITIVE_LABEL', None, None, positive_label)]

    #auc_spec = [
    #    ('STAT_NAME', NVARCHAR(100)),
    #    ('STAT_VALUE', DOUBLE),
    #]

    #roc_spec = [
    #    ('ID', INTEGER),
    #    ('FPR', DOUBLE),
    #    ('TPR', DOUBLE)
    #]

    try:
        #materialize(conn_context, data_tbl, data)
        #if positive_label is not None:
        #    create(conn_context, ParameterTable(param_tbl).with_data(param_rows))
        #else:
        #    create(conn_context, ParameterTable(param_tbl))
        #create(conn_context, Table(auc_tbl, auc_spec))
        #create(conn_context, Table(roc_tbl, roc_spec))
        call_pal_auto(conn_context,
                      'PAL_AUC',
                      data,
                      ParameterTable().with_data(param_rows),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

    return (float(conn_context.table(auc_tbl).collect().iloc[0]['STAT_VALUE']),
            conn_context.table(roc_tbl))

def multiclass_auc(conn_context, data_original, data_predict):
    """
    Compute area under curve (AUC) to evaluate the performance
    of multi-class classification algorithms.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    data_original : DataFrame

        True class data, structured as follows:

            - Data point ID column.
            - True class of the data point.

    data_predict : DataFrame

        Predicted class data, structured as follows:

            - Data point ID column.
            - Possible class.
            - Classifier-computed probability that the data point belongs
              to that particular class.

        For each data point ID, there should be one row for each possible
        class.

    Returns
    -------

    auc : float

        The area under the receiver operating characteristic curve.

    roc : DataFrame

        False positive rate and true positive rate, structured as follows:

            - ID column, type INTEGER.
            - FPR, type DOUBLE, representing false positive rate.
            - TPR, type DOUBLE, representing true positive rate.

    Examples
    --------

    Input data:

    >>> df_original.collect()
       ID  ORIGINAL
    0   1         1
    1   2         1
    2   3         1
    3   4         2
    4   5         2
    5   6         2
    6   7         3
    7   8         3
    8   9         3
    9  10         3

    >>> df_predict.collect()
        ID  PREDICT  PROB
    0    1        1  0.90
    1    1        2  0.05
    2    1        3  0.05
    3    2        1  0.80
    4    2        2  0.05
    5    2        3  0.15
    6    3        1  0.80
    7    3        2  0.10
    8    3        3  0.10
    9    4        1  0.10
    10   4        2  0.80
    11   4        3  0.10
    12   5        1  0.20
    13   5        2  0.70
    14   5        3  0.10
    15   6        1  0.05
    16   6        2  0.90
    17   6        3  0.05
    18   7        1  0.10
    19   7        2  0.10
    20   7        3  0.80
    21   8        1  0.00
    22   8        2  0.00
    23   8        3  1.00
    24   9        1  0.20
    25   9        2  0.10
    26   9        3  0.70
    27  10        1  0.20
    28  10        2  0.20
    29  10        3  0.60

    Compute Area Under Curve:

    >>> auc, roc = multiclass_auc(cc, df_original, df_predict)

    Ideal output:

    >>> print(auc)
    1.0

    >>> roc.collect()
        ID   FPR  TPR
    0    0  1.00  1.0
    1    1  0.90  1.0
    2    2  0.65  1.0
    3    3  0.25  1.0
    4    4  0.20  1.0
    5    5  0.00  1.0
    6    6  0.00  0.9
    7    7  0.00  0.7
    8    8  0.00  0.3
    9    9  0.00  0.1
    10  10  0.00  0.0
    """
    require_pal_usable(conn_context)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    tables = ['AUC', 'ROC']
    tables = ['#PAL_AUC_{}_TBL_{}'.format(name, unique_id)
              for name in tables]
    auc_tbl, roc_tbl = tables

    #auc_spec = [
    #    ('STAT_NAME', NVARCHAR(100)),
    #    ('STAT_VALUE', DOUBLE),
    #]

    #roc_spec = [
    #    ('ID', INTEGER),
    #    ('FPR', DOUBLE),
    #    ('TPR', DOUBLE)
    #]

    try:
        #materialize(conn_context, data_tbl_original, data_original)
        #materialize(conn_context, data_tbl_predict, data_predict)
        #create(conn_context, ParameterTable(param_tbl))
        #create(conn_context, Table(auc_tbl, auc_spec))
        #create(conn_context, Table(roc_tbl, roc_spec))
        call_pal_auto(conn_context,
                      'PAL_MULTICLASS_AUC',
                      data_original,
                      data_predict,
                      ParameterTable(),
                      *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, tables)
        raise

    return (float(conn_context.table(auc_tbl).collect().iloc[0]['STAT_VALUE']),
            conn_context.table(roc_tbl))

def accuracy_score(conn_context, data, label_true, label_pred):
    """
    Compute mean accuracy score for classification results. That is,
    the proportion of the correctly predicted results among the total
    number of cases examined.

    Parameters
    ----------

    conn_context : ConnectionContext

        HANA connection.

    data : DataFrame

        DataFrame of true and predicted labels.

    label_true : str

        Name of the column containing ground truth labels.

    label_pred : str

        Name of the column containing predicted labels, as returned
        by a classifier.

    Returns
    -------

    accuracy : float

        Accuracy classification score. A lower accuracy indicates that the
        classifier was able to predict less of the labels in the input
        correctly.

    Examples
    --------

    Actual and predicted labels for a hypothetical classification:

    >>> df.collect()
       ACTUAL  PREDICTED
    0    1        0
    1    0        0
    2    0        0
    3    1        1
    4    1        1

    Accuracy score for these predictions:

    >>> accuracy_score(cc, df, label_true='ACTUAL', label_pred='PREDICTED')
    0.8

    Compare that to null accuracy (accuracy that could be achieved by always
    predicting the most frequent class):

    >>> df_dummy.collect()
       ACTUAL  PREDICTED
    0    1       1
    1    0       1
    2    0       1
    3    1       1
    4    1       1
    >>> accuracy_score(cc, df, label_true='ACTUAL', label_pred='PREDICTED')
    0.6

    A perfect predictor:

    >>> df_perfect.collect()
       ACTUAL  PREDICTED
    0    1       1
    1    0       0
    2    0       0
    3    1       1
    4    1       1
    >>> accuracy_score(cc, df, label_true='ACTUAL', label_pred='PREDICTED')
    1.0
    """
    label_true = arg('label_true', label_true, str, required=True)
    label_pred = arg('label_pred', label_pred, str, required=True)

    data = data.select(label_true, label_pred).rename_columns(
        ['ACTUAL', 'PREDICTED'])
    if data.empty() or data.hasna():
        msg = ('Input dataframe is empty or has NULL values.')
        logger.error(msg)
        raise ValueError(msg)

    accuracy_select = ('SELECT COUNT(CASE WHEN ACTUAL = PREDICTED THEN 1 END)/' +
                       'COUNT(*) FROM ({data})').format(
                           data=data.select_statement)

    try:
        with conn_context.connection.cursor() as cur:
            execute_logged(cur, accuracy_select)
            return float(cur.fetchone()[0])
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise

def r2_score(conn_context, data, label_true, label_pred):
    """
    Compute coefficient of determination for regression results.

    Parameters
    ----------

    conn_context : ConnectionContext

        HANA connection.

    data : DataFrame

        DataFrame of true and predicted values.

    label_true : str

        Name of the column containing true values.

    label_pred : str

        Name of the column containing values predicted by regression.

    Returns
    -------

    r2 : float

        Coefficient of determination. 1.0 indicates an exact match between
        true and predicted values. A lower coefficient of determination
        indicates that the regression was able to predict less of the
        variance in the input. A negative value indicates that the regression
        performed worse than just taking the mean of the true values and
        using that for every prediction.

    Examples
    --------

    Actual and predicted values for a hypothetical regression:

    >>> df.collect()
       ACTUAL  PREDICTED
    0    0.10        0.2
    1    0.90        1.0
    2    2.10        1.9
    3    3.05        3.0
    4    4.00        3.5

    R^2 score for these predictions:

    >>> r2_score(cc, df, label_true='ACTUAL', label_pred='PREDICTED')
    0.9685233682514102

    Compare that to the score for a perfect predictor:

    >>> df_perfect.collect()
       ACTUAL  PREDICTED
    0    0.10       0.10
    1    0.90       0.90
    2    2.10       2.10
    3    3.05       3.05
    4    4.00       4.00
    >>> r2_score(cc, df_perfect, label_true='ACTUAL', label_pred='PREDICTED')
    1.0

    A naive mean predictor:

    >>> df_mean.collect()
       ACTUAL  PREDICTED
    0    0.10       2.03
    1    0.90       2.03
    2    2.10       2.03
    3    3.05       2.03
    4    4.00       2.03
    >>> r2_score(cc, df_mean, label_true='ACTUAL', label_pred='PREDICTED')
    0.0

    And a really awful predictor:

    >>> df_awful.collect()
       ACTUAL  PREDICTED
    0    0.10    12345.0
    1    0.90    91923.0
    2    2.10    -4444.0
    3    3.05    -8888.0
    4    4.00    -9999.0
    >>> r2_score(cc, df_awful, label_true='ACTUAL', label_pred='PREDICTED')
    -886477397.139857
    """
    # No defaults for label_true and label_pred, for now.
    # if label_true is None:
    #     label_true = data.columns[0]
    # if label_pred is None:
    #     label_pred = data.columns[1]
    label_true = arg('label_true', label_true, str, required=True)
    label_pred = arg('label_pred', label_pred, str, required=True)

    data = data.select(label_true, label_pred).rename_columns(
        ['ACTUAL', 'PREDICTED'])
    if data.empty() or data.hasna():
        msg = ('Input dataframe is empty or has NULL values.')
        logger.error(msg)
        raise ValueError(msg)

    select_mean = 'SELECT AVG(ACTUAL) AS AV FROM ({})'.format(
        data.select_statement)

    # residual sum of squares
    rss = 'SUM(POWER(ACTUAL - PREDICTED, 2))'
    # total sum of squares
    tss = 'SUM(POWER(ACTUAL - AV, 2))'

    r2_select = ('SELECT 1- {rss}/{tss} FROM ({data}) as data, ' +
                 '({select_mean}) as avg_tbl').format(
                     rss=rss,
                     tss=tss,
                     data=data.select_statement,
                     select_mean=select_mean)

    try:
        with conn_context.connection.cursor() as cur:
            execute_logged(cur, r2_select)
            return cur.fetchone()[0]
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise
