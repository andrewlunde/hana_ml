"""
This module contains PAL wrappers for the k-nearest neighbors algorithms.

The following classes are available:

    * :class:`KNN`
"""

#pylint:disable=too-many-lines
import logging
import uuid

from hdbcli import dbapi

from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings
)

from . import metrics

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class KNN(PALBase):#pylint:disable=too-many-instance-attributes
    """
    K-Nearest Neighbor(KNN) model that handles classification problems.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA sytem.

    n_neighbors : int, optional

        Number of nearest neighbors.

        Defaults to 1.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates
        a single thread, and 1 indicates up to all available threads.
        Values between 0 and 1 will use up to that percentage of available
        threads. Values outside this range tell PAL to heuristically determine
        the number of threads to use.

        Defaults to 0.

    voting_type : {'majority', 'distance-weighted'}, optional

        Method used to vote for the most frequent label of the K
        nearest neighbors.

        Defaults to distance-weighted.

    stat_info : bool, optional

        Controls whether to return a statistic information table containing
        the distance between each point in the prediction set and its
        k nearest neighbors in the training set.
        If true, the table will be returned.

        Defaults to True.

    metric : {'manhattan', 'euclidean', 'minkowski', 'chebyshev'}, optional

        Ways to compute the distance between data points.

        Defaults to euclidean.

    minkowski_power : float, optional

        When Minkowski is used for ``metric``, this parameter controls the value
        of power.
        Only valid when ``metric`` is Minkowski.

        Defaults to 3.0.

    algorithm : {'brute-force', 'kd-tree'}, optional

        Algorithm used to compute the nearest neighbors.

        Defaults to brute-force.

    Examples
    --------
    Training data:

    >>> df.collect()
       ID      X1      X2  TYPE
    0   0     1.0     1.0     2
    1   1    10.0    10.0     3
    2   2    10.0    11.0     3
    3   3    10.0    10.0     3
    4   4  1000.0  1000.0     1
    5   5  1000.0  1001.0     1
    6   6  1000.0   999.0     1
    7   7   999.0   999.0     1
    8   8   999.0  1000.0     1
    9   9  1000.0  1000.0     1

    Create KNN instance and call fit:

    >>> knn = KNN(connection_context, n_neighbors=3, voting_type='majority',
    ...           thread_ratio=0.1, stat_info=False)
    >>> knn.fit(df, 'ID', features=['X1', 'X2'], label='TYPE')
    >>> pred_df = connection_context.table("PAL_KNN_CLASSDATA_TBL")

    Call predict:

    >>> res, stat = knn.predict(pred_df, "ID")
    >>> res.collect()
       ID  TYPE
    0   0     3
    1   1     3
    2   2     3
    3   3     1
    4   4     1
    5   5     1
    6   6     1
    7   7     1
    """

    voting_map = {'majority':0, 'distance-weighted':1}
    metric_map = {'manhattan':1, 'euclidean':2, 'minkowski':3, 'chebyshev':4}
    algorithm_map = {'brute-force':0, 'kd-tree':1}

    def __init__(self, conn_context, n_neighbors=None, thread_ratio=None,#pylint:disable=too-many-arguments, too-many-branches
                 voting_type=None, stat_info=True, metric=None, minkowski_power=None,
                 algorithm=None):
        super(KNN, self).__init__(conn_context)
        self.n_neighbors = self._arg('n_neighbors', n_neighbors, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.voting_type = self._arg('voting_type', voting_type, self.voting_map)
        self.stat_info = self._arg('stat_info', stat_info, bool)
        self.metric = self._arg('metric', metric, self.metric_map)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, float)
        if self.metric != 3 and minkowski_power is not None:
            msg = 'Minkowski_power will only be valid if distance metric is Minkowski.'
            logger.error(msg)
            raise ValueError(msg)
        self.algorithm = self._arg('algorithm', algorithm, self.algorithm_map)

    def fit(self, data, key, features=None, label=None):
        """
        Fit the model when given training set.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all the
            non-ID and non-label columns.

        label : str, optional

            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.
        """

        #ID is necessary for training set and should be placed as the first column
        #PAL default: ID, Label, Features
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        col_left = data.columns
        col_left.remove(key)
        if label is None:
            label = col_left[-1]
        col_left.remove(label)
        if features is None:
            features = col_left
        # training_data will be used in the prediction and score
        training_data = data[[key] + [label] + features]
        self._training_set = training_data#pylint:disable=attribute-defined-outside-init

    def predict(self, data, key, features=None):#pylint:disable=too-many-locals
        r"""
        Predict the class labels for the provided data

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        Returns
        -------

        result_df : DataFrame

            Predicted result, structured as follows:

              - ID column, with same name and type as ``data`` 's ID column.
              - Label column, with same name and type as training data's label
                column.

        nearest_neighbors_df : DataFrame

            The distance between each point in ``data`` and its k nearest
            neighbors in the training set.
            Only returned if ``stat_info`` is True.

            Structured as follows:

              - TEST\_ + ``data`` 's ID name, with same type as ``data`` 's ID column,
                query data ID.
              - K, type INTEGER, K number.
              - TRAIN\_ + training data's ID name, with same type as training
                data's ID column, neighbor point's ID.
              - DISTANCE, type DOUBLE, distance.
        """

        if not hasattr(self, '_training_set'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        col_left = data.columns
        col_left.remove(key)
        # 'ID' is necessary for prediction data(PAL default: ID, Feature)
        if features is None:
            features = col_left
        # check feature length
        features_train = self._training_set.columns[2:]
        if len(features_train) != len(features):
            msg = ("The number of features must be the same for both training data and " +
                   "prediction data.")
            logger.error(msg)
            raise ValueError(msg)
        data = data[[key] + features]
        #data_tbl = "#KNN_PREDICT_DATA_TBL_{}_{}".format(self.id, unique_id)
        ## parameter update
        #param_tbl = '#KNN_PREDICT_CONTROL_TBL_{}_{}'.format(self.id, unique_id)
        param_array = [('K_NEAREST_NEIGHBOURS', self.n_neighbors, None, None),
                       ('THREAD_RATIO', None, self.thread_ratio, None),
                       ('ATTRIBUTE_NUM', len(features), None, None),
                       ('VOTING_TYPE', self.voting_type, None, None),
                       ('STAT_INFO', self.stat_info, None, None),
                       ('DISTANCE_LEVEL', self.metric, None, None),
                       ('MINKOWSKI_POWER', None, self.minkowski_power, None),
                       ('METHOD', self.algorithm, None, None)]
        #index_name, index_type = parse_one_dtype(data.dtypes()[0])
        #class_dtype = self._training_set.dtypes([self._training_set.columns[1]])[0]
        #class_name, class_type = parse_one_dtype(class_dtype)
        #training_index_dtype = self._training_set.dtypes([self._training_set.columns[0]])[0]
        #training_index_name, training_index_type = parse_one_dtype(training_index_dtype)
        result_tbl = '#KNN_PREDICT_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        #result_specs = [(index_name, index_type),
        #                (class_name, class_type)]
        stat_tbl = '#KNN_PREDICT_STAT_TBL_{}_{}'.format(self.id, unique_id)
        #stat_specs = [('TEST_' + index_name, index_type),
        #              ('K', INTEGER),
        #              ('TRAIN_' + training_index_name, training_index_type),
        #              ('DISTANCE', DOUBLE)]
        #training_tbl = "#KNN_TRAINING_DATA_TBL_{}".format(self.id, unique_id)
        tables = [result_tbl, stat_tbl]
        try:
            #self._materialize(training_tbl, self._training_set)
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_array))
            #self._create(Table(result_tbl, result_specs))
            #self._create(Table(stat_tbl, stat_specs))
            self._call_pal_auto("PAL_KNN",
                                self._training_set,
                                data,
                                ParameterTable().with_data(param_array),
                                *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop(tables)
            raise
        if self.stat_info:
            return self.conn_context.table(result_tbl), self.conn_context.table(stat_tbl)
        return self.conn_context.table(result_tbl)

    def score(self, data, key, features=None, label=None):#pylint:disable=too-many-locals
        """
        Return a scalar accuracy value after comparing the predicted
        and original label.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID and non-label columns.

        label : str, optional

            Name of the label column.
            If ``label`` is not provided, it defaults to the last column.

        Returns
        -------

        accuracy : float

            Scalar accuracy value after comparing the predicted label and
            original label.
        """

        if not hasattr(self, '_training_set'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols_left = data.columns
        cols_left.remove(key)
        if label is None:
            label = cols_left[-1]
        cols_left.remove(label)
        if features is None:
            features = cols_left
        if self.stat_info:
            prediction, _ = self.predict(data=data, key=key,
                                         features=features)
        else:
            prediction = self.predict(data=data, key=key,
                                      features=features)

        prediction = prediction.select(key, 'TARGET').rename_columns(['ID_P', 'PREDICTION'])

        actual = data.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')

        return metrics.accuracy_score(self.conn_context, joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')
