"""
This module includes mixture modeling algorithms.

The following classes are available:

    * :class:`GaussianMixture`
"""

#pylint: disable=too-many-locals,unused-variable, line-too-long, relative-beyond-top-level
import logging
from hdbcli import dbapi
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings
)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class GaussianMixture(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Representation of a Gaussian mixture model probability distribution.

    Parameters
    ----------
    conn_context : ConnectionContext
        Connection to the HANA system.
    init_param : {'farthest_first_traversal','manual','random_means','kmeans++'}
        Specifies the initialization mode.

          - farthest_first_traversal: The initial centers are given by
            the farthest-first traversal algorithm.
          - manual: The initial centers are the init_centers given by
            user.
          - random_means: The initial centers are the means of all the data
            that are randomly weighted.
          - kmeans++: The initial centers are given using the k-means++ approach.
    n_components : int
        Specifies the number of Gaussian distributions.
        Mandatory when `init_param` is not 'manual'.
    init_centers : list of int
        Specifies the data (by using sequence number of the data in the data
        table (starting from 0)) to be used as init_centers.
        Mandatory when `init_param` is 'manual'.
    covariance_type : {'full', 'diag', 'tied_diag'}, optional
        Specifies the type of covariance matrices in the model.

          - full: use full covariance matrices.
          - diag: use diagonal covariance matrices.
          - tied_diag: use diagonal covariance matrices with all equal
            diagonal entries.

        Defaults to 'full'.
    shared_covariance : bool, optional
        All clusters share the same covariance matrix if True.
        Defaults to False.
    thread_ratio : float, optional
        Controls the proportion of available threads that can be used.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this
        range tell PAL to heuristically determine the number of threads to use.
        Defaults to 0.
    max_iter : int, optional
        Specifies the maximum number of iterations for the EM algorithm.
        Default value: 100.
    categorical_variable : str or list of str, optional
        Specifies INTEGER column(s) that should be be treated as categorical.
        Other INTEGER columns will be treated as continuous.
    category_weight : float, optional
        Represents the weight of category attributes.
        Defaults to 0.707.
    error_tol : float, optional
        Specifies the error tolerance, which is the stop condition.
        Defaults to 1e-5.
    regularization : float, optional
        Regularization to be added to the diagonal of covariance matrices
        to ensure positive-definite.
        Defaults to 1e-6.
    random_seed : int, optional
        Indicates the seed used to initialize the random number generator:

          - 0: Uses the system time.
          - Not 0: Uses the provided value.

        Defaults to 0.

    Attributes
    ----------
    model_ : DataFrame
        Trained model content.
    labels_ : DataFrame
        Cluster membership probabilties for each data point.
    stats_ : DataFrame
        Statistics.

    Examples
    --------
    Input dataframe for training:

    >>> df1.collect()
        ID     X1     X2  X3
    0    0   0.10   0.10   1
    1    1   0.11   0.10   1
    2    2   0.10   0.11   1
    3    3   0.11   0.11   1
    4    4   0.12   0.11   1
    5    5   0.11   0.12   1
    6    6   0.12   0.12   1
    7    7   0.12   0.13   1
    8    8   0.13   0.12   2
    9    9   0.13   0.13   2
    10  10   0.13   0.14   2
    11  11   0.14   0.13   2
    12  12  10.10  10.10   1
    13  13  10.11  10.10   1
    14  14  10.10  10.11   1
    15  15  10.11  10.11   1
    16  16  10.11  10.12   2
    17  17  10.12  10.11   2
    18  18  10.12  10.12   2
    19  19  10.12  10.13   2
    20  20  10.13  10.12   2
    21  21  10.13  10.13   2
    22  22  10.13  10.14   2
    23  23  10.14  10.13   2

    Creating the GMM instance:

    >>> gmm = GaussianMixture(conn_context=cc,
    ...                       init_param='farthest_first_traversal',
    ...                       n_components=2, covariance_type='full',
    ...                       shared_covariance=False, max_iter=500,
    ...                       error_tol=0.001, thread_ratio=0.5,
    ...                       categorical_variable=['X3'], random_seed=1)

    Performing fit() on the given dataframe:

    >>> gmm.fit(df1, key='ID')
    >>> gmm.labels_.head(14).collect()
        ID  CLUSTER_ID  PROBABILITY
    0    0           0          0.0
    1    1           0          0.0
    2    2           0          0.0
    3    4           0          0.0
    4    5           0          0.0
    5    6           0          0.0
    6    7           0          0.0
    7    8           0          0.0
    8    9           0          0.0
    9    10          0          1.0
    10   11          0          1.0
    11   12          0          1.0
    12   13          0          1.0
    13   14          0          0.0
    """
    init_param_map = {'farthest_first_traversal': 0,
                      'manual': 1,
                      'random_means': 2,
                      'k_means++': 3}
    covariance_type_map = {'full': 0, 'diag': 1, 'tied_diag': 2}

    def __init__(self, #pylint: disable=too-many-arguments
                 conn_context,
                 init_param,
                 n_components=None,
                 init_centers=None,
                 covariance_type=None,
                 shared_covariance=False,
                 thread_ratio=None,
                 max_iter=None,
                 categorical_variable=None,
                 category_weight=None,
                 error_tol=None,
                 regularization=None,
                 random_seed=None):
        super(GaussianMixture, self).__init__(conn_context)
        self.init_param = self._arg('init_param',
                                    init_param,
                                    self.init_param_map,
                                    required=True)
        self.n_components = self._arg('n_components', n_components, int)
        self.init_centers = self._arg('init_centers', init_centers, list)
        if init_param == 'manual':
            if init_centers is None:
                msg = ("Parameter init_centers is required when init_param is manual.")
                logger.error(msg)
                raise ValueError(msg)
            if n_components is not None:
                msg = ("Parameter n_components is not applicable when " +
                       "init_param is manual.")
                logger.error(msg)
                raise ValueError(msg)
        else:
            if n_components is None:
                msg = ("Parameter n_components is required when init_param is " +
                       "farthest_first_traversal, random_means and k_means++.")
                logger.error(msg)
                raise ValueError(msg)
            if init_centers is not None:
                msg = ("Parameter init_centers is not applicable when init_param is " +
                       "farthest_first_traversal, random_means and k_means++.")
                logger.error(msg)
                raise ValueError(msg)
        self.covariance_type = self._arg('covariance_type',
                                         covariance_type,
                                         self.covariance_type_map)
        self.shared_covariance = self._arg('shared_covariance', shared_covariance, bool)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.category_weight = self._arg('category_weight',
                                         category_weight, float)
        self.error_tol = self._arg('error_tol', error_tol, float)
        self.regularization = self._arg('regularization', regularization, float)
        self.random_seed = self._arg('random_seed', random_seed, int)

    def fit(self, data, key, features=None, categorical_variable=None):#pylint: disable=invalid-name, too-many-locals
        """
        Perform GMM clustering on input dataset.

        Parameters
        ----------
        data : DataFrame
            Data to be clustered.
        key : str
            Name of the ID column.
        features : list of str, optional
            List of strings specifying feature columns.
            If a list of features is not given, all the columns except the ID column
            are taken as features.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.
            Other INTEGER columns will be treated as continuous.
        """
        key = self._arg('key', key, str, required=True)
        features = self._arg('features', features, ListOfStrings)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data = data[[key] + features]

        outputs = ['RESULT', 'MODEL', 'STATISTICS', 'PLACEHOLDER']
        outputs = ['#PAL_GMM_{}_TBL_{}'.format(name, self.id)
                   for name in outputs]
        result_tbl, model_tbl, statistics_tbl, placeholder_tbl = outputs
        init_param_data = self._prep_init_param()

        param_rows = [
            ("INIT_MODE", self.init_param, None, None),
            ("COVARIANCE_TYPE", self.covariance_type, None, None),
            ("SHARED_COVARIANCE", self.shared_covariance, None, None),
            ("CATEGORY_WEIGHT", None, self.category_weight, None),
            ("MAX_ITERATION", self.max_iter, None, None),
            ("THREAD_RATIO", None, self.thread_ratio, None),
            ("ERROR_TOL", None, self.error_tol, None),
            ("SEED", self.random_seed, None, None)
            ]
        if self.categorical_variable is not None:
            param_rows.extend(("CATEGORICAL_VARIABLE", None, None, variable)
                              for variable in self.categorical_variable)

        if categorical_variable is not None:
            param_rows.extend(("CATEGORICAL_VARIABLE", None, None, variable)
                              for variable in categorical_variable)

        try:
            param_t = ParameterTable(name='#PARAM_TBL')
            self._call_pal_auto('PAL_GMM',
                                data,
                                init_param_data,
                                param_t.with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop('#PARAM_TBL')
            self._try_drop(outputs)
            raise

        # pylint: disable=attribute-defined-outside-init
        self.model_ = self.conn_context.table(model_tbl)
        self.labels_ = self.conn_context.table(result_tbl)
        self.stats_ = self.conn_context.table(statistics_tbl)

    def fit_predict(self, data, key, features=None, categorical_variable=None):#pylint: disable=invalid-name, too-many-locals
        """
        Perform GMM clustering on input dataset and return cluster membership
        probabilties for each data point.

        Parameters
        ----------
        data : DataFrame
            Data to be clustered.
        key : str
            Name of the ID column.
        features : list of str, optional
            List of strings specifying feature columns.
            If a list of features is not given, all the columns except the ID column
            are taken as features.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) specified that should be treated as categorical.
            Other INTEGER columns will be treated as continuous.

        Returns
        -------
        DataFrame
            Cluster membership probabilities.
        """
        self.fit(data, key, features, categorical_variable)
        return self.labels_

    def _prep_init_param(self):

        init_param_tbl = '#PAL_GMM_INITIALIZE_PARAMETER_TBL'

        if self.n_components is not None:
            with self.conn_context.connection.cursor() as cur:
                cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("ID" INTEGER, "CLUSTER_NUMBER" INTEGER);'.format(init_param_tbl))
                cur.execute('INSERT INTO {} VALUES (0, {});'.format(init_param_tbl, self.n_components))
                init_param_data = self.conn_context.table(init_param_tbl)
        elif self.init_centers is not None:
            with self.conn_context.connection.cursor() as cur:
                cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {} ("ID" INTEGER, "SEEDS" INTEGER);'.format(init_param_tbl))
                for idx, val in enumerate(self.init_centers):
                    cur.execute('INSERT INTO {} VALUES ( {}, {} );'.format(init_param_tbl, idx, val))
                init_param_data = self.conn_context.table(init_param_tbl)
        return init_param_data
