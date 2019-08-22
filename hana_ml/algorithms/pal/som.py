"""This module contains PAL wrapper and helper functions for SOM algorithms.
The following classes are available:

    * :class:`SOM`
"""

#pylint: disable=too-many-lines, line-too-long, relative-beyond-top-level
import logging
import uuid
from hdbcli import dbapi
from hana_ml.algorithms.pal.clustering import _ClusterAssignmentMixin
from .pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings
)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class SOM(PALBase, _ClusterAssignmentMixin):#pylint: disable=too-many-instance-attributes
    r"""
    Self-organizing feature maps (SOMs) are one of the most popular
    neural network methods for cluster analysis.
    They are sometimes referred to as Kohonen self-organizing feature maps,
    after their creator, Teuvo Kohonen, or as topologically ordered maps.

    Parameters
    ----------
    conn_context : ConnectionContext
        Connection to the HANA system.

    convergence_criterion : float, optional
        If the largest difference of the successive maps is less than this value,
        the calculation is regarded as convergence, and SOM is completed consequently.

        Default to 1.0e-6.

    normalization : {'0', '1', '2'}, int, optional
        Normalization type:

        - 0: No
        - 1: Transform to new range (0.0, 1.0)
        - 2: Z-score normalization

        Default to 0.

    random_seed : {'1', '0', 'Other value'}, int, optional

        - 1: Random
        - 0: Sets every weight to zero
        - Other value: Uses this value as seed

       Default to -1.

    height_of_map : int, optional
        Indicates the height of the map.

        Default to 10.

    width_of_map : int, optional
        Indicates the width of the map.

        Default to 10.

    kernel_function : int, optional
        Represents the neighborhood kernel function.

        - 1: Gaussian
        - 2: Bubble/Flat

        Default to 1.

    alpha : float, optional
        Specifies the learning rate.

        Default to 0.5

    learning_rate : int, optional
        Indicates the decay function for learning rate.

        - 1: Exponential
        - 2: Linear

        Default to 1.

    shape_of_grid : int, optional
        Indicates the shape of the grid.

        - 1: Rectangle
        - 2: Hexagon

        Default to 2.

    radius : float, optional
        Specifies the scan radius.

        Defautl to the bigger value of ``height_of_map`` and ``width_of_map``.

    batch_som : {'0', '1'}, int, optional
        Indicates whether batch SOM is carried out.

        - 0: Classical SOM
        - 1: Batch SOM

        For batch SOM, ``kernel_function`` is always Gaussian,
        and the ``learning_rate`` factors take no effect.

        Default to 0.

    max_iter : int, optional
        Maximum number of iterations.
        Note that the training might not converge if this value is too small,
        for example, less than 1000.

        Default to 1000 plus 500 times the number of neurons in the lattice.

    Attributes
    ----------

    map_ : DataFrame
         The map after training. The structure is as follows:

        - 1st column: CLUSTER_ID, int. Unit cell ID.
        - Other columns except the last one: FEATURE (in input data)
          column with prefix "WEIGHT\_", float. Weight vectors used to simulate
          the original tuples.
        - Last column: COUNT, int. Number of original tuples that
          every unit cell contains.

    label_: DataFrame
        The label of input data. The structure is as follows:

            - 1st column: ID (in input table) data type, ID (in input table) column name
              ID of original tuples.
            - 2nd column: BMU, int. Best match unit (BMU).
            - 3rd column: DISTANCE, float, The distance between the tuple and its BMU.
            - 4th column: SECOND_BMU, int, Second BMU.
            - 5th column: IS_ADJACENT. int. Indicates whether the BMU and the second BMU are adjacent.
                - 0: Not adjacent
                - 1: Adjacent

    model_ : DataFrame
        The SOM model. The structure is as follows:

    Examples
    --------

    Input dataframe for clustering:

    >>> df.collect()
        TRANS_ID    V000    V001
    0      0        0.10    0.20
    1      1        0.22    0.25
    2      2        0.30    0.40
    ...
    18     18       55.30   50.40
    19     19       50.40   56.50

    Create SOM instance:

    >>> som = SOM(conn,covergence_criterion=1.0e-6, normalization=0,
                 random_seed=1, height_of_map=4, width_of_map=4,
                 kernel_function='gaussian', alpha=None,
                 learning_rate='exponential',shape_of_grid='hexagon',
                 radius=None, batch_som='classical', max_iter=4000)

    Perform fit on the given data:

    >>> som.fit(df, key='TRANS_ID')

    Expected output:

    >>> som.map_.collect().head(3)
            CLUSTER_ID	WEIGHT_V000    WEIGHT_V001    COUNT
        0    0          52.837688      53.465327      2
        1    1          50.150251      49.245226      2
        2    2          18.597607      27.174590      0

    >>> som.labels_.collect().head(3)
        TRANS_ID    BMU    DISTANCE    SECOND_BMU    IS_ADJACENT
        0           0      15          0.342564	14      1
        1           1      15          0.239676	14      1
        2           2      15          0.073968	14      1

    >>> som.model_.collect()
            ROW_INDEX      MODEL_CONTENT
      0      0             {"Algorithm":"SOM","Cluster":[{"CellID":0,"Cel...

    After we get the model, we could use it to predict
    Input dataframe for prediction:

    >>> df_predict.collect()
        TRANS_ID    V000    V001
    0      33       0.2     0.10
    1      34       1.2     4.1

    Preform predict on the givenn data:

    >>> label = som.predict(df2, key='TRANS_ID')

    Expected output:

    >>> label.collect()
        TRANS_ID    CLUSTER_ID     DISTANCE
    0    33          15            0.388460
    1    34          11            0.156418

 """

    kernel_function_map = {'gaussian':1, 'flat':2}
    learning_rate_map = {'exponential':1, 'linear':2}
    shape_of_grid_map = {'rectangle':1, 'hexagon':2}
    batch_som_map = {'classical':0, 'batch':1}

    def __init__(self, conn_context, covergence_criterion=None, normalization=None,#pylint: disable=too-many-arguments
                 random_seed=None, height_of_map=None, width_of_map=None,
                 kernel_function=None, alpha=None, learning_rate=None,
                 shape_of_grid=None, radius=None, batch_som=None, max_iter=None):

        super(SOM, self).__init__(conn_context)

        self.covergence_criterion = self._arg('covergence_criterion', covergence_criterion, float)
        self.normalization = self._arg('normalization', normalization, int)
        self.random_seed = self._arg('random_seed', random_seed, int)
        self.height_of_map = self._arg('height_of_map', height_of_map, int)
        self.width_of_map = self._arg('width_of_map', width_of_map, int)
        self.kernel_function = self._arg('kernel_function', kernel_function, self.kernel_function_map)
        self.alpha = self._arg('alpha', alpha, float)
        self.learning_rate = self._arg('learning_rate', learning_rate, self.learning_rate_map)
        self.shape_of_grid = self._arg('shape_of_grid', shape_of_grid, self.shape_of_grid_map)
        self.radius = self._arg('radius', radius, float)
        self.batch_som = self._arg('batch_som', batch_som, self.batch_som_map)
        self.max_iter = self._arg('max_iter', max_iter, int)

    def fit(self, data, key, features=None):#pylint: disable=too-many-locals
        r"""
        Fit the SOM model when given the training dataset.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the features columns.
            If `features` is not provided, it defaults to all the non-ID
            columns.
        """

        key = self._arg('key', key, str, True)
        features = self._arg('features', features, ListOfStrings)
        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#SOM_{}_TBL_{}_{}'.format(name, self.id, unique_id) for
                   name in ['MAP', 'ASSIGNMENT', 'MODEL', 'STAT', 'PL']]
        #pylint: disable=unused-variable
        map_tbl, assignment_tbl, model_tbl, statistics_tbl, placeholder_tbl = outputs[:5]

        param_array = [('COVERGENCE_CRITERION', None, self.covergence_criterion, None),
                       ('NORMALIZATION', self.normalization, None, None),
                       ('RANDOM_SEED', self.random_seed, None, None),
                       ('HEIGHT_OF_MAP', self.height_of_map, None, None),
                       ('WIDTH_OF_MAP', self.width_of_map, None, None),
                       ('KERNEL_FUNCTION', self.kernel_function, None, None),
                       ('ALPHA', None, self.alpha, None),
                       ('LEARNING_RATE', self.learning_rate, None, None),
                       ('SHAPE_OF_GRID', self.shape_of_grid, None, None),
                       ('RADIUS', None, self.radius, None),
                       ('BATCH_SOM', self.batch_som, None, None),
                       ('MAX_ITERATION', self.max_iter, None, None)]

        try:
            self._call_pal_auto('PAL_SOM',
                                data[[key] + features],
                                ParameterTable().with_data(param_array),
                                *outputs)
        except dbapi.Error:
            logger.exception('HANA error during PAL SOM fit')
            raise
        self.map_ = self.conn_context.table(map_tbl)#pylint:disable=attribute-defined-outside-init
        self.labels_ = self.conn_context.table(assignment_tbl)#pylint:disable=attribute-defined-outside-init
        self.model_ = self.conn_context.table(model_tbl)#pylint:disable=attribute-defined-outside-init

    def fit_predict(self, data, key, features=None):
        r"""
        Fit the dataset and return the labels.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str

            Name of the ID column.

        features : list of str, optional

            Names of the features columns.
            If `features` is not provided, it defaults to all the non-ID
            columns.

        Returns
        -------

         The label of given data. The structure is as follows:

            - 1st column: ID (in input table) data type, ID (in input table) column name
              ID of original tuples.
            - 2nd column: BMU, int. Best match unit (BMU).
            - 3rd column: DISTANCE, float, The distance between the tuple and its BMU.
            - 4th column: SECOND_BMU, int, Second BMU.
            - 5th column: IS_ADJACENT. int. Indicates whether the BMU and the second BMU are adjacent.
                - 0: Not adjacent
                - 1: Adjacent
        """

        self.fit(data, key, features)
        return self.labels_

    def predict(self, data, key, features=None):
        r"""
        Assign clusters to data based on a fitted model.

        The output structure of this method does not match that of
        fit_predict().

        Parameters
        ----------

        data : DataFrame

            Data points to match against computed clusters.
            This dataframe's column structure should match that
            of the data used for fit().

        key : str

            Name of ID column.

        features : list of str, optional.

            Names of feature columns.
            If ``features`` is not provided, it defaults to all
            the non-ID columns.

        Returns
        -------

        DataFrame

            Cluster assignment results, with 3 columns:

              - Data point ID, with name and type taken from the input
                ID column.
              - CLUSTER_ID, type int, representing the cluster the
                data point is assigned to.
              - DISTANCE, type DOUBLE, representing the distance between
                the data point and the nearest core point.
        """

        return super(SOM, self)._predict(data, key, features)
