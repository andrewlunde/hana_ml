"""
This module contains python wrapper of PAL link prediction function.

The following classes are available:

    * :class:`LinkPrediction`
"""
import logging
import uuid

from hdbcli import dbapi
from .pal_base import (
    PALBase,
    ParameterTable
)

logger = logging.getLogger(__name__)#pylint:disable=invalid-name


class LinkPrediction(PALBase):#pylint:disable=too-few-public-methods
    """
    Link predictor for calculating, in a network, proximity scores between
    nodes that are not directly linked, which is helpful for predicting missing
    links(the higher the proximity score is, the more likely the two nodes are
    to be linked).

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    method : {'common_neighbors', 'jaccard', 'adamic_adar', 'katz'}

        Method for computing the proximity between 2 nodes that are not directly
        linked.

    beta : float, optional

        A parameter included in the calculation of Katz similarity(proximity) score.
        Valid only when ``method`` is 'katz'.

        Defaults to 0.005.

    min_score : float, optional

        The links whose scores are lower than 'min_score' will be filtered out
        from the result table.

        Defaults to 0.

    Attributes
    ----------

    Examples
    --------
    Input network data:

    >>> df.collect()
       NODE1  NODE2
    0      1      2
    1      1      4
    2      2      3
    3      3      4
    4      5      1
    5      6      2
    6      7      4
    7      7      5
    8      6      7
    9      5      4

    Create an instance of 'LinkPrediction' object, and choose 'common_neighbors'
    as the method for proximity score calculation:

    >>> lp = LinkPrediction(conn_context=conn,
    ...                     method='common_neighbors',
    ...                     beta=0.005,
    ...                     min_score=0,
    ...                     thread_ratio=0.2)

    Calculate the proximity score of all nodes in the network with
    missing links, and check the result:

    >>> res = lp.proximity_score(df, node1='NODE1', node2='NODE2')
    >>> res.collect()
        NODE1  NODE2     SCORE
    0       1      3  0.285714
    1       1      6  0.142857
    2       1      7  0.285714
    3       2      4  0.285714
    4       2      5  0.142857
    5       2      7  0.142857
    6       4      6  0.142857
    7       3      5  0.142857
    8       3      6  0.142857
    9       3      7  0.142857
    10      5      6  0.142857
    """
    #pylint:disable=too-many-arguments, too-many-locals
    method_map = {'common_neighbors' : 1, 'jaccard' : 2,
                  'adamic_adar' : 3, 'katz' : 4}
    def __init__(self,
                 conn_context,
                 method,
                 beta=None,
                 min_score=None,
                 thread_ratio=None):
        super(LinkPrediction, self).__init__(conn_context)
        self.method = self._arg('method', method, self.method_map, required=True)
        if self.method == 4:
            self.beta = self._arg('beta', beta, float)
            #if self.beta is not None:
            #    if self.beta > 1 or self.beta < 0:
            #        msg = ("Input value of 'beta' is not between 0 and 1.")
            #        logger.error(msg)
            #        raise ValueError(msg)
        self.min_score = self._arg('min_score', min_score, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def proximity_score(self, data, node1=None, node2=None):#pylint:disable=invalid-name
        """
        For predicting proximity scores between nodes under current choice
        of method.

        Parameters
        ----------

        data : DataFrame

            Network data with nodes and links. \
            Nodes are in columns while links in rows, where each link \
            is represented by a pair of adjacent nodes as follows \
            (node1, node2).

        node1 : str, optional

            Column name of ``data`` that gives 'node1' of all available \
            links (see ``data``).

            Defaults to the name of the first column of ``data`` if not provided.

        node2 : str, optional

            Column name of ``data`` that gives 'node2' of all available \
            links (see ``data``).

            Defaults to the name of the last column of ``data`` if not provided.

        Returns
        -------

        DataFrame:

            The proximity scores of pairs of nodes with missing links \
            between them that are above 'min_score', structured \
            as follows:

                - 1st column: 'node1' of a link
                - 2nd column: 'node2' of a link
                - 3rd column:  proximity score of the two nodes
        """
        cols = data.columns
        if len(cols) < 2:
            msg = ("Input data contains less than 2 columns, which is "+
                   "insufficient for proximity score calculation.")
            logger.error(msg)
            raise ValueError(msg)
        source = self._arg('node1', node1, str)#pylint:disable=invalid-name
        if source is None:
            source = cols[0]
        sink = self._arg('node2', node2, str)#pylint:disable=invalid-name
        if sink is None:
            sink = cols[-1]
        used_cols = [source, sink]
        data = data[used_cols]
        param_rows = [
            ('METHOD', self.method, None, None),
            ('MIN_SCORE', None, self.min_score, None),
            ('THREAD_RATIO', None, self.thread_ratio, None)
        ]
        if self.method == 4:
            param_rows.extend([('BETA', None, self.beta, None)])
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_LINK_PREDICT_RESULT_TBL_{}_{}".format(self.id, unique_id)
        try:
            self._call_pal_auto('PAL_LINK_PREDICT',
                                data,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.error(str(db_err))
            self._try_drop(result_tbl)
            raise
        return self.conn_context.table(result_tbl)
