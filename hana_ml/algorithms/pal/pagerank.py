"""This module contains PAL wrapper for PageRank algorithms.

The following classes are available:

    * :class:`PageRank`
"""

import logging
import uuid
from hdbcli import dbapi
from .pal_base import (
    PALBase,
    ParameterTable
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class PageRank(PALBase):#pylint:disable=too-few-public-methods
    r"""
    A page rank model.

    Parameters
    ----------

    conn_context : ConnectionContext
        Connection to the HANA system.
    damping: float, optional
        The damping factor d.

        Defaults to 0.85.
    max_iter: int, optional
        The maximum number of iterations of power method.
        The value 0 means no maximum number of iterations is set
        and the calculation stops when the result converges.

        Defaults to 0.
    tol: float, optional
        Specifies the stop condition.
        When the mean improvement value of ranks is less than this value,
        the program stops calculation.

        Defaults to 1e-6.
    thread_ratio: float, optional
        Specifies the ratio of total number of threads that can be used by this function.
        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means
        using at most all the currently available threads. Values outside the range will be
        ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    None

    Examples
    --------

    Training data:

    >>> df.collect()
       FROM_NODE    TO_NODE
    0   NODE1       NODE2
    1   NODE1       NODE3
    2   NODE1       NODE4
    3   NODE2       NODE3
    4   NODE2       NODE4
    5   NODE3       NODE1
    6   NODE4       NODE1
    7   NODE4       NODE3

    Create a PageRank instance:

    >>> pr = PageRank(cc)

    Call run() on given data sequence:

    >>> result = pr.run(df)
    >>> result.collect()
       NODE     RANK
    0   NODE1   0.368152
    1   NODE2   0.141808
    2   NODE3   0.287962
    3   NODE4   0.202078
    """
    #pylint: disable=too-many-arguments
    def __init__(self,
                 conn_context,
                 damping=None, # float
                 max_iter=None, # int
                 tol=None, # float
                 thread_ratio=None):  # float
        super(PageRank, self).__init__(conn_context)
        self.conn_context = conn_context
        self.damping = self._arg('damping', damping, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.tol = self._arg('tol', tol, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

    def run(self, data):
        r"""
        This method reads link information and calculates rank for each node.

        Parameters
        ----------

        data : DataFrame
            Data for predicting the class labels.

        Returns
        -------

        DataFrame:
            Calculated rank values and corresponding node names, structured as follows:

            - NODE: node names.
            - RANK: the PageRank of the corresponding node.

        """
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_PAGERANK_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        param_rows = [
            ('DAMPING', None, None if self.damping is None else float(self.damping), None),
            ('MAX_ITERATION', None if self.max_iter is None else int(self.max_iter), None, None),
            ('THRESHOLD', None, None if self.tol is None else float(self.tol), None),
            ('THREAD_RATIO', None,
             None if self.thread_ratio is None else float(self.thread_ratio), None)
            ]
        try:
            self._call_pal_auto('PAL_PAGERANK',
                                data,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            self._try_drop([result_tbl])
            raise
        return self.conn_context.table(result_tbl)
