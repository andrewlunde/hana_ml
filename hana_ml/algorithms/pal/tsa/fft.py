"""
This module contains Python wrappers for PAL Fast-Fourier-Transform(FFT) algorithm.

The following classes are available:

    * :class:`FFT`
"""

# pylint:disable=line-too-long, too-few-public-methods
import logging
import uuid
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    #Table,
    ParameterTable,
    #INTEGER,
    #DOUBLE
)
logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class FFT(PALBase):
    r"""
    Fast Fourier Transform to apply to discrete data sequence.

    Parameters
    ----------

    conn_context : ConnectionContext

        Connection to the HANA system.

    Attributes
    ----------

    None

    Examples
    --------

    Training data:

    >>> df.collect()
       ID   RE   IM
    0   1  2.0  9.0
    1   2  3.0 -3.0
    2   3  5.0  0.0
    3   4  0.0  0.0
    4   5 -2.0 -2.0
    5   6 -9.0 -7.0
    6   7  7.0  0.0

    Create an FFT instance:

    >>> fft = FFT(cc)

    Call apply() on given data sequence:

    >>> result = fft.apply(df, inverse=False)
    >>> result.collect()
       ID       REAL       IMAG
    0   1   6.000000  -3.000000
    1   2  16.273688  -0.900317
    2   3  -5.393946  26.265112
    3   4 -13.883222  18.514840
    4   5  -4.233990  -2.947800
    5   6   9.657319   3.189618
    6   7   5.580151  21.878547

    """

    number_type_map = {'real':1, 'imag':2}
    def apply(self, data, num_type=None, inverse=None):
        """
        Apply Fast-Fourier-Transfrom(FFT) to the input data, and return the transformed data.

        Parameters
        ----------

        data : DataFrame

            DataFrame to apply FFT to, which contains at most 3 columns. \
            First column of the Input Data must be ID, which indicates order \
            and must be of INTEGER type; other columns indicates the real/imaginary parts.

        num_type : {'real', 'imag'}, optional

            Number type for the second column of the input data.
            Valid only when the input data contains 3 columns.
            Default value is 'real'.

        inverse : bool, optional
            If False, forward FFT is applied; otherwise inverse FFT is applied.
            Default value is False.

        Returns
        -------

        result : DataFrame
            Dataframe containing the transformed sequence, structured as follows:

                - 1st column: ID, with same name and type as input data
                - 2nd column: REAL, type DOUBLE, representing real part of the transformed sequence
                - 3rd column: IMAG, type DOUBLE, represneting imaginary part of the transformed sequence

        """
        inverse = self._arg('inverse', inverse, bool)
        num_type = self._arg('num_type', num_type, self.number_type_map)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        #tables = ['DATA', 'PARAM', 'RESULT']
        result_tbl = '#PAL_FFT_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        #data_tbl, param_tbl, result_tbl = tables

        param_rows = [
            ('INVERSE', None if inverse is None else int(inverse), None, None),
            ('NUMBER_TYPE',
             None if num_type is None else int(num_type),
             None, None)
            ]
        #result_spec = [
        #    ("ID", INTEGER),
        #    ("REAL", DOUBLE),
        #    ("IMAG", DOUBLE)
        #    ]
        try:
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))

            self._call_pal_auto("PAL_FFT",
                                data,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            #msg = ('HANA error while attempting to apply FFT.')
            logger.exception(str(db_err))
            self._try_drop(result_tbl)
            raise

        return self.conn_context.table(result_tbl)
