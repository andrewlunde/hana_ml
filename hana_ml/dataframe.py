"""
This module represents a database query as a dataframe.
Most operations are designed to not bring data back from the database
unless explicitly asked for.

The following classes and functions are available:

    * :class:`ConnectionContext`
    * :class:`DataFrame`
    * :func:`quotename`
    * :func:`create_dataframe_from_pandas`
"""
#pylint: disable=too-many-lines
#pylint: disable=line-too-long
#pylint: disable=relative-beyond-top-level
#pylint: disable=fixme
#pylint: disable=deprecated-lambda
#pylint: disable=too-many-locals
#pylint: disable=too-many-branches

import logging
import sys

import numpy as np
import pandas as pd
# import six

from hdbcli import dbapi
from .ml_exceptions import BadSQLError
from .type_codes import get_type_code_map

TYPE_CODES = get_type_code_map()
logger = logging.getLogger(__name__) #pylint: disable=invalid-name

if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INTEGER_TYPES = (int, long)
    _STRING_TYPES = (str, unicode)
else:
    _INTEGER_TYPES = (int,)
    _STRING_TYPES = (str,)


def quotename(name):
    """
    Escape a schema, table, or column name for use in SQL.

    hana_ml functions and methods that take schema, table, or column names
    already escape their input by default, but those that take SQL don't
    (and can't) perform escaping automatically.

    Parameters
    ----------
    name : str
        Schema, table, or column name.

    Returns
    -------
    str
        Escaped name. The string is surrounded in quotation marks,
        and existing quotation marks are escaped by doubling them.
    """
    return '"{}"'.format(name.replace('"', '""'))

class ConnectionContext(object):
    """
    Represents a connection to a HANA system.

    ConnectionContext includes methods for creating DataFrames from data
    on HANA. DataFrames are tied to a ConnectionContext, and are unusable
    once their ConnectionContext is closed.

    Parameters
    ----------
    Same as hdbcli.dbapi.connect. (See notes.)

    The online docs for hdbcli.dbapi.connect are available at https://help.sap.com/viewer/0eec0d68141541d1b07893a39944924e/latest/en-US/ee592e89dcce4480a99571a4ae7a702f.html.

    Examples
    --------
    Querying data from HANA into a Pandas DataFrame:

    >>> with ConnectionContext('address', port, 'user', 'password') as cc:
    ...     df = (cc.table('MY_TABLE', schema='MY_SCHEMA')
    ...             .filter('COL3 > 5')
    ...             .select('COL1', 'COL2'))
    ...     pandas_df = df.collect()

    The underlying hdbcli.dbapi.Connection can be accessed if necessary:

    >>> with ConnectionContext('127.0.0.1', 30215, 'MLGUY', 'manager') as cc:
    ...     cc.connection.setclientinfo('SOMEKEY', 'somevalue')
    ...     df = cc.sql('some sql that needs that session variable')
    ...     ...

    Attributes
    ----------
    connection : hdbcli.dbapi.Connection
        Underlying dbapi connection. This can be used to run SQL directly,
        or to access connection methods like getclientinfo/setclientinfo.
    """

    def __init__(self,            #pylint: disable=too-many-arguments
                 address='',
                 port=0,
                 user='',
                 password='',
                 autocommit=True,
                 packetsize=None,
                 userkey=None,
                 **properties):
        self.connection = dbapi.connect(
            address,
            port,
            user,
            password,
            autocommit=autocommit,
            packetsize=packetsize,
            userkey=userkey,
            **properties
        )
        self._pal_check_passed = False
        self.sql_tracer = SqlTrace() # SQLTRACE

    def hana_version(self):
        """
        Return the version of HANA.

        Parameters
        ----------
        None

        Returns
        -------
        str
            HANA version.
        """
        return DataFrame(self, "SELECT VALUE FROM SYS.M_SYSTEM_OVERVIEW WHERE NAME='Version'").collect().iat[0, 0]

    def close(self):
        """
        Closes the existing connection.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def sql(self, sql):
        """
        Returns a DataFrame representing a query.

        Parameters
        ----------
        sql : str
            SQL query.

        Returns
        -------
        DataFrame
            DataFrame for the query.

        Examples
        --------
        >>> df = cc.sql('SELECT T.A, T2.B FROM T, T2 WHERE T.C=T2.C')
        """
        return DataFrame(self, sql)

    def table(self, table, **kwargs):
        """
        cc.table(table, *, schema=None)

        Returns a DataFrame representing the specified table.

        Parameters
        ----------
        table : str
            Table name.
        schema : str, optional, keyword-only
            Schema name. If not provided or set to None, defaults to the
            ConnectionContext's current schema.

        Returns
        -------
        DataFrame
            DataFrame selecting data from that table.

        Examples
        --------
        >>> df1 = cc.table('MY_TABLE')
        >>> df2 = cc.table('MY_OTHER_TABLE', schema='MY_SCHEMA')
        """
        schema = kwargs.pop('schema', None)
        if kwargs:
            message = 'table() got an unexpected keyword argument {!r}'.format(
                next(iter(kwargs)))
            raise TypeError(message)
        if schema is None:
            table_reference = quotename(table)
        else:
            table_reference = '.'.join(map(quotename, (schema, table)))
        select = 'SELECT * FROM {}'.format(table_reference)

        result = DataFrame(self, select)

        # pylint: disable=protected-access
        if table.startswith('#'):
            result._ttab_handling = 'ttab'
            result._ttab_reference = table_reference
        else:
            result._ttab_handling = 'safe'

        # SQLTRACE
        # Checking the trace_sql_active is unnecessary however as this is
        # done in the sql_tracer object as well. But as not to impact
        # current test cases this has been added. If the ehck is removed 2
        # test cases will fail. Changing the test cases would be the better
        # option going forward
        if self.sql_tracer.trace_sql_active:
            self.sql_tracer.trace_object({
                'name': table,
                'table_type': result.generate_table_type(),
                'select': result.select_statement,
                'reference': table_reference,
                'schema': schema
            }, sub_cat='output_tables')

        return result

# SQLTRACE Class for Core Functions
class SqlTrace(object):
    """
    Class to provide functions to track the generated sql.

    It stores the trace in a dictionary in the following format:

    {
        'algorithm':{
            'function':{
                'subcategory':[]
            }
        }
    }

    Attributes
    ----------
    trace_sql_log : dictionary
        The SQL Trace dictionary
    trace_sql_active : boolean
        Flag to define if tracing should occur
    trace_sql_algo : string
        Current algorithm under which is being traced
    trace_sql_function : string
        Current function under which is being traced
    trace_history : boolean
        Track multiple runs of the same algo. If enabled
        the algo_tracker will track the count of the algorithm
        and will add a sequence number to the algorithm name in the dictionary
        so that each run is traced seperately.
    trace_algo_tracker : dictionary
        If trace_history is enabled tracks the count of the number of times
        the same algorithm is run

    Examples
    --------
    >>> Example snippet of the SQL trace dictionary:
    {
        "RandomForestClassifier": {
            "Fit": {
                "input_tables": [
                    {
                        "name": "#PAL_RANDOM_FOREST_DATA_TBL_0",
                        "type": "table (\"AccountID\" INT,\"ServiceType\" VARCHAR(21),\"ServiceName\" VARCHAR(14),\"DataAllowance_MB\" INT,\"VoiceAllowance_Minutes\" INT,\"SMSAllowance_N_Messages\" INT,\"DataUsage_PCT\" DOUBLE,\"DataUsage_PCT_PM\" DOUBLE,\"DataUsage_PCT_PPM\" DOUBLE,\"VoiceUsage_PCT\" DOUBLE,\"VoiceUsage_PCT_PM\" DOUBLE,\"VoiceUsage_PCT_PPM\" DOUBLE,\"SMSUsage_PCT\" DOUBLE,\"SMSUsage_PCT_PM\" DOUBLE,\"SMSUsage_PCT_PPM\" DOUBLE,\"Revenue_Month\" DOUBLE,\"Revenue_Month_PM\" DOUBLE,\"Revenue_Month_PPM\" DOUBLE,\"Revenue_Month_PPPM\" DOUBLE,\"ServiceFailureRate_PCT\" DOUBLE,\"ServiceFailureRate_PCT_PM\" DOUBLE,\"ServiceFailureRate_PCT_PPM\" DOUBLE,\"CustomerLifetimeValue_USD\" DOUBLE,\"CustomerLifetimeValue_USD_PM\" DOUBLE,\"CustomerLifetimeValue_USD_PPM\" DOUBLE,\"Device_Lifetime\" INT,\"Device_Lifetime_PM\" INT,\"Device_Lifetime_PPM\" INT,\"ContractActivityLABEL\" VARCHAR(8))",
                        "select": "SELECT \"AccountID\", \"ServiceType\", \"ServiceName\", \"DataAllowance_MB\", \"VoiceAllowance_Minutes\", \"SMSAllowance_N_Messages\", \"DataUsage_PCT\", \"DataUsage_PCT_PM\", \"DataUsage_PCT_PPM\", \"VoiceUsage_PCT\", \"VoiceUsage_PCT_PM\", \"VoiceUsage_PCT_PPM\", \"SMSUsage_PCT\", \"SMSUsage_PCT_PM\", \"SMSUsage_PCT_PPM\", \"Revenue_Month\", \"Revenue_Month_PM\", \"Revenue_Month_PPM\", \"Revenue_Month_PPPM\", \"ServiceFailureRate_PCT\", \"ServiceFailureRate_PCT_PM\", \"ServiceFailureRate_PCT_PPM\", \"CustomerLifetimeValue_USD\", \"CustomerLifetimeValue_USD_PM\", \"CustomerLifetimeValue_USD_PPM\", \"Device_Lifetime\", \"Device_Lifetime_PM\", \"Device_Lifetime_PPM\", \"ContractActivityLABEL\" FROM (SELECT a.* FROM #PAL_PARTITION_DATA_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C a inner join #PAL_PARTITION_RESULT_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C b        on a.\"AccountID\" = b.\"AccountID\" where b.\"PARTITION_TYPE\" = 1) AS \"DT_2\""
                    }
                ],
                "internal_tables":[
                    {
                        "name": "#PAL_RANDOM_FOREST_PARAM_TBL_0",
                        "type": [
                            [
                                "PARAM_NAME",
                                "NVARCHAR(5000)"
                            ],
                            [
                                "INT_VALUE",
                                "INTEGER"
                            ],
                            [
                                "DOUBLE_VALUE",
                                "DOUBLE"
                            ],
                            [
                                "STRING_VALUE",
                                "NVARCHAR(5000)"
                            ]
                        ]
                    }
                ],
                "output_tables":[
                    {
                        "name": "#PAL_RANDOM_FOREST_MODEL_TBL_0",
                        "type": "table (\"ROW_INDEX\" INT,\"TREE_INDEX\" INT,\"MODEL_CONTENT\" NVARCHAR(5000))",
                        "select": "SELECT * FROM \"#PAL_RANDOM_FOREST_MODEL_TBL_0\"",
                        "reference": "\"#PAL_RANDOM_FOREST_MODEL_TBL_0\"",
                        "schema": null
                    }
                ],
                "function":[
                    {
                        "name": "PAL_RANDOM_DECISION_TREES",
                        "schema": "_SYS_AFL",
                        "type": "pal"
                    }
                ],
                "sql":[
                    "DROP TABLE \"#PAL_RANDOM_FOREST_DATA_TBL_0\"",
                    "CREATE LOCAL TEMPORARY COLUMN TABLE \"#PAL_RANDOM_FOREST_DATA_TBL_0\" AS (SELECT \"AccountID\", \"ServiceType\", \"ServiceName\", \"DataAllowance_MB\", \"VoiceAllowance_Minutes\", \"SMSAllowance_N_Messages\", \"DataUsage_PCT\", \"DataUsage_PCT_PM\", \"DataUsage_PCT_PPM\", \"VoiceUsage_PCT\", \"VoiceUsage_PCT_PM\", \"VoiceUsage_PCT_PPM\", \"SMSUsage_PCT\", \"SMSUsage_PCT_PM\", \"SMSUsage_PCT_PPM\", \"Revenue_Month\", \"Revenue_Month_PM\", \"Revenue_Month_PPM\", \"Revenue_Month_PPPM\", \"ServiceFailureRate_PCT\", \"ServiceFailureRate_PCT_PM\", \"ServiceFailureRate_PCT_PPM\", \"CustomerLifetimeValue_USD\", \"CustomerLifetimeValue_USD_PM\", \"CustomerLifetimeValue_USD_PPM\", \"Device_Lifetime\", \"Device_Lifetime_PM\", \"Device_Lifetime_PPM\", \"ContractActivityLABEL\" FROM (SELECT a.* FROM #PAL_PARTITION_DATA_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C a inner join #PAL_PARTITION_RESULT_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C b        on a.\"AccountID\" = b.\"AccountID\" where b.\"PARTITION_TYPE\" = 1) AS \"DT_2\")",
                    "DROP TABLE \"#PAL_RANDOM_FOREST_PARAM_TBL_0\"",
                    "CREATE LOCAL TEMPORARY COLUMN TABLE \"#PAL_RANDOM_FOREST_PARAM_TBL_0\" (\n    \"PARAM_NAME\" NVARCHAR(5000),\n    \"INT_VALUE\" INTEGER,\n    \"DOUBLE_VALUE\" DOUBLE,\n    \"STRING_VALUE\" NVARCHAR(5000)\n)"
                ]
        }

    }
    """
    def __init__(self):
        self.trace_sql_log = {}
        self.trace_sql_active = False
        self.trace_sql_algo = None
        self.trace_sql_function = None
        self.trace_history = False
        self.trace_algo_tracker = {}

    def enable_sql_trace(self, enable):
        """
        Enable or disable the sql trace

        Parameters
        ----------
        enable : boolean
            flag to enable or disable the sql trace
        """
        self.trace_sql_active = enable

    def set_sql_trace(self, algo_object=None, algo=None, function=None):
        """
        Activates the trace for a certain algo and function. Any subsequent calls will be placed
        under the respective substructure (algo -> function) in the trace dictionary.

        The algo_oject is the actual algorithm instance object from a algorithm class. This is used
        in case history of multiple calls to the same algorithm needs to be traced to check if
        the same object is being used or not.

        Parameters
        ----------
        algo_object : object
            The actual algorithm object.
        algo : string
            Algorithm name
        function : string
            Algorithm function
        """

        if self.trace_history and algo_object:
            new_id = 0
            # Check if we already have run the algo based on the instance object.
            if str(id(algo_object)) in self.trace_algo_tracker:
                new_id = self.trace_algo_tracker[str(id(algo_object))]
            elif algo in self.trace_algo_tracker:
                new_id = self.trace_algo_tracker[algo] + 1
                self.trace_algo_tracker[algo] = new_id
                self.trace_algo_tracker[str(id(algo_object))] = new_id
            else:
                new_id += 1
                self.trace_algo_tracker[algo] = new_id
                self.trace_algo_tracker[str(id(algo_object))] = new_id
            algo += str(new_id)
        self.trace_sql_algo = algo
        self.trace_sql_function = function

    def enable_trace_history(self, enable):
        """
        Enable the trace history on algorithm level. This allows for multiple calls to the same algorithm to be stored
        seperately in the dictionary with a sequence number attached the algorith name. Please be aware that this
        is currently only on algorithm level. So using the same algorithm instance and calling the same function (ie Fit)
        twice would still overwrite the the previous call.

        Parameters
        ----------
        enable : boolean
            flag to enable or disable the trace history
        """
        self.trace_history = enable

    def get_sql_trace(self):
        """
        Returns the sql trace distionary

        Returns
        -------
        SQL Trace : dict
            SQL Trace dictionary object
        """
        return self.trace_sql_log

    def trace_sql(self, value=None):
        """
        Add the sql value to the current active algorithm and funtion in the sql trace dictionary

        Parameters
        ----------
        value : str
            the sql entry
        """
        self._trace_data(value, 'sql')

    def trace_object(self, value=None, sub_cat='nocat'):
        """
        Trace additional objects outside of the sql entries itself. This to support more finegrained context
        and information that the sql trace consists of. For example the input tables, output tables and function
        being used for example. These are convenience objects to be able to more easily distill how the sql is being structured. Generally
        speaking these objects are dictionaries themselves and that is the current use case. However it is not required to be.

        Parameters
        ----------
        value : object
            Ambiguous type of object to provide additional context outside of the sql trace itself.
        sub_cat: str
            The subcategory or key under which the value needs to be placed. For example 'output_tables'

        """
        self._trace_data(value, sub_cat)

    def trace_sql_many(self, statement, param_entries=None):
        """
        The trace sql many is to convert the insert executemany method on the hdbcli cursor object to actual multiple insert SQL statements.
        This is to assure we are dealing with with pure sql only.

        Parameters
        ----------
        statement : str
            The sql statement
        param_entries : list of tuple, or None
            The data of the insert statement
        """
        if self.trace_sql_active:
            for param_entry in param_entries:
                processed_statement = statement
                for param_value in param_entry:
                    if isinstance(param_value, str):
                        processed_statement = processed_statement.replace('?', "'"+param_value+"'", 1)
                    else:
                        processed_statement = processed_statement.replace('?', str(param_value), 1)
                # Additional processing to assure proper SQL
                processed_statement = processed_statement.replace('None', 'null')
                processed_statement = processed_statement.replace('True', '1')
                processed_statement = processed_statement.replace('False', '0')
                self.trace_sql(processed_statement)

    def _trace_data(self, value=None, sub_cat='nocat'):
        """
        The method that does the actual storage of the data in the sql trace

        Parameters
        ----------
        value : str or dict
            The value that needs to be stored in the dictionary
        sub_cat : str
            The sub category under the function key where the data needs to be stored.
            the sub_cat will become the key in the dict

        """
        if self.trace_sql_active:
            if not self.trace_sql_algo in self.trace_sql_log:
                self.trace_sql_log[self.trace_sql_algo] = {}
            if not self.trace_sql_function in self.trace_sql_log[self.trace_sql_algo]:
                self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function] = {}
            if not sub_cat in self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function]:
                self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function][sub_cat] = []

            self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function][sub_cat].append(value)

class DataFrame(object):#pylint: disable=too-many-public-methods
    """
    This class represents a frame that is backed by a database sql statement.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection to the HANA system.
    select_statement : str
        SQL query backing the dataframe.

        .. note ::
            Parameters beyond `connection_context` and `select_statement` are intended for internal usage. Do not rely on them; they may change without notice.
    """
    #pylint: disable=too-many-public-methods,too-many-instance-attributes
    _df_count = 0
    def __init__(self, connection_context, select_statement, _name=None):
        self.connection_context = connection_context
        self.select_statement = select_statement
        self._columns = None
        self._dtypes = None
        if _name is None:
            self._name = 'DT_{}'.format(DataFrame._df_count)
            self._quoted_name = quotename(self._name)
            DataFrame._df_count += 1
        else:
            self._name = _name
            self._quoted_name = quotename(_name)
        self._ttab_handling = 'unknown'
        self._ttab_reference = None

    @property
    def columns(self):
        """
        Lists the current DataFrame's column names. Computed lazily and cached.

        Each access to this property creates a new copy; mutating the list \
        will not alter or corrupt the DataFrame.
        """
        if self._columns is None:
            self._columns = self.__populate_columns()
        return self._columns[:]

    @property
    def name(self):
        """
        Returns the name of the DataFrame. Does not correspond to any HANA table name.

        This is useful for join predicates when the joining DataFrames have \
        columns with the same name.
        """
        return self._name

    @property
    def quoted_name(self):
        """
        The escaped name of the original Dataframe.

        Default-generated DataFrame names are safe to use in SQL without \
        escaping, but names set with DataFrame.alias may require escaping.
        """
        return self._quoted_name

    def __getitem__(self, index):
        if (isinstance(index, list) and
                all(isinstance(col, _STRING_TYPES) for col in index)):
            return self.select(*index)
        raise TypeError(
            '__getitem__ argument not understood: {!r}'.format(index))

    def __populate_columns(self):
        nodata_select = 'SELECT * FROM ({}) WHERE 0=1'.format(
            self.select_statement)
        with self.connection_context.connection.cursor() as cur:
            cur.execute(nodata_select)
            return [descr[0] for descr in cur.description]

    def __run_query(self, query):
        """
        Runs a query over the DataFrame's connection.

        Parameters
        ----------
        query : str
            SQL statement to run.

        Returns
        -------
        output : list of hdbcli.resultrow.ResultRow objects \
                 Query results.
        """
        with self.connection_context.connection.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    @staticmethod
    def _get_list_of_str(cols, message):
        """
        Generates a list of values after checking that all values in the list are \
        strings. If cols is a string, a list is created and returned.

        Parameters
        ----------
        cols : str or list of str
            The list of columns to check.
        message : str
            Error message in case the type of parameter cols is incorrect.

        Returns
        -------
        List
            cols if it is already a list otherwise a list containing cols
        """
        if isinstance(cols, _STRING_TYPES):
            cols = [cols]
        if (not cols or not isinstance(cols, list) or
                not all(isinstance(col, _STRING_TYPES) for col in cols)):
            raise TypeError(message)
        return cols

    def _validate_columns_in_dataframe(self, cols):
        """
        Checks if the specified columns are in the dataframe.
        Raises an error if any column in cols is not in the current dataframe.

        Parameters
        ----------
        cols : list
            The list of columns to check.
        """
        valid_set = set(self.columns)
        check_names = set(col for col in cols
                          if isinstance(col, _STRING_TYPES))
        if not valid_set.issuperset(check_names):
            invalid_names = [name for name in check_names
                             if name not in valid_set]
            message = "Column(s) not in DataFrame: {}".format(invalid_names)
            raise ValueError(message)

    def declare_lttab_usage(self, usage):
        """
        Declare whether this DataFrame makes use of local temporary tables.

        Some PAL execution routines can execute more efficiently if they \
        know up front whether a DataFrame's select statement requires \
        access to local temporary tables.

        Parameters
        ----------
        usage : bool, optional
            Whether this DataFrame uses local temporary tables.
        """
        if self._ttab_handling == 'safe':
            if usage:
                raise ValueError(
                    "declare_lttab_usage(True) called on a DataFrame " +
                    "believed to involve no local temporary tables.")
        elif self._ttab_handling in ('unsafe', 'ttab'):
            if not usage:
                raise ValueError(
                    "declare_lttab_usage(False) called on a DataFrame " +
                    "believed to involve local temporary tables.")
        else:
            self._ttab_handling = 'unsafe' if usage else 'safe'

    def _propagate_safety(self, other):
        # pylint:disable=protected-access
        if self._ttab_handling == 'safe':
            other._ttab_handling = 'safe'
        elif self._ttab_handling in ('ttab', 'unsafe'):
            other._ttab_handling = 'unsafe'

    def _df(self, select_statement, name=None, propagate_safety=False):
        # Because writing "DataFrame(self.connection_context" everywhere
        # is way too verbose.
        retval = DataFrame(self.connection_context, select_statement, name)
        if propagate_safety:
            self._propagate_safety(retval)
        return retval

    def alias(self, alias):
        """
        Returns a new DataFrame with an alias set.

        Parameters
        ----------
        alias : str
            Name of the DataFrame.

        Returns
        -------
        aliased_df : DataFrame
            DataFrame with an alias set.

        .. seealso ::
            DataFrame.rename_columns : For renaming individual columns.
        """
        retval = self._df(self.select_statement, alias)
        # pylint:disable=protected-access
        retval._ttab_handling = self._ttab_handling
        retval._ttab_reference = self._ttab_reference
        return retval

    def count(self):
        """
        Computes the number of rows in the DataFrame.

        Parameters
        ----------
        none

        Returns
        -------
        count : int
            Number of rows in the DataFrame.
        """
        new_select_statement = "SELECT COUNT(*) FROM ({})".format(self.select_statement)
        try:
            results = self.__run_query(new_select_statement)
        except dbapi.Error as exc:
            logger.error("Failed to get row count for the current Dataframe, %s", exc)
            raise
        return results[0][0]

    def empty(self):
        """
        Returns True if this DataFrame has 0 rows.

        Parameters
        ----------
        none

        Returns
        -------
        empty : bool
            True if the DataFrame is empty.

        Notes
        -----
        If a DataFrame contains only NULLs, it is not considered empty.

        Examples
        --------
        >>> df1.collect()
        Empty DataFrame
        Columns: [ACTUAL, PREDICT]
        Index: []
        >>> df1.empty()
        True
        >>> df2.collect()
          ACTUAL PREDICT
        0   None    None
        >>> df2.empty()
        False
        """
        return self.count() == 0

    def drop(self, cols):
        """
        Returns a new DataFrame without specified columns.

        Parameters
        ----------
        cols : list of str
            List of column names to drop.

        Returns
        -------
        filtered_df : DataFrame
            New DataFrame retaining only columns not in `cols`.

        Examples
        --------
        >>> df.collect()
           A  B
        0  1  3
        1  2  4
        >>> df.drop(['B']).collect()
           A
        0  1
        1  2
        """
        dropped_set = set(cols)
        if not dropped_set.issubset(self.columns):
            own_columns = set(self.columns)
            invalid_columns = [col for col in cols if col not in own_columns]
            raise ValueError("Can't drop nonexistent column(s): {}".format(invalid_columns))

        cols_kept = [quotename(col) for col in self.columns if col not in dropped_set]
        cols_kept = ', '.join(cols_kept)
        # TO DO: if cols_kept are the same as self.columns, then return nothing/self
        select_template = 'SELECT {} FROM ({}) AS {}'
        new_select_statement = select_template.format(
            cols_kept, self.select_statement, self.quoted_name)
        return self._df(new_select_statement, propagate_safety=True)

    def _generate_colname(self, prefix='GEN_COL'):
        # If the input prefix is safe to use unquoted, the output name
        # will be safe to use unquoted too.
        # Otherwise, you'll probably want to quotename the result before
        # using it in SQL.
        if not prefix:
            prefix = 'GEN_COL'
        if prefix not in self.columns:
            return prefix
        for i in range(1+len(self.columns)):
            gen_col = '{}_{}'.format(prefix, i)
            if gen_col not in self.columns:
                return gen_col
        # To get here, we would have to try more new names than this dataframe
        # has columns, and we would have to find that all of those names
        # were taken.
        raise AssertionError("This shouldn't be reachable.")

    def distinct(self, cols=None):
        """
        Returns a new DataFrame with distinct values for the specified columns.
        If no columns are specified, the distinct row values from all columns are returned.

        Parameters
        ----------
        cols : str or list of str, optional
            Column or list of columns to consider when getting distinct
            values. Defaults to use all columns.

        Returns
        -------
        distinct_df : DataFrame
            DataFrame with distinct values for cols.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B    C
        0  1  A  100
        1  1  A  101
        2  1  A  102
        3  1  B  100
        4  1  B  101
        5  1  B  102
        6  1  B  103
        7  2  A  100
        8  2  A  100

        Distinct values in a column:

        >>> df.distinct("B").collect()
           B
        0  A
        1  B

        Distinct values of a subset of columns:

        >>> df.distinct(["A", "B"]).collect()
           A  B
        0  1  B
        1  2  A
        2  1  A

        Distinct values of the entire data set:

        >>> df.distinct().collect()
           A  B    C
        0  1  A  102
        1  1  B  103
        2  1  A  101
        3  2  A  100
        4  1  B  101
        5  1  A  100
        6  1  B  100
        7  1  B  102
        """
        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns
        select_statement = "SELECT DISTINCT {} FROM ({}) AS {}".format(
            ', '.join([quotename(col) for col in cols]),
            self.select_statement, self.quoted_name)
        return self._df(select_statement, propagate_safety=True)

    def drop_duplicates(self, subset=None):
        """
        Returns a new DataFrame with duplicate rows removed. All columns in the
        DataFrame are returned. There is no way to keep specific duplicate rows.

        .. warning::
           Specifying a non-None value of `subset` may produce an "unstable" \
           DataFrame, the contents of which may be different every time you \
           look at it. Specifically, if two rows are duplicates in their \
           `subset` columns and have different values in other columns, \
           a different row could be picked every time you look at the result.

        Parameters
        ----------
        subset : list of str, optional
            List of columns to consider when deciding whether rows are \
            duplicates of each other. Defaults to use all columns.

        Returns
        -------
        deduplicated_df : DataFrame
            DataFrame with only one copy of duplicate rows.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B    C
        0  1  A  100
        1  1  A  101
        2  1  A  102
        3  1  B  100
        4  1  B  101
        5  1  B  102
        6  1  B  103
        7  2  A  100
        8  2  A  100

        Drop duplicates based on the values of a subset of columns:

        >>> df.drop_duplicates(["A", "B"]).collect()
           A  B    C
        0  1  A  100
        1  1  B  100
        2  2  A  100

        Distinct values on the entire data set:

        >>> df.drop_duplicates().collect()
           A  B    C
        0  1  A  102
        1  1  B  103
        2  1  A  101
        3  2  A  100
        4  1  B  101
        5  1  A  100
        6  1  B  100
        7  1  B  102
        """

        if subset is None:
            return self._df("SELECT DISTINCT * FROM ({}) AS {}".format(
                self.select_statement, self.quoted_name), propagate_safety=True)

        if not subset:
            raise ValueError("drop_duplicates requires at least one column in subset")

        keep_columns = ', '.join([quotename(col) for col in self.columns])
        partition_by = ', '.join([quotename(col) for col in subset])
        seqnum_col = quotename(self._generate_colname('SEQNUM'))
        seqnum_template = "SELECT *, ROW_NUMBER() OVER (PARTITION BY {}) AS {} FROM ({})"
        select_with_seqnum = seqnum_template.format(
            partition_by, seqnum_col, self.select_statement)
        new_select_statement = "SELECT {} FROM ({}) WHERE {} = 1".format(
            keep_columns, select_with_seqnum, seqnum_col)
        return self._df(new_select_statement, propagate_safety=True)

    def dropna(self, how=None, thresh=None, subset=None):
        # need to test
        """
        Returns a new DataFrame with NULLs removed.

        Parameters
        ----------
        how : {'any', 'all'}, optional
            If provided, 'any' eliminates rows with any NULLs, \
            and 'all' eliminates rows that are entirely NULLs. \
            If neither `how` nor `thresh` are provided, `how` \
            defaults to 'any'.
        thresh : int, optional
            If provided, rows with less than `thresh` non-NULL values \
            are dropped. \
            `how` and `thresh` cannot both be provided.
        subset : list of str, optional
            Columns to consider when looking for NULLs. Values in
            other columns will be ignored, NULL or not.
            Defaults to all columns.

        Returns
        -------
        filtered_df : DataFrame
            New DataFrame with select statement that removes NULLs.

        Examples
        --------
        Dropping rows with any nulls:

        >>> df.collect()
             A    B    C
        0  1.0  3.0  5.0
        1  2.0  4.0  NaN
        2  3.0  NaN  NaN
        3  NaN  NaN  NaN
        >>> df.dropna().collect()
             A    B    C
        0  1.0  3.0  5.0

        Dropping rows that are entirely nulls:

        >>> df.dropna(how='all').collect()
             A    B    C
        0  1.0  3.0  5.0
        1  2.0  4.0  NaN
        2  3.0  NaN  NaN

        Dropping rows with less than 2 non-null values:

        >>> df.dropna(thresh=2).collect()
             A    B    C
        0  1.0  3.0  5.0
        1  2.0  4.0  NaN
        """

        if how is not None and thresh is not None:
            raise ValueError("Cannot provide both how and thresh.")

        if subset is not None:
            cols = subset
        else:
            cols = self.columns

        cols = [quotename(col) for col in cols]

        if thresh is None:
            if how in {'any', None}:
                and_or = ' OR '
            elif how == 'all':
                and_or = ' AND '
            else:
                raise ValueError("Invalid value of how: {}".format(how))
            drop_if = and_or.join('{} IS NULL'.format(col) for col in cols)
            keep_if = 'NOT ({})'.format(drop_if)

            retval = self.filter(keep_if)
            self._propagate_safety(retval)
            return retval

        count_expression = '+'.join(
            ['(CASE WHEN {} IS NULL THEN 0 ELSE 1 END)'.format(col) for col in cols])
        count_colname = self._generate_colname('CT')
        select_with_count = 'SELECT *, ({}) AS {} FROM ({}) {}'.format(
            count_expression, count_colname, self.select_statement, self.quoted_name)
        projection = ', '.join([quotename(col) for col in self.columns])
        new_select_statement = 'SELECT {} FROM ({}) WHERE {} >= {}'.format(
            projection, select_with_count, count_colname, thresh)
        return self._df(new_select_statement, propagate_safety=True)

    def dtypes(self, subset=None):
        """
        Returns a sequence of tuples describing the DataFrame's SQL types.

        The tuples list the name, SQL type name, and display size
        corresponding to the DataFrame's columns.

        Parameters
        ----------
        subset : list of str, optional
            The columns that the information will be generated from.
            Defaults to all columns.

        Returns
        -------
        dtypes : list of tuples
            Each tuple consists of the name, SQL type name, and display size
            for one of the DataFrame's columns. The list will be in the order
            specified by the `subset`, or in the DataFrame's original column
            order if a `subset` is not provided.
        """
        if self._dtypes is None:
            with self.connection_context.connection.cursor() as cur:
                cur.execute(self.select_statement)
                self._dtypes = [(c[0], TYPE_CODES[c[1]], c[2]) for c in cur.description]
        if subset is None:
            return self._dtypes[:]
        dtype_map = {descr[0]: descr for descr in self._dtypes}
        return [dtype_map[col] for col in subset]

    _ARBITRARY_PSEUDOTOKEN_LIMIT = 200

    def _token_validate(self, sql):
        """
        Calls IS_SQL_INJECTION_SAFE on input. Does not guarantee injection-safety.

        Parameters
        ----------
        sql : str
            SQL text.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If IS_SQL_INJECTION_SAFE returns 0.

        Notes
        -----
        This does not guarantee injection-safety. Any parts of this library
        that take SQL statements, expressions, or predicates are not safe
        against SQL injection. This only catches some instances of comments
        and malformed tokens in the input.

        IS_SQL_INJECTION_SAFE may produce false positives or false negatives.
        """
        with self.connection_context.connection.cursor() as cur:
            cur.execute('SELECT IS_SQL_INJECTION_SAFE(?, ?) FROM DUMMY',
                        (sql, self._ARBITRARY_PSEUDOTOKEN_LIMIT))
            val = cur.fetchone()[0]
        if not val:
            msg = 'SQL token validation failed for string {!r}'.format(sql)
            raise BadSQLError(msg)

    def filter(self, condition):
        """
        Selects rows matching the given condition.

        Very little checking is done on the condition string.
        Use only with trusted inputs.

        Parameters
        ----------
        condition : str
            Filter condition. Format as SQL <condition>.

        Returns
        -------
        filtered_df : DataFrame
            DataFrame with rows matching the given condition.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If comments or malformed tokens are detected in `condition`.
            May have false positives and false negatives.

        Examples
        --------
        >>> df.collect()
           A  B
        0  1  3
        1  2  4
        >>> df.filter('B < 4').collect()
           A  B
        0  1  3
        """
        self._token_validate(condition)
        return self._df('SELECT * FROM ({}) AS {} WHERE {}'.format(
            self.select_statement, self.quoted_name, condition))

    def has(self, col):
        """
        Returns True if a column is in the DataFrame.

        Parameters
        ----------
        col : str
            Name of column to search in the projection list of this DataFrame.

        Returns
        ------
        has : boolean
            Returns True if the column exists in the DataFrame's projection list.

        Examples
        --------
        >>> df.columns
        ['A', 'B']
        >>> df.has('A')
        True
        >>> df.has('C')
        False
        """
        # df.has(col) doesn't really give much benefit over using
        # col in df.columns directly. It may not be worth
        # having this method at all.
        return col in self.columns

    def head(self, n=1): #pylint: disable=invalid-name
        """
        Returns a DataFrame of the first `n` rows in the current DataFrame.

        Parameters
        ----------
        n : int, optional
            Number of rows returned. Default value: 1.

        Returns
        -------
        sub_df : DataFrame
            New DataFrame of the first `n` rows of the current DataFrame.
        """
        head_select = 'SELECT TOP {} * FROM ({}) dt'.format(
            n, self.select_statement)
        return self._df(head_select, propagate_safety=True)

    def hasna(self, cols=None):
        """
        Returns True if a DataFrame contains NULLs.

        Parameters
        ----------
        cols : str or list of str, optional
            Column or list of columns to be checked for NULL values.
            Defaults to all columns.

        Returns
        -------
        hasna : bool
            True if this DataFrame contains NULLs.

        Examples
        --------
        >>> df1.collect()
          ACTUAL PREDICT
        0   1.0    None
        >>> df1.hasna()
        True
        """
        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns

        count_cols = []
        for col in cols:
            quoted_col = quotename(col)
            count_cols.append("count({})".format(quoted_col))
        minus_expression = ' + '.join(count_cols)

        count_statement = "SELECT COUNT(*)*{} - ({}) FROM ({})".format(
            len(cols),
            minus_expression,
            self.select_statement)
        try:
            count = self.__run_query(count_statement)
        except dbapi.Error as exc:
            logger.error("Failed to get NULL value count for the current Dataframe, %s", exc)
            raise
        return count[0][0] > 0

    def fillna(self, value, subset=None):
        # need to test
        # need to talk to nanda about error handling
        """
        Returns a DataFrame with NULLs replaced with a specified value.

        Currently only supports filling numeric columns.

        Parameters
        ----------
        value : int or float
            The value that will replace NULL. `value` should have a type
            appropriate for the selected columns.
        subset : list of str, optional
            List of columns that will have their NULL values replaced.
            Defaults to all columns.

        Returns
        -------
        filled_df : DataFrame
            New DataFrame with the NULL values replaced.
        """
        if subset is not None:
            filled_set = set(subset)
        else:
            filled_set = set(self.columns)
        if not isinstance(value, (float,) + _INTEGER_TYPES):
            # We currently can't escape other types correctly,
            # and our sql-statement-based design doesn't support
            # query parameters.
            raise TypeError("Fill values currently must be ints or floats.")
        # todo ensure type matches/talk to nanda on error handling
        #untouched columns
        select_values = []
        for col in self.columns:
            quoted_col = quotename(col)
            if col in filled_set:
                #pylint: disable=W0511
                # TODO: This is not a safe way to insert the value into the SQL.
                # It won't quote strings properly, it won't handle date/time
                # types, and it will sometimes (Python 2 only?) lose precision
                # on numeric values.
                select_values.append("COALESCE({0}, {1}) AS {0}".format(quoted_col, value))
            else:
                select_values.append(quoted_col)
        cols = ', '.join(select_values)
        new_select_statement = 'SELECT {} FROM ({}) dt'.format(cols, self.select_statement)
        return self._df(new_select_statement, propagate_safety=True)

    def join(self, other, condition, how='inner', select=None):
        """
        Returns a new DataFrame that is a join of the current DataFrame with
        another specified DataFrame.

        Parameters
        ----------
        other : DataFrame
            The DataFrame to join with.
        condition : str
            Join predicate.
        how : {'inner', 'left', 'right', 'outer'}, optional
            Type of join. Defaults to 'inner'.
        select : list, optional
            If provided, each element specifies a column in the result.
            A string in the `select` list should be the name of a column in
            one of the input dataframes. An (expression, name) tuple creates
            a new column with the given name, computed from the given
            expression.
            If not provided, defaults to selecting all columns from both
            dataframes, with the left dataframe's columns first.

        Returns
        -------
        joined_df : DataFrame
            A new DataFrame object made from the join of the current DataFrame
            with another DataFrame.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If comments or malformed tokens are detected in `condition`
            or in a column expression.
            May have false positives and false negatives.

        Examples
        --------
        The expression selection functionality can be used to disambiguate duplicate
        column names in a join:

        >>> df1.collect()
           A  B    C
        0  1  2  3.5
        1  2  4  5.6
        2  3  3  1.1
        >>> df2.collect()
           A  B     D
        0  2  1  14.0
        1  3  4   5.6
        2  4  3   0.0
        >>> df1.alias('L').join(df2.alias('R'), 'L.A = R.A', select=[
        ...     ('L.A', 'A'),
        ...     ('L.B', 'B_LEFT'),
        ...     ('R.B', 'B_RIGHT'),
        ...     'C',
        ...     'D']).collect()
           A  B_LEFT  B_RIGHT    C     D
        0  2       4        1  5.6  14.0
        1  3       3        4  1.1   5.6
        """
        #if left and right joins are improper (ex: int on string) then there will be a sql error
        if how not in ['inner', 'outer', 'left', 'right']:
            raise ValueError('Invalid value for "how" argument: {!r}'.format(how))
        self._token_validate(condition)
        on_clause = 'ON ' + condition
        join_type_map = {
            'inner': 'INNER',
            'left': 'LEFT OUTER',
            'right': 'RIGHT OUTER',
            'outer': 'FULL OUTER'
        }
        join_type = join_type_map[how]

        if select is None:
            projection_string = '*'
        else:
            projection_string = self._stringify_select_list(select)

        select_template = 'SELECT {} FROM ({}) AS {} {} JOIN ({}) AS {} {}'
        new_select_statement = select_template.format(
            projection_string,
            self.select_statement, self.quoted_name, join_type,
            other.select_statement, other.quoted_name, on_clause)
        return self._df(new_select_statement)

#    def sample(self, replace=False, fraction=0.3, seed=None):
#        # need to test
#        # not sure how to incorporate replace
#        # seed probably not possible since no seed in HANA SQL function rand()
#        """
#        Return a sampled subset of the DataFrame.
#
#        .. warning::
#           The current implementation of this method produces an "unstable"
#           DataFrame that may have different contents every time you look
#           at it.
#
#           As such, this method should probably not be used yet.
#
#
#        Parameters
#        ----------
#        replace : boolean, optional
#            Whether to sample with or without replacement.
#            Samples with replacement are currently unsupported.
#            Defaults to False, for a sample without replacement.
#        fraction : float, optional
#            Proportion of rows to take. Defaults to 0.3
#        seed : None, do not provide
#            Currently unused. May be used to seed the RNG for a
#            repeatable sample in future versions.
#
#        Returns
#        -------
#        sampled_df : DataFrame
#            New DataFrame containing random subset of the original's rows.
#        """
#        num_rows_sample = int(fraction * self.count())
#        select_template = 'SELECT * FROM ({}) dt ORDER BY RAND() LIMIT {}'
#        new_select_statement = select_template.format(
#            self.select_statement, num_rows_sample)
#        return DataFrame(self.connection_context, new_select_statement)

    def save(self, where, table_type=None, force=False):
        """
        Creates a table or view holding the current DataFrame's data.

        Parameters
        ----------
        where : str or (str, str) tuple
            Table name or (schema name, table name) tuple. If no schema
            is provided, the table or view will be created in the current
            schema.
        table_type : str, optional
            Type of table to create. Case-insensitive.

            Permanent table options:
              - "ROW"
              - "COLUMN"
              - "HISTORY COLUMN"

            Temporary table options:
              - "GLOBAL TEMPORARY"
              - "GLOBAL TEMPORARY COLUMN"
              - "LOCAL TEMPORARY"
              - "LOCAL TEMPORARY COLUMN"

            Not a table:
              - "VIEW"

            Defaults to 'LOCAL TEMPORARY COLUMN' if `where` starts
            with '#'. Otherwise, the default is 'COLUMN'.
        force : bool
            If force is True, it will replace the existing table.

        Returns
        -------
        DataFrame
            DataFrame representing the new table or view.

        Notes
        -----
        For this operation to succeed, the table name must not be in
        use, the schema must exist, and the user must have permission
        to create tables (or views) in the target schema.
        """
        if isinstance(where, tuple):
            schema, table = where
        else:
            schema, table = None, where
        if table_type is None:
            if table.startswith('#'):
                table_type = 'LOCAL TEMPORARY COLUMN'
            else:
                table_type = 'COLUMN'
        if table_type.upper() not in {
                'ROW',
                'COLUMN',
                'HISTORY COLUMN',
                'GLOBAL TEMPORARY',
                'GLOBAL TEMPORARY COLUMN',
                'LOCAL TEMPORARY',
                'LOCAL TEMPORARY COLUMN',
                'VIEW'}:
            raise ValueError("{!r} is not a valid value of table_type".format(
                table_type))

        if schema is None:
            where_string = quotename(table)
        else:
            where_string = '{}.{}'.format(*map(quotename, where))

        table_type = table_type.upper()
        if table_type != 'VIEW':
            table_type += ' TABLE'

        with self.connection_context.connection.cursor() as cur:
            if force is True:
                try:
                    cur.execute("DROP TABLE {};").format(where_string)
                except dbapi.Error:
                    pass
            cur.execute("CREATE {} {} AS ({})".format(
                table_type, where_string, self.select_statement))
        return self.connection_context.table(table, schema=schema)

    def sort(self, cols, desc=False):
        """
        Returns a new DataFrame sorted by the specified columns.

        Parameters
        ----------
        cols : str or list of str
            Column or list of columns to sort by.
            If a list is specified, the sort order in parameter desc is used
            for all columns.
        desc : boolean, optional
            Set to True to sort in descending order. Defaults to False,
            for ascending order.

        Returns
        -------
        sorted_df : DataFrame
            New DataFrame object with rows sorted as specified.
        """
        # Issue: DataFrames constructed from a sorted DataFrame may not
        # respect its order, since there's no guarantee that subqueries
        # and derived tables will preserve order.
        #
        # collect() is the only thing we can really guarantee will work.
        #
        # We'd have to change our model to propagate ordering constraints
        # explicitly, outside of the select_statement, to guarantee
        # ordering.
        if not cols:
            raise ValueError("Can't sort by 0 columns")
        cols = DataFrame._get_list_of_str(cols,
                                          'Parameter cols must be a string or a list of strings')
        self._validate_columns_in_dataframe(cols)

        cols = [quotename(c) for c in cols]
        template = '{} DESC' if desc else '{} ASC'
        order_by = 'ORDER BY ' + ', '.join(template.format(col) for col in cols)
        new_select_statement = 'SELECT * FROM ({}) AS {} {}'.format(
            self.select_statement, self.quoted_name, order_by)
        return self._df(new_select_statement, propagate_safety=True)

    def _stringify_select_list(self, select):
        projection = []
        for col in select:
            if isinstance(col, _STRING_TYPES):
                if '*' in col:
                    projection.append(col)
                else:
                    projection.append(quotename(col))
            else:
                expr, name = col
                self._token_validate(expr)
                projection.append('{} AS {}'.format(expr, quotename(name)))
        return ', '.join(projection)

    def select(self, *cols):
        """
        Returns a new DataFrame with columns derived from the current DataFrame.

        .. warning::
            There is no check that inputs interpreted as SQL expressions are
            actually valid expressions; an "expression" like
            "A FROM TAB; DROP TABLE IMPORTANT_THINGS; SELECT A" can cause
            a lot of damage.

        Parameters
        ----------
        cols : str or (str, str) tuple.
            Columns of the new DataFrame. A string is treated as the name
            of a column to select; a (str, str) tuple is treated as
            (SQL expression, alias). As a special case, '*' is expanded
            to all columns of the original DataFrame.

        Returns
        -------
        projected_df : DataFrame
            New DataFrame object with the specified columns.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If comments or malformed tokens are detected in a column
            expression. May have false positives and false negatives.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B  C
        0  1  2  3

        Selecting a subset of existing columns:

        >>> df.select('A', 'B').collect()
           A  B
        0  1  2

        Computing a new column based on existing columns:

        >>> df.select('*', ('B * 4', 'D')).collect()
           A  B  C  D
        0  1  2  3  8
        """
        columns = []
        for col in cols:
            if isinstance(col, list):
                # Compatibility with old df.select(['a', 'b']) style.
                columns.extend(col)
            elif col == '*':
                columns.extend(self.columns)
            else:
                columns.append(col)

        self._validate_columns_in_dataframe(columns)

        projection_string = self._stringify_select_list(columns)
        new_select_statement = 'SELECT {} FROM ({}) AS {}'.format(
            projection_string, self.select_statement, self.quoted_name)

        newdf = self._df(new_select_statement)
        newdf._columns = [col if isinstance(col, _STRING_TYPES) else col[1]  #pylint: disable=protected-access
                          for col in columns]
        if all(isinstance(col, _STRING_TYPES) for col in columns):
            self._propagate_safety(newdf)
        return newdf

    def union(self,
              other,
              all=True, # pylint:disable=redefined-builtin
              # by='position',
             ):
        """
        Combine this DataFrame's rows and another DataFrame's rows into
        one DataFrame. Equivalent to a SQL UNION ALL by default.

        Parameters
        ----------
        other : DataFrame
            Right side of the union.
        all : bool, optional
            If True, keep duplicate rows, equivalent to UNION ALL in SQL.
            If False, keep only one copy of duplicate rows (even if they
            come from the same side of the union), equivalent to a UNION
            or a UNION ALL followed by DISTINCT in SQL.
            Defaults to True.

        Returns
        -------
        DataFrame
            Combined data from `self` and `other`.

        Examples
        --------
        We have two DataFrames we want to union, with some duplicate rows:

        >>> df1.collect()
           A  B
        0  1  2
        1  1  2
        2  2  3
        >>> df2.collect()
           A  B
        0  2  3
        1  3  4

        union() will produce a DataFrame containing all rows of both df1
        and df2, like a UNION ALL:

        >>> df1.union(df2).collect()
           A  B
        0  1  2
        1  1  2
        2  2  3
        3  2  3
        4  3  4

        To get the deduplication effect of a UNION DISTINCT, pass
        all=False or call distinct() after union():

        >>> df1.union(df2, all=False).collect()
           A  B
        0  1  2
        1  2  3
        2  3  4
        >>> df1.union(df2).distinct().collect()
           A  B
        0  1  2
        1  2  3
        2  3  4
        """
#        if by == 'name':
#            if sorted(self.columns) != sorted(other.columns):
#                msg = "Union by name needs matching column names."
#                raise ValueError(msg)
#            other = other[self.columns]
#        elif by != 'position':
#            raise ValueError("Unrecognized value of 'by': {!r}".format(by))
        new_select = '(({}) {} ({}))'.format(
            self.select_statement,
            'UNION ALL' if all else 'UNION',
            other.select_statement)
        retval = self._df(new_select)
        # pylint:disable=protected-access
        if self._ttab_handling == other._ttab_handling == 'safe':
            retval._ttab_handling = 'safe'
        elif {self._ttab_handling, other._ttab_handling} & {'ttab', 'unsafe'}:
            retval._ttab_handling = 'unsafe'
        return retval

    def collect(self):
        """
        Copies the current DataFrame to a new Pandas DataFrame.

        Parameters
        ----------
        none

        Returns
        -------
        pandas_df : pandas.DataFrame
            Pandas DataFrame containing the current DataFrame's data.

        Examples
        --------
        Viewing a hana_ml DataFrame doesn't reveal much on its own,
        because it doesn't execute the underlying SQL or fetch data:

        >>> df = cc.table('T')
        >>> df
        <hana_ml.dataframe.DataFrame object at 0x7f2b7f24ddd8>

        Using collect() will execute the SQL and fetch the results
        into a Pandas DataFrame:

        >>> df.collect()
           A  B
        0  1  3
        1  2  4
        >>> type(df.collect())
        <class 'pandas.core.frame.DataFrame'>
        """
        #pylint: disable=W0511
        # TODO: This produces wrong dtypes for an empty DataFrame.
        # We can't rely on type inference for an empty DataFrame;
        # we will need to provide dtype information explicitly.
        # The pandas.DataFrame constructor's dtype argument only allows
        # a single dtype for the whole DataFrame, but we should be able
        # to use the astype method instead.
        try:
            results = self.__run_query(self.select_statement)
        except dbapi.Error as exc:
            logger.error("Failed to retrieve data for the current dataframe, %s", exc)
            raise
        return pd.DataFrame(results, columns=self.columns)

    def rename_columns(self, names):
        """
        Returns a DataFrame with renamed columns.

        Parameters
        ----------
        names : list or dict
            If a list, specifies new names for every column in this dataframe.
            If a dict, each dict entry maps an existing name to a new name,
            and not all columns need to be renamed.

        Returns
        -------
        DataFrame
            The same data as the original DataFrame, with new column names.

        See Also
        --------
        DataFrame.alias : For renaming the DataFrame itself.

        Examples
        --------
        >>> df.collect()
           A  B
        0  1  3
        1  2  4
        >>> df.rename_columns(['C', 'D']).collect()
           C  D
        0  1  3
        1  2  4
        >>> df.rename_columns({'B': 'D'}).collect()
           A  D
        0  1  3
        1  2  4
        """
        if isinstance(names, list):
            if len(names) != len(self.columns):
                if len(names) > len(self.columns):
                    problem = "Too many"
                else:
                    problem = "Not enough"
                raise ValueError(problem + ' columns in rename_columns list.')
            names = dict(zip(self.columns, names))
        elif isinstance(names, dict):
            bad_names = set(names).difference(self.columns)
            if bad_names:
                raise ValueError("Column(s) not in DataFrame: {}".format(
                    sorted(bad_names)))
        else:
            raise TypeError("names should be a list or dict, not {}".format(
                type(names)))
        retval = self.select(*[(quotename(orig), names[orig]) if orig in names
                               else orig
                               for orig in self.columns])
        self._propagate_safety(retval)
        return retval

    def cast(self, cols, new_type):
        """
        Returns a DataFrame with columns cast to a new type.

        The name of the column in the returned DataFrame is the same as the original column.
         .. warning::
           Type compatibility between the existing column type and the new type is not checked.
           An incompatibility will result in an error.

        Parameters
        ----------
        cols : str or list
            The column(s) to be cast to a different type.
        new_type : str
            The database datatype to cast the column(s) to.
            No checks are performed to see if the new type is valid.
            An invalid type can lead to SQL errors or even SQL injection vulnerabilities.

        Returns
        -------
        DataFrame
            The same data as this DataFrame but with columns cast to the specified type.

        Examples
        --------
        Input:

        >>> df1 = cc.sql('SELECT "AGE", "PDAYS", "HOUSING" FROM DBM_TRAINING_TBL')
        >>> df1.dtypes()
        [('AGE', 'INT', 10), ('PDAYS', 'INT', 10), ('HOUSING', 'VARCHAR', 100)]

        Casting a column to NVARCHAR(20):

        >>> df2 = df1.cast('AGE', 'NVARCHAR(20)')
        >>> df2.dtypes()
        [('AGE', 'NVARCHAR', 20), ('PDAYS', 'INT', 10), ('HOUSING', 'VARCHAR', 100)]

        Casting a list of columns to NVARCHAR(50):

        >>> df3 = df1.cast(['AGE', 'PDAYS'], 'NVARCHAR(50)')
        >>> df3.dtypes()
        [('AGE', 'NVARCHAR', 50), ('PDAYS', 'NVARCHAR', 50), ('HOUSING', 'VARCHAR', 100)]
        """
        cols = DataFrame._get_list_of_str(cols,
                                          'Parameter cols must be a string or a list of strings')
        self._validate_columns_in_dataframe(cols)

        # Need to check for valid types in newtype, skip for now
        projection = [quotename(col) if not col in cols
                      else 'CAST({0} AS {1}) AS {0}'.format(quotename(col), new_type)
                      for col in self.columns]

        cast_select = 'SELECT {} FROM ({}) AS {}'.format(', '.join(projection),
                                                         self.select_statement,
                                                         self.quoted_name)
        return self._df(cast_select, propagate_safety=True)

    def to_head(self, col):
        """
        Returns a DataFrame with specified column as the first item in projection.

        Parameters
        ----------
        col : str
            The column to move to head.

        Returns
        -------
        DataFrame
            The same data as this DataFrame but with the named column in the first position.

        Examples
        --------
        Input:

        >>> df1 = cc.table("DBM_TRAINING")
        >>> import pprint
        >>> pprint.pprint(df1.columns)
        ['ID',
         'AGE',
         'JOB',
         'MARITAL',
         'EDUCATION',
         'DBM_DEFAULT',
         'HOUSING',
         'LOAN',
         'CONTACT',
         'DBM_MONTH',
         'DAY_OF_WEEK',
         'DURATION',
         'CAMPAIGN',
         'PDAYS',
         'PREVIOUS',
         'POUTCOME',
         'EMP_VAR_RATE',
         'CONS_PRICE_IDX',
         'CONS_CONF_IDX',
         'EURIBOR3M',
         'NREMPLOYED',
         'LABEL']

        Moving the column 'LABEL' to head:

        >>> df2 = df1.to_head('LABEL')
        >>> pprint.pprint(df2.columns)
        ['LABEL',
         'ID',
         'AGE',
         'JOB',
         'MARITAL',
         'EDUCATION',
         'DBM_DEFAULT',
         'HOUSING',
         'LOAN',
         'CONTACT',
         'DBM_MONTH',
         'DAY_OF_WEEK',
         'DURATION',
         'CAMPAIGN',
         'PDAYS',
         'PREVIOUS',
         'POUTCOME',
         'EMP_VAR_RATE',
         'CONS_PRICE_IDX',
         'CONS_CONF_IDX',
         'EURIBOR3M',
         'NREMPLOYED']
        """
        if not isinstance(col, _STRING_TYPES):
            raise TypeError('Parameter col must be a string')

        if col not in self.columns:
            raise ValueError('Column {} is not a column in this DataFrame'.format(quotename(col)))
        cols = self.columns
        cols.insert(0, cols.pop(cols.index(col)))
        return self[cols]

    def describe(self, cols=None):
        # The disable of line lengths is for the example in docstring.
        # The example is copy pasted after a run and may result in the output not quite lining up
        """
        Returns a DataFrame containing various statistics for the requested column(s).

        Parameters
        ----------
        cols : str or list
            The column(s) to be described. Defaults to all columns.

        Returns
        -------
        DataFrame
            A DataFrame containing statistics for the specified column(s)
            in the current DataFrame.

            The statistics included are the count of rows ("count"),
            number of distinct values ("unique"),
            number of nulls ("nulls"), average ("mean"), standard deviation("std")
            median ("median"), minimum value ("min"), maximum value ("max"),
            25% percentile when treated as continuous variable ("25_percent_cont"),
            25% percentile when treated as discrete variable ("25_percent_disc"),
            50% percentile when treated as continuous variable ("50_percent_cont"),
            50% percentile when treated as discrete variable ("50_percent_disc"),
            75% percentile when treated as continuous variable ("75_percent_cont"),
            75% percentile when treated as discrete variable ("75_percent_disc").

            For columns that are strings, statistics such as average ("mean"),
            standard deviation ("std"), median ("median"), and the various percentiles
            are NULLs.

            If the list of columns contain both string and numeric data types,
            minimum and maximum values become NULLs.

        Examples
        --------
        Input:

        >>> df1 = cc.table("DBM_TRAINING")
        >>> import pprint
        >>> pprint.pprint(df2.columns)
        ['LABEL',
         'ID',
         'AGE',
         'JOB',
         'MARITAL',
         'EDUCATION',
         'DBM_DEFAULT',
         'HOUSING',
         'LOAN',
         'CONTACT',
         'DBM_MONTH',
         'DAY_OF_WEEK',
         'DURATION',
         'CAMPAIGN',
         'PDAYS',
         'PREVIOUS',
         'POUTCOME',
         'EMP_VAR_RATE',
         'CONS_PRICE_IDX',
         'CONS_CONF_IDX',
         'EURIBOR3M',
         'NREMPLOYED']

        Describe a few numeric columns and collect to return a Pandas DataFrame:

        >>> df1.describe(['AGE', 'PDAYS']).collect()
          column  count  unique  nulls        mean         std  min  max  median
        0    AGE  16895      78      0   40.051376   10.716907   17   98      38
        1  PDAYS  16895      24      0  944.406688  226.331944    0  999     999
           25_percent_cont  25_percent_disc  50_percent_cont  50_percent_disc
        0             32.0               32             38.0               38
        1            999.0              999            999.0              999
           75_percent_cont  75_percent_disc
        0             47.0               47
        1            999.0              999

        Describe a few non-numeric columns and collect to return a Pandas DataFrame:

        >>> df1.describe(['JOB', 'MARITAL']).collect()
            column  count  unique  nulls  mean   std       min      max median
        0      JOB  16895      12      0  None  None    admin.  unknown   None
        1  MARITAL  16895       4      0  None  None  divorced  unknown   None
          25_percent_cont 25_percent_disc 50_percent_cont 50_percent_disc
        0            None            None            None            None
        1            None            None            None            None
          75_percent_cont 75_percent_disc
        0            None            None
        1            None            None

        Describe all columns in a DataFrame:

        >>> df1.describe().collect()
                    column  count  unique  nulls          mean           std
        0               ID  16895   16895      0  21282.286652  12209.759725
        1              AGE  16895      78      0     40.051376     10.716907
        2         DURATION  16895    1267      0    263.965670    264.331384
        3         CAMPAIGN  16895      35      0      2.344658      2.428449
        4            PDAYS  16895      24      0    944.406688    226.331944
        5         PREVIOUS  16895       7      0      0.209529      0.539450
        6     EMP_VAR_RATE  16895      10      0     -0.038798      1.621945
        7   CONS_PRICE_IDX  16895      26      0     93.538844      0.579189
        8    CONS_CONF_IDX  16895      26      0    -40.334123      4.865720
        9        EURIBOR3M  16895     283      0      3.499297      1.777986
        10      NREMPLOYED  16895      11      0   5160.371885     75.320580
        11             JOB  16895      12      0           NaN           NaN
        12         MARITAL  16895       4      0           NaN           NaN
        13       EDUCATION  16895       8      0           NaN           NaN
        14     DBM_DEFAULT  16895       2      0           NaN           NaN
        15         HOUSING  16895       3      0           NaN           NaN
        16            LOAN  16895       3      0           NaN           NaN
        17         CONTACT  16895       2      0           NaN           NaN
        18       DBM_MONTH  16895      10      0           NaN           NaN
        19     DAY_OF_WEEK  16895       5      0           NaN           NaN
        20        POUTCOME  16895       3      0           NaN           NaN
        21           LABEL  16895       2      0           NaN           NaN
                 min        max     median  25_percent_cont  25_percent_disc
        0      5.000  41187.000  21786.000        10583.500        10583.000
        1     17.000     98.000     38.000           32.000           32.000
        2      0.000   4918.000    184.000          107.000          107.000
        3      1.000     43.000      2.000            1.000            1.000
        4      0.000    999.000    999.000          999.000          999.000
        5      0.000      6.000      0.000            0.000            0.000
        6     -3.400      1.400      1.100           -1.800           -1.800
        7     92.201     94.767     93.444           93.075           93.075
        8    -50.800    -26.900    -41.800          -42.700          -42.700
        9      0.634      5.045      4.856            1.313            1.313
        10  4963.000   5228.000   5191.000         5099.000         5099.000
        11       NaN        NaN        NaN              NaN              NaN
        12       NaN        NaN        NaN              NaN              NaN
        13       NaN        NaN        NaN              NaN              NaN
        14       NaN        NaN        NaN              NaN              NaN
        15       NaN        NaN        NaN              NaN              NaN
        16       NaN        NaN        NaN              NaN              NaN
        17       NaN        NaN        NaN              NaN              NaN
        18       NaN        NaN        NaN              NaN              NaN
        19       NaN        NaN        NaN              NaN              NaN
        20       NaN        NaN        NaN              NaN              NaN
        21       NaN        NaN        NaN              NaN              NaN
            50_percent_cont  50_percent_disc  75_percent_cont  75_percent_disc
        0         21786.000        21786.000        32067.500        32068.000
        1            38.000           38.000           47.000           47.000
        2           184.000          184.000          324.000          324.000
        3             2.000            2.000            3.000            3.000
        4           999.000          999.000          999.000          999.000
        5             0.000            0.000            0.000            0.000
        6             1.100            1.100            1.400            1.400
        7            93.444           93.444           93.994           93.994
        8           -41.800          -41.800          -36.400          -36.400
        9             4.856            4.856            4.961            4.961
        10         5191.000         5191.000         5228.000         5228.000
        11              NaN              NaN              NaN              NaN
        12              NaN              NaN              NaN              NaN
        13              NaN              NaN              NaN              NaN
        14              NaN              NaN              NaN              NaN
        15              NaN              NaN              NaN              NaN
        16              NaN              NaN              NaN              NaN
        17              NaN              NaN              NaN              NaN
        18              NaN              NaN              NaN              NaN
        19              NaN              NaN              NaN              NaN
        20              NaN              NaN              NaN              NaN
        21              NaN              NaN              NaN              NaN
        """
        #pylint:disable=too-many-locals

        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns

        dtypes = self.dtypes(cols)
        # Not sure if this is the complete list but should cover most data types
        # Note that we don't cover BLOB/LOB types
        numerics = [col_info[0] for col_info in dtypes
                    if col_info[1] == "INT" or
                    col_info[1] == 'TINYINT' or
                    col_info[1] == 'BIGINT' or
                    col_info[1] == 'INTEGER' or
                    col_info[1] == 'DOUBLE' or
                    col_info[1] == 'DECIMAL' or
                    col_info[1] == 'FLOAT']
        non_numerics = [col_info[0] for col_info in dtypes
                        if col_info[1] == "NCHAR" or
                        col_info[1] == 'NVARCHAR' or
                        col_info[1] == 'CHAR' or
                        col_info[1] == 'VARCHAR' or
                        col_info[1] == 'STRING']

        # The reason to separate numerics and non-numerics is the calculation
        # of min and max. These functions are of different types when calculating
        # for different types. So, a numeric type will return a numeric value
        # while a character column will return a character value for min and max
        # When both column types are present, the values for min and max would be null
        # for non numeric types
        sql_numerics = None
        sql_non_numerics = None
        min_max = 'MIN({0}) as "min", MAX({0}) as "max", '
        if numerics:
            sql_for_numerics = ('select {3} as "column", COUNT({0}) as "count", ' +
                                'COUNT(DISTINCT {0}) as "unique", ' +
                                'SUM(CASE WHEN {0} is NULL THEN 1 ELSE 0 END) as "nulls", ' +
                                'AVG(TO_DOUBLE({0})) as "mean", STDDEV({0}) as "std", ' +
                                min_max + 'MEDIAN({0}) as "median" ' +
                                'FROM ({1}) AS {2}')
            union = [sql_for_numerics.format(quotename(col), self.select_statement,
                                             self.quoted_name,
                                             "'{}'".format(col.replace("'", "''")))
                     for col in numerics]
            sql_simple_stats = " UNION ALL ".join(union)

            percentiles = ('SELECT {3} as "column", * FROM (SELECT ' +
                           'percentile_cont(0.25) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "25_percent_cont", '+
                           'percentile_disc(0.25) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "25_percent_disc", '+
                           'percentile_cont(0.50) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "50_percent_cont", '+
                           'percentile_disc(0.50) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "50_percent_disc", '+
                           'percentile_cont(0.75) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "75_percent_cont", '+
                           'percentile_disc(0.75) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "75_percent_disc" '+
                           'FROM ({1}) AS {2})')
            union = [percentiles.format(quotename(col), self.select_statement,
                                        self.quoted_name,
                                        "'{}'".format(col.replace("'", "''"))) for col in numerics]
            sql_percentiles = " UNION ALL ".join(union)

            sql_numerics = ('SELECT {0}.*, '.format('"SimpleStats"') +
                            ', '.join(['{0}."{1}_percent_cont", {0}."{1}_percent_disc"'.
                                       format('"Percentiles"', percentile)
                                       for percentile in [25, 50, 75]]) +
                            ' FROM ({0}) AS {1}, ({2}) AS {3}'.
                            format(sql_simple_stats, '"SimpleStats"',
                                   sql_percentiles, '"Percentiles"') +
                            ' WHERE {0}."column" = {1}."column"'.
                            format('"SimpleStats"', '"Percentiles"'))
            # This is to handle the case for non-numerics since min and max values
            # are now not compatible between numerics and non-numerics
            min_max = 'CAST(NULL AS DOUBLE) AS "min", CAST(NULL AS DOUBLE) AS "max", '

        if non_numerics:
            sql_for_non_numerics = ('select {3} as "column", COUNT({0}) as "count", ' +
                                    'COUNT(DISTINCT {0}) as "unique", ' +
                                    'SUM(CASE WHEN {0} IS NULL THEN 1 ELSE 0 END) as "nulls", ' +
                                    'CAST(NULL as DOUBLE) AS "mean", ' +
                                    'CAST(NULL as double) as "std", ' +
                                    min_max + 'CAST(NULL as DOUBLE) AS "median", ' +
                                    'CAST(NULL AS DOUBLE) AS "25_percent_cont", ' +
                                    'CAST(NULL AS DOUBLE) AS "25_percent_disc", ' +
                                    'CAST(NULL AS DOUBLE) AS "50_percent_cont", ' +
                                    'CAST(NULL AS DOUBLE) AS "50_percent_disc", ' +
                                    'CAST(NULL AS DOUBLE) AS "75_percent_cont", ' +
                                    'CAST(NULL AS DOUBLE) AS "75_percent_disc" ' +
                                    'FROM ({1}) AS {2}')
            union = [sql_for_non_numerics.format(quotename(col), self.select_statement,
                                                 self.quoted_name,
                                                 "'{}'".format(col.replace("'", "''")))
                     for col in non_numerics]
            sql_non_numerics = " UNION ALL ".join(union)

        if sql_numerics is None and sql_non_numerics is None:
            raise ValueError('Parameter cols cannot be described')

        if sql_numerics is not None and sql_non_numerics is not None:
            sql_combined = ('SELECT * FROM ({0}) AS {1} UNION ALL SELECT * FROM ({2}) AS {3}'.
                            format(sql_numerics,
                                   '"Numerics"',
                                   sql_non_numerics,
                                   '"NonNumerics"'))
            return self._df(sql_combined, propagate_safety=True)
        if sql_numerics is not None:
            return self._df(sql_numerics, propagate_safety=True)

        return self._df(sql_non_numerics, propagate_safety=True)

    def bin(self, col, strategy='uniform_number', bins=None, bin_width=None,  #pylint: disable=too-many-arguments
            bin_column='BIN_NUMBER'):
        """
        Returns a DataFrame with original columns along with bin assignments.

        The name of the columns in the returned DataFrame is the same as the
        original column. Column "BIN_NUMBER" or the specified value in
        `bin_column` is added and corresponds to the bin assigned.

        Parameters
        ----------
        col : str
            The column on which binning is performed.
            The column must be numeric.
        strategy : {'uniform_number', 'uniform_size'}
            Binning methods:
                - 'uniform_number': Equal widths based on the number of bins.
                - 'uniform_size': Equal widths based on the bin size.
        bins : int, optional
            The number of equal width bins.
            Only valid when `strategy` is 'uniform_number'.
            Defaults to 10.
        bin_width : int, optional
            The interval width of each bin.
            Only valid when `strategy` is 'uniform_size'.
        bin_column : str, optional
            Name of the output column containing the bin number.

        Returns
        -------
        DataFrame
            Binned dataset with the same data as this DataFrame
            along with an additional column "BIN_NUMBER" or the value specified
            in `bin_column`. This additional column contains the
            assigned bin for each row.

        Examples
        --------
        Input:

        >>> df.collect()
           C1   C2    C3       C4
        0   1  1.2   2.0      1.0
        1   2  1.4   4.0      3.0
        2   3  1.6   6.0      9.0
        3   4  1.8   8.0     27.0
        4   5  2.0  10.0     81.0
        5   6  2.2  12.0    243.0
        6   7  2.4  14.0    729.0
        7   8  2.6  16.0   2187.0
        8   9  2.8  18.0   6561.0
        9  10  3.0  20.0  19683.0

        Create 5 bins of equal widths on C1

        >>> df.bin('C1', strategy='uniform_number', bins=5).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           2
        3   4  1.8   8.0     27.0           2
        4   5  2.0  10.0     81.0           3
        5   6  2.2  12.0    243.0           3
        6   7  2.4  14.0    729.0           4
        7   8  2.6  16.0   2187.0           4
        8   9  2.8  18.0   6561.0           5
        9  10  3.0  20.0  19683.0           5

        Create 5 bins of equal widths on C2

        >>> df.bin('C3', strategy='uniform_number', bins=5).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           2
        3   4  1.8   8.0     27.0           2
        4   5  2.0  10.0     81.0           3
        5   6  2.2  12.0    243.0           3
        6   7  2.4  14.0    729.0           4
        7   8  2.6  16.0   2187.0           4
        8   9  2.8  18.0   6561.0           5
        9  10  3.0  20.0  19683.0           5

        Create 5 bins of equal widths on a column that varies significantly

        >>> df.bin('C4', strategy='uniform_number', bins=5).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           1
        3   4  1.8   8.0     27.0           1
        4   5  2.0  10.0     81.0           1
        5   6  2.2  12.0    243.0           1
        6   7  2.4  14.0    729.0           1
        7   8  2.6  16.0   2187.0           1
        8   9  2.8  18.0   6561.0           2
        9  10  3.0  20.0  19683.0           5

        Create bins of equal width

        >>> df.bin('C1', strategy='uniform_size', bin_width=3).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           2
        3   4  1.8   8.0     27.0           2
        4   5  2.0  10.0     81.0           2
        5   6  2.2  12.0    243.0           3
        6   7  2.4  14.0    729.0           3
        7   8  2.6  16.0   2187.0           3
        8   9  2.8  18.0   6561.0           4
        9  10  3.0  20.0  19683.0           4
        """
        if not isinstance(col, _STRING_TYPES):
            raise TypeError('Parameter col must be a string.')

        self._validate_columns_in_dataframe([col])

        if not self.is_numeric(col):
            raise ValueError('Parameter col must be a numeric column.')
        if not isinstance(strategy, _STRING_TYPES):
            raise TypeError('Parameter strategy must be a string')
        strategy = strategy.lower()
        if strategy not in ['uniform_number', 'uniform_size']:
            raise TypeError('Parameter strategy must be one of "uniform_number", "uniform_size".')

        if strategy == 'uniform_number':
            if bins is None:
                bins = 10
            if bin_width is not None:
                raise ValueError('Parameter bin_size invalid with strategy "uniform_number"')
        elif strategy == 'uniform_size':
            if (not isinstance(bin_width, _INTEGER_TYPES) and
                    not isinstance(bin_width, float)):
                raise TypeError('Parameter bin_width must be a numeric.')

        sql_template = 'SELECT *, BINNING(VALUE => {}, {} => {}) OVER() AS {} FROM ({}) AS {}'

        bin_select = sql_template.format(
            quotename(col),
            'BIN_COUNT' if strategy == 'uniform_number' else 'BIN_WIDTH',
            bins if strategy == 'uniform_number' else bin_width,
            quotename(bin_column), self.select_statement, self.quoted_name)
        return self._df(bin_select, propagate_safety=True)

    def agg(self, agg_list, group_by=None):
        """
        Returns a DataFrame with the group_by column along with the aggregates.

        The name of the column in the returned DataFrame is the same as the
        original column.

        Parameters
        ----------

        agg_list : list of tuple

            A list of tuples. Each tuple is a triplet.
            The triplet consists of (aggregate_operator, expression, name) where:

                - aggregate_operator is one of ['max', 'min', 'count', 'avg']
                - expression is a str that is a column or column expression
                  name is the name of this aggregate in the project list.

        group_by : str or list of str, optional

            Group by column. Note that only a column is allowed although
            expressions are allowed in SQL. To group by an expression, a
            dataframe would have to be created by providing the entire SQL.
            So, if you have a table T with columns C1, C2, and C3 that are all
            integers, to calculate the max(C1) grouped by (C2+C3) a DataFrame
            would need to be created as below:

                cc.sql('SELECT "C2"+"C3", max("C1") FROM "T" GROUP BY "C2"+"C3"')

        Returns
        -------

        DataFrame
            DataFrame containing the group_by column if it exists along with
            the aggregate expressions that are aliased with the specified names.

        Examples
        --------

        Input:

        >>> df.collect()
            ID  SEPALLENGTHCM  SEPALWIDTHCM  PETALLENGTHCM  PETALWIDTHCM          SPECIES
        0    1            5.1           3.5            1.4           0.2      Iris-setosa
        1    2            4.9           3.0            1.4           0.2      Iris-setosa
        2    3            4.7           3.2            1.3           0.2      Iris-setosa
        3   51            7.0           3.2            4.7           1.4  Iris-versicolor
        4   52            6.4           3.2            4.5           1.5  Iris-versicolor
        5  101            6.3           3.3            6.0           2.5   Iris-virginica
        6  102            5.8           2.7            5.1           1.9   Iris-virginica
        7  103            7.1           3.0            5.9           2.1   Iris-virginica
        8  104            6.3           2.9            5.6           1.8   Iris-virginica

        Another way to do a count.

        >>> df.agg([('count', 'SPECIES', 'COUNT')]).collect()
            COUNT
        0      9

        Get counts by SPECIES

        >>> df.agg([('count', 'SPECIES', 'COUNT')], group_by='SPECIES').collect()
                   SPECIES  COUNT
        0  Iris-versicolor      2
        1   Iris-virginica      4
        2      Iris-setosa      3

        Get max values of SEPALLENGTHCM by SPECIES

        >>> df.agg([('max', 'SEPALLENGTHCM', 'MAX_SEPAL_LENGTH')], group_by='SPECIES').collect()
                   SPECIES  MAX_SEPAL_LENGTH
        0  Iris-versicolor               7.0
        1   Iris-virginica               7.1
        2      Iris-setosa               5.1

        Get max and min values of SEPALLENGTHCM by SPECIES

        >>> df.agg([('max', 'SEPALLENGTHCM', 'MAX_SEPAL_LENGTH'),
            ('min', 'SEPALLENGTHCM', 'MIN_SEPAL_LENGTH')], group_by=['SPECIES']).collect()
                   SPECIES  MAX_SEPAL_LENGTH  MIN_SEPAL_LENGTH
        0  Iris-versicolor               7.0               6.4
        1   Iris-virginica               7.1               5.8
        2      Iris-setosa               5.1               4.7

        Get aggregate grouping by multiple columns.

        >>> df.agg([('count', 'SEPALLENGTHCM', 'COUNT_SEPAL_LENGTH')],
                    group_by=['SPECIES', 'PETALLENGTHCM']).collect()
                   SPECIES  PETALLENGTHCM  COUNT_SEPAL_LENGTH
        0   Iris-virginica            6.0                   1
        1      Iris-setosa            1.3                   1
        2   Iris-virginica            5.9                   1
        3   Iris-virginica            5.6                   1
        4      Iris-setosa            1.4                   2
        5  Iris-versicolor            4.7                   1
        6  Iris-versicolor            4.5                   1
        7   Iris-virginica            5.1                   1
        """

        valid_aggregates = ['max', 'min', 'count', 'avg', 'median']

        if group_by is not None:
            msg = 'Parameter group_by must be a string or a list of strings.'
            group_by = DataFrame._get_list_of_str(group_by, msg)
            self._validate_columns_in_dataframe(group_by)
            group_by = [quotename(gb) for gb in group_by]

        if not isinstance(agg_list, list):
            raise TypeError('Parameter agg_list must be a list.')
        if not agg_list:
            raise ValueError('Parameter agg_list must contain at least one tuple.')
        aggregates = []
        for item in agg_list:
            if not isinstance(item, tuple):
                raise TypeError('Parameter agg_list must be a tuple with 3 elements.')
            if len(item) != 3:
                raise TypeError('Parameter agg_list must be a list of tuples with 3 elements.')
            (agg, expr, name) = item
            agg = agg.lower()
            if agg not in valid_aggregates:
                raise ValueError('The first element of each tuple must be one of {}.'
                                 .format(valid_aggregates))
            aggregates.append('{}({}) AS {}'.format(agg, quotename(expr), quotename(name)))

        sql = ('SELECT {} {} FROM ({}) AS {}{}'
               .format('' if group_by is None else (','.join(group_by) + ','),
                       ', '.join(aggregates),
                       self.select_statement,
                       self.quoted_name,
                       '' if group_by is None
                       else ' GROUP BY {}'.format(','.join(group_by))))
        return self._df(sql)

    def is_numeric(self, cols=None):
        """
        Returns True if the column(s) in the DataFrame are numeric.

        Parameters
        ----------
        cols : str or list
            The column(s) to be tested for being numeric. Defaults to all columns.

        Returns
        -------
        bool
            True if all the cols are numeric.

        Examples
        --------
        Input:

        >>> df.head(5).collect()
        ID  SEPALLENGTHCM  SEPALWIDTHCM  PETALLENGTHCM  PETALWIDTHCM      SPECIES
        0   1            5.1           3.5            1.4           0.2  Iris-setosa
        1   2            4.9           3.0            1.4           0.2  Iris-setosa
        2   3            4.7           3.2            1.3           0.2  Iris-setosa
        3   4            4.6           3.1            1.5           0.2  Iris-setosa
        4   5            5.0           3.6            1.4           0.2  Iris-setosa
        >>> pprint.pprint(df.dtypes())
        [('ID', 'INT', 10),
         ('SEPALLENGTHCM', 'DOUBLE', 15),
         ('SEPALWIDTHCM', 'DOUBLE', 15),
         ('PETALLENGTHCM', 'DOUBLE', 15),
         ('PETALWIDTHCM', 'DOUBLE', 15),
         ('SPECIES', 'NVARCHAR', 15)]

        Test a single column

        >>> df.is_numeric('ID')
        True
        >>> df.is_numeric('SEPALLENGTHCM')
        True
        >>> df.is_numeric(['SPECIES'])
        False

        Test a list of columns

        >>> df.is_numeric(['SEPALLENGTHCM', 'PETALLENGTHCM', 'PETALWIDTHCM'])
        True
        >>> df.is_numeric(['SEPALLENGTHCM', 'PETALLENGTHCM', 'SPECIES'])
        False
        """
        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings.'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns

        return all(col_info[1] in ['INT', 'TINYINT' 'BIGINT',
                                   'INTEGER', 'DOUBLE', 'DECIMAL', 'FLOAT']
                   for col_info in self.dtypes(cols))

    def corr(self, first_col, second_col):
        """
        Returns a DataFrame that gives the correlation coefficient between two
        numeric columns.

        All rows with NULL values for first_col or second_col are removed
        prior to calculating the correlation coefficient.

        The correlation coefficient is:
            1/(n-1) * sum((col1_value - avg(col1)) * (col2 - avg(col2))) /
            (stddev(col1) * stddev(col2))

        Parameters
        ----------
        first_col : str
            The first column for calculating the correlation coefficient.
        first_col : str
            The second column for calculating the correlation coefficient.

        Returns
        -------
        DataFrame
            A DataFrame with one value containing the correlation coefficient.
            The name of the column is CORR_COEFF.

        Examples
        --------
        Input:

        >>> df.columns
        ['C1', 'C2', 'C3', 'C4']
        >>> df.collect()
           C1   C2      C3       C4
        0   1  1.2     2.0      1.0
        1   2  1.4     4.0      3.0
        2   3  1.6     8.0      9.0
        3   4  1.8    16.0     27.0
        4   5  2.0    32.0     81.0
        5   6  2.2    64.0    243.0
        6   7  2.4   128.0    729.0
        7   8  2.6   256.0   2187.0
        8   9  2.8   512.0   6561.0
        9  10  3.0  1024.0  19683.0

        Correlation with columns that are well correlated.

        >>> df.corr('C1', 'C2').collect()
              CORR_COEFF
        0         1.0

        >>> df.corr('C1', 'C3').collect()
            CORR_COEFF
        0         1.0

        Correlation with a column whose value is 3 times its previous value

        >>> df.corr('C1', 'C4').collect()
            CORR_COEFF
        0    0.696325
        """
        if not isinstance(first_col, _STRING_TYPES):
            raise TypeError('Parameter first_col must be a string.')
        if not isinstance(second_col, _STRING_TYPES):
            raise TypeError('Parameter second_col must be a string.')
        if not self.is_numeric([first_col, second_col]):
            raise ValueError('Correlation columns {0} and {1} must be numeric.'.
                             format(first_col, second_col))
        dt_sql = ('SELECT * FROM ({0}) AS {1} WHERE ' +
                  '{1}.{2} is not NULL and {1}.{3} is not NULL')
        derived_table = self._df(dt_sql.format(self.select_statement,
                                               self.quoted_name, quotename(first_col), quotename(second_col)))

        corr_select = ('select A.A/B.B AS CORR_COEFF from (select (sum(({0} - ' +
                       '(select avg({0}) from ({2}) )) * ({1} - ' +
                       '(select avg({1}) from ({2}) )))) as A from ({2})) A, ' +
                       '(select stddev({0})*stddev({1})*(count(*)-1) as B ' +
                       'from ({2})) B ').format(quotename(first_col), quotename(second_col),
                                                derived_table.select_statement)

        dfc = self._df(corr_select, propagate_safety=True)
        return dfc

    def pivot_table(self, values, index, columns, aggfunc='avg'):
        """
        Returns a DataFrame that gives the pivoted table.

        aggfunc is indentical to HANA agg functions.

        Parameters
        ----------
        values : str
            The targted values for pivoting.
        index : str
            The index of the DataFrame.
        columns : str
            The pivoting columns.

        Returns
        -------
        DataFrame
            A pivoted DataFrame

        Examples
        --------
        >>> df.pivot_table(values='C2', index='C1', columns='C3', aggfunc='max')

        """
        col_set = self.__run_query('SELECT ' + ' distinct "' + columns + '"' + ' FROM (' + self.select_statement + ')')
        col_set = set(map(lambda x: x[0], col_set))
        sql_script = 'SELECT ' + index + ', '
        for col in col_set:
            sql_script = sql_script + aggfunc + '(CASE WHEN ' + '"' + columns + '"' + \
            '=' + "'" + col + "'" + ' THEN ' + '"' + values + '"' + ' END) AS ' + '"' + col + '"' + ','
        sql_script = sql_script[:-1]
        sql_script = sql_script + ' FROM (' + self.select_statement + ') GROUP BY ' + '"' + index + '"'
        return self._df(sql_script)

    # SQLTRACE
    def generate_table_type(self):
        """
        Generate hana table type based on the dtypes of the dataframe. Convenience method for sql trace

        Returns
        -------
        Table Type : str
            The table type in string form

        """
        dtypes = self.dtypes()
        type_string = "" + "("
        first = True
        for rows in dtypes:
            if first:
                first = False
            else:
                type_string = type_string + ","
            if rows[1] == 'INT' or rows[1] == 'DOUBLE':
                type_part = "\"" + rows[0] +"\""  + " " + rows[1]
                type_string = type_string + type_part
            if rows[1] == 'VARCHAR' or rows[1] == 'NVARCHAR':
                type_part = "\"" + rows[0] +"\"" + " " + rows[1] +"(" + str(rows[2]) + ")"
                type_string = type_string + type_part
        type_string = type_string + ")"
        table_types = "table {}".format(type_string)
        return table_types

def create_dataframe_from_pandas(connection_context, pandas_df, table_name, force=False, replace=True):
    """
    Upload data from pandas dataframe to HANA DB and returns a HANA DataFrame.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection to the HANA system.
    pandas_df : pandas.DataFrame
        Pandas DataFrame for uploading to HANA DB.
    table_name : str
        Table name in HANA DB.
    replace : bool
        If replace is True, HANA table will do the missing value handling.
    force : bool
        If force is True, HANA table with table_name will be dropped.

    Returns
    -------
    DataFrame
        HANA DataFrame containing data in pandas_df.

    Examples
    --------
    >>> create_dataframe_from_pandas(connection_context, p_df, 'test', force=False, replace=True)
    <hana_ml.dataframe.DataFrame at 0x7efbcb26fbe0>
    """
    cursor = connection_context.connection.cursor()
    if force is True:
        sql_script = 'DROP TABLE "{}";'.format(table_name)
        #execute drop table with try catch
        try:
            cursor.execute(sql_script)
        except dbapi.Error as db_er:
            logger.warning(str(db_er))
    #create table script
    sql_script = 'CREATE TABLE "{}" ('.format(table_name)
    if '#' in table_name[:1]:
        sql_script = 'CREATE LOCAL TEMPORARY TABLE {} ('.format(table_name)
    dtypes_list = list(map(str, pandas_df.dtypes.values))
    for col, dtype in zip(pandas_df.columns, dtypes_list):
        if 'int' in dtype:
            sql_script = sql_script + '"{}" {}, '.format(col, 'INT')
        elif 'float' in dtype or 'double' in dtype:
            sql_script = sql_script + '"{}" {}, '.format(col, 'DOUBLE')
        else:
            sql_script = sql_script + '"{}" {}, '.format(col, 'VARCHAR(5000)')
    sql_script = sql_script[:-2]
    sql_script = sql_script + ');'
    try:
        cursor.execute(sql_script)
    except dbapi.Error as db_er:
        logger.error(str(db_er))
        cursor.close()
        raise
    pandas_df_temp = pandas_df
    if pandas_df_temp.isnull().values.any():
        logger.warning("Replace nan with 0 in numeric columns and '' in string columns.")
    for col in pandas_df_temp:
        #get dtype for column
        dtc = pandas_df_temp[col].dtype
        #check if it is a number
        if replace is True:
            if dtc == int or dtc == float:
                pandas_df_temp[col].fillna(0, inplace=True)
            else:
                pandas_df_temp[col].fillna("", inplace=True)
        else:
            pandas_df_temp[col] = np.where(pandas_df_temp[col].isnull(), None, pandas_df_temp[col])
    rows = tuple(map(tuple, [[element.item() if 'numpy' in str(type(element)) else element for element in lines] for lines in pandas_df_temp.values]))
    if rows:
        parms = ("?," * len(rows[0]))[:-1]
        sql = 'INSERT INTO "{}" VALUES ({})'.format(table_name, parms)
        cursor.executemany(sql, rows)
    cursor.close()
    connection_context.connection.commit()
    return DataFrame(connection_context, 'SELECT * FROM "{}"'.format(table_name))
