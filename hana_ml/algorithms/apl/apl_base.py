#pylint: disable=too-many-lines
"""
This module contains root abstract class and common helper for SAP HANA APL.

"""
import re
import logging
from math import isnan
import sys
import uuid
import functools
import pandas as pd
from hdbcli import dbapi
from hana_ml.dataframe import (
    quotename)
from hana_ml.ml_base import execute_logged
from hana_ml.ml_base import (
    MLBase,
    Table,
    sql_for_drop_table,
    sql_for_insert_values,
    ListOfStrings,
    try_drop,
    _TEXT_TYPES
)
from hana_ml.ml_exceptions import FitIncompleteError

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

# ----- Static  ---------------------

# Type conversion mapping for ANY
TYPE_CONVERSION_MAP = {
    'TINYINT': 'TO_INTEGER({NAME})',
    'SMALLINT': 'TO_INTEGER({NAME})',
    'BOOLEAN': 'TO_INTEGER({NAME})',
    'ALPHANUM': 'TO_NVARCHAR({NAME})',
    'SHORTTEXT': 'TO_NVARCHAR({NAME})',
    'SMALLDECIMAL': 'TO_DOUBLE({NAME})',
    'REAL': 'TO_DOUBLE({NAME})',
    'FLOAT': 'TO_DOUBLE({NAME})'}


def get_apl_version(conn):
    """
    Gets the version and configuration information about the installation of SAP HANA APL.
    Arguments:
    ---------
    - conn_context : SAP HANA connection
    Returns:
    -------
    A pandas Dataframe with detailed information about the current version.

    Error is raised when the call fails.
    The cause can be that either SAP HANA APL is not installed
    or the current user does not have the appropriate rights.
    """
    with conn.connection.cursor() as cur:
        try:
            sql = ('DROP TABLE #PING_OUTPUT')
            execute_logged(cur, sql)
        except dbapi.Error:
            pass
        try:
            sql = ('create local temporary column table #PING_OUTPUT'
                   + APLArtifactTable.get_tbl_def_for_ping())
            execute_logged(cur, sql)
            sql = 'call _SYS_AFL.APL_AREA_PING_PROC(#PING_OUTPUT) with overview'
            execute_logged(cur, sql)
            sql = 'select * from #PING_OUTPUT'
            execute_logged(cur, sql)
            res = cur.fetchall()
            return pd.DataFrame(res, columns=['name', 'value'])
        except dbapi.Error:
            raise Exception('Unable to APL. please check with your database administrator')


def is_apl_version_compatible(conn):
    """
    Checks the installed SAP HANA APL version is the right one.
    Arguments:
    ---------
    - conn_context : SAP HANA connection
    Returns:
    -------
    True if the installed version is correct
    False if it is not
    """
    try:
        version = get_apl_version(conn)
        apl_major_ver = int(version.loc[version['name'] == 'APL.Version.Major',
                                        'value'].values[0])
        apl_minor_ver = int(version.loc[version['name'] == 'APL.Version.Minor',
                                        'value'].values[0])
        apl_sp = int(version.loc[version['name'] == 'APL.Version.ServicePack',
                                 'value'].values[0])
        afl_ver = version.loc[version['name'] == 'AFLSDK.Info', 'value'].values[0]
        msg = "Installed Version (APL={apl_major_ver}.{apl_minor_ver} SP {apl_sp}/" \
              " AFL={afl_ver})".format(apl_major_ver=apl_major_ver,
                                       apl_minor_ver=apl_minor_ver,
                                       apl_sp=apl_sp,
                                       afl_ver=afl_ver)
        if apl_sp >= 1811 and afl_ver.startswith('2.13'):
            msg = msg + " is compatible"
            logger.info(msg)
            return True

        msg = msg + " is NOT compatible"
        logger.error(msg)
        return False
    except dbapi.Error:
        # The version checking simply because APL versionning has been changed over time.
        # The easiest is to update APL to a most recent version
        errmsg = sys.exc_info()[0]
        msg = "Cannot check version (Error: {}). Please update APL to the latest version".format(
            errmsg
        )
        logger.error(msg)
    return False


def config_logger(
        log_path=None, logfile_name=None,
        log_console=True, log_level=logging.ERROR):
    """
    Configures the logger so it can either display the Python API log in a console
    or in a file.
    Arguments:
    - log_path : str
        The folder in which logs will be written
    - logfile_name : str
        The file name
    - log_console : boolean
        If true, the log will display in a console
    - log_level : int (logging.ERROR, logging.INFO, ...)
    """
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if log_path and logfile_name:
        file_handler = logging.FileHandler(
            "{0}/{1}.log".format(log_path, logfile_name))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if log_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)


class APLArtifactTable(Table):
    """
    Class used to represent APL artifact tables (model, input and output tables).
    """
    # --- Artifact Table type name
    FUNCTION_HEADER = 'FUNCTION_HEADER'
    OPERATION_CONFIG_EXTENDED = 'OPERATION_CONFIG_EXTENDED'
    VARIABLE_DESC_OID = 'VARIABLE_DESC_OID'
    VARIABLE_ROLES_WITH_COMPOSITES_OID = 'VARIABLE_ROLES_WITH_COMPOSITES_OID'
    MODEL_BIN_OID = 'MODEL_BIN_OID'
    OPERATION_LOG = 'OPERATION_LOG'
    SUMMARY = 'SUMMARY'
    INDICATORS = 'INDICATORS'
    TABLE_TYPE = 'TABLE_TYPE'
    PING_OUTPUT = 'PING_OUTPUT'
    TS_DEFAULT_APPLYOUT = 'TS_DEFAULT_APPLYOUT'

    # Default artifact Table Definitions
    _tbl_def_map = {
        'FUNCTION_HEADER': '(KEY NVARCHAR(50), VALUE NVARCHAR(255))',
        'OPERATION_CONFIG_EXTENDED': ('(KEY NVARCHAR(1000), VALUE NCLOB, '
                                      'CONTEXT NVARCHAR(100))'),
        'VARIABLE_DESC_OID': ('(RANK INT, NAME NVARCHAR(127), STORAGE NVARCHAR(10), '
                              'VALUETYPE NVARCHAR(10), KEYLEVEL INT, ORDERLEVEL INT, '
                              'MISSINGSTRING NVARCHAR(255), GROUPNAME NVARCHAR(255), '
                              'DESCRIPTION NVARCHAR(255), OID NVARCHAR(50))'),
        'VARIABLE_ROLES_WITH_COMPOSITES_OID': ('( NAME NVARCHAR(127), ROLE NVARCHAR(10), '
                                               'COMPOSITION_TYPE NVARCHAR(10), '
                                               'COMPONENT_NAME NVARCHAR(127), OID NVARCHAR(50))'),
        'MODEL_BIN_OID': '(OID NVARCHAR(50), FORMAT NVARCHAR(50), LOB CLOB)',
        'OPERATION_LOG': ('(OID NVARCHAR(50), TIMESTAMP LONGDATE, LEVEL INT, ORIGIN NVARCHAR(50), '
                          'MESSAGE NCLOB)'),
        'SUMMARY': '(OID NVARCHAR(50), KEY NVARCHAR(100), VALUE NVARCHAR(200))',
        'INDICATORS': ('(OID NVARCHAR(50), VARIABLE NVARCHAR(127), TARGET NVARCHAR(127), '
                       'KEY NVARCHAR(127), VALUE NCLOB, '
                       'DETAIL NCLOB)'),
        'TABLE_TYPE': ('(OID NVARCHAR(50), POSITION INT, NAME NVARCHAR(127), KIND NVARCHAR(50), '
                       'PRECISION INT, SCALE INT, MAXIMUM_LENGTH INT)'),
        'PING_OUTPUT': '(name NVARCHAR(128), value NCLOB)',
        }

    def __init__(self, name, type_name, apl_version, data=None):
        """
        Parameters
        ---------
        name:  str
            The new table name
        apl_version: int
            The current version number of SAP HANA APL.
            This parameter is meant to serve in the futur when the table definitions vary with
            the version.
        type_name: str
            The type name. It is defined by APL. Use class variable, for example,
            APLArtifactTable.FUNCTION_HEADER.
        data: List of tuples
            The table content presented by a list of tuples.
            Each tuple is made of the different values of a row.
        """
        super(APLArtifactTable, self).__init__(name, None)
        self.apl_version = apl_version
        self.type_name = type_name
        self.spec = None
        self.data = data
        self._artifact_tables = []

    @staticmethod
    def get_tbl_def_for_ping():
        """
        Returns
        -------
        The table definition of PING_OUT: str
        """
        return APLArtifactTable._tbl_def_map[APLArtifactTable.PING_OUTPUT]

    def get_table_definition(self):
        """
        Returns
        -------
        The table definition: str. For example, '(KEY NVARCHAR(50), VALUE NVARCHAR(255))'
        """
        # For the moment, the apl version does not matter
        return self._tbl_def_map[self.type_name]  # default value

    def create_table_with_hana_type(self, conn, force=True):
        """
        Creates a temporary table for APL artifact.
        The creation is based on a predefined APL type.

        Arguments:
        ---------
        conn_context :  ConnectionContext
            The database connection object
        force: boolean
            If True, the table with the same name will be dropped when it exists.
        """
        with conn.connection.cursor() as cur:
            if force:
                try:
                    execute_logged(cur, sql_for_drop_table(self.name))
                except dbapi.Error:
                    pass
            table_spec = self.get_table_definition()
            sql_create_table = (
                "CREATE LOCAL TEMPORARY COLUMN TABLE {} {}".format(
                    quotename(self.name),
                    table_spec)
                )
            execute_logged(cur, sql_create_table)
            if self.data:
                # non-None, non-empty
                statement = sql_for_insert_values(self.name, self.data)
                logger.info("Prepare SQL: %s", statement)
                for row in self.data:
                    logger.info("Executing SQL: INSERT INTO %s values %s",
                                quotename(self.name),
                                str(row))
                cur.executemany(statement, self.data)


class APLBase(MLBase): #pylint: disable=too-many-instance-attributes
    """
    Common abstract root class for all APL algorithms.
    """
    # Dictionary : model parameter <-> APL alias (to be used in OPERATION_CONFIG)
    APL_ALIAS_KEYS = {
        'model_type': 'APL/ModelType',
        '_algorithm_name': 'APL/AlgorithmName',
        'correlations_lower_bound': 'APL/CorrelationsLowerBound',
        'correlations_max_kept': 'APL/CorrelationsMaxKept',
        'cutting_strategy': 'APL/CuttingStrategy',
        'exclude_low_predictive_confidence': 'APL/ExcludeLowPredictiveConfidence',
        'polynomial_degree': 'APL/PolynonimalDegree',
        'risk_fitting': 'APL/RiskFitting',
        'risk_fitting_min_cumulated_frequency': 'APL/RiskFittingMinCumulatedFrequency',
        'risk_fitting_nb_pdo': 'APL/RiskFittingNbPDO',
        'risk_fitting_use_weights': 'APL/RiskFittingUseWeights',
        'risk_gdo': 'APL/RiskGDO',
        'risk_mode': 'APL/RiskMode',
        'risk_pdo': 'APL/RiskPDO',
        'risk_score': 'APL/RiskScore',
        'score_bins_count': 'APL/ScoreBinsCount',
        'target_key': 'APL/TargetKey',
        'variable_auto_selection': 'APL/VariableAutoSelection',
        'variable_selection_best_iteration': 'APL/VariableSelectionBestIteration',
        'variable_selection_min_nb_of_final_variables':
            'APL/VariableSelectionMinNbOfFinalVariables',
        'variable_selection_max_nb_of_final_variables':
            'APL/VariableSelectionMaxNbOfFinalVariables',
        'variable_selection_mode': 'APL/VariableSelectionMode',
        'variable_selection_nb_variables_removed_by_step':
            'APL/VariableSelectionNbVariablesRemovedByStep',
        'variable_selection_percentage_of_contribution_kept_by_step':
            'APL/VariableSelectionPercentageOfContributionKeptByStep',
        'variable_selection_quality_bar': 'APL/VariableSelectionQualityBar',
        'variable_selection_quality_criteria': 'APL/VariableSelectionQualityCriteria'
        }

    def __init__(self, #pylint: disable=too-many-arguments
                 conn_context,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params):
        super(APLBase, self).__init__(conn_context)
        # --- Uniq id
        # --- overwrite id set by MLBase
        uniq_id = str(uuid.uuid4()).replace('-', '_').upper()
        self.id = uniq_id + '_' + str(self.id)
        # --- parameters for variables description
        self.variable_storages = None
        self.variable_value_types = None
        self.variable_missing_strings = None
        self.extra_applyout_settings = None
        if variable_storages is not None:
            self.set_params(variable_storages=variable_storages)
        if variable_value_types is not None:
            self.set_params(variable_value_types=variable_value_types)
        if variable_missing_strings is not None:
            self.set_params(variable_missing_strings=variable_missing_strings)
        if extra_applyout_settings is not None:
            self.set_params(extra_applyout_settings=extra_applyout_settings)
        # other params
        self.set_params(**other_params)
        # if other_params is not None:
        #    for param_name in other_params:
        #        self.set_params(**{param_name: other_params[param_name]})
        # technical params
        self._model_type = None  # it must be set by final class
        self._view_name = None
        self._force_target_var_type = None  # continuous or nominal or none
        self.log_level = 8
        self.model_format = 'bin'
        self.language = 'en'
        self.model_table_ = None
        self.model_ = None
        # memorize artifact tables 'table_name':APLArtifactTable
        self._artifact_tables = {}
        self._apl_version = get_apl_version(self.conn_context)

    def get_params(self):
        """
        Retrieves attributes of the current object.
        This method is implemented for compatibility with Scikit Learn.

        Returns
        -------
        The attribute-values of the model : dictionary

        """
        attrs = {}
        for attrname, value in self.__dict__.items():
            # except id, it is for class attribut
            if attrname != 'id':
                attrs[attrname] = value
        return attrs

    def set_params(self, **parameters):
        """
        Sets attributes of the current model.
        This method is implemented for compatibility with Scikit Learn.

        Parameters
        ----------
        params : dictionary
            The attribute names and values
        """
        # check the param names are subset of possible values
        possible_param_names = set(self.APL_ALIAS_KEYS.keys())
        possible_param_names = (possible_param_names -
                                set(['model_type']))

        for parameter, value in parameters.items():
            if parameter == 'variable_storages':
                self.variable_storages = self._arg_dictionary(
                    name='variable_storages',
                    dictionary=value,
                    possible_values=['integer', 'number', 'string', 'date',
                                     'datetime', 'angle']
                    )
                continue
            if parameter == 'variable_value_types':
                # 'continuous,' 'nominal', 'ordinal'
                self.variable_value_types = self._arg_dictionary(
                    name='variable_value_types',
                    dictionary=value,
                    possible_values=['continuous', 'nominal', 'ordinal',
                                     'textual']
                    )
                continue
            if parameter == 'variable_missing_strings':
                self.variable_missing_strings = self._arg(
                    'variable_missing_strings',
                    value,
                    dict)
                continue
            if parameter == 'extra_applyout_settings':
                self.extra_applyout_settings = self._arg(
                    'extra_applyout_settings',
                    value,
                    dict)
                continue
            if parameter not in possible_param_names:
                template = 'Unknown parameter {!r}'
                raise TypeError(template.format(parameter))

            setattr(self, parameter, value)
        return self

    def _create_artifact_table(self, table):
        """
        Creates a temporary table in the SAP HANA database that is used as artifact
        for SAP HANA APL.
        Argument:
        -------
        - Table : An object of APLArtifactTable
        """
        table.create_table_with_hana_type(conn=self.conn_context)
        self._artifact_tables[table.name] = table

    def _drop_artifact_tables(self):
        tables = self._artifact_tables.values()
        table_names = [t.name for t in tables]
        self._try_drop(table_names)
        self._artifact_tables = {}

    def _call_apl(self, funcname, *tables):
        """
        Calls a SAP HANA APL procedure.
        Parameters
        ----------
        conn : ConnectionContext
            The SAP HANA connection
        funcname : str
            The procedure name
        tablenames : list of str
            The table names used in procedure call
        """
        # Retrieve tablenames. Element in the list can be Table or string
        tablenames = [t if isinstance(t, str) else t.name for t in tables]
        header = 'CALL _SYS_AFL.{}('.format(quotename(funcname))
        arglines_nosep = ['{}'.format(quotename(tabname))
                          for tabname in tablenames]
        arglines_string = ', '.join(arglines_nosep)
        footer = ') WITH OVERVIEW'
        call_string = '{} {} {}'.format(header, arglines_string, footer)
        with self.conn_context.connection.cursor() as cur:
            execute_logged(cur, call_string)

    def _create_func_header_table(self):
        """
        Creates a new APLArtifactTable object for FUNCTION_HEADER table.
        Returns:
        ------
            An APLArtifactTable object for Function Header with data
        """
        # -- Prepare Input parameter tables
        data = [
            ('Oid', '{}'.format(self.id)),
            ('LogLevel', str(self.log_level)),
            ('ModelFormat', self.model_format)]
        if self.language is not None:
            data.append(('Language', self.language))
        func_header_table = APLArtifactTable(
            name='#FUNC_HEADER_{}'.format(self.id),
            type_name=APLArtifactTable.FUNCTION_HEADER,
            apl_version=self._apl_version,
            data=data)
        return func_header_table

    def _create_operation_log_table(self, name=None):
        """
        Creates a new APLArtifactTable object for OPERATION_LOG table.
        Returns:
        ------
            An APLArtifactTable object for Function Header with data
        """
        # -- Prepare Input parameter tables
        if name is None:
            name = '#OPERATION_LOG_{}'.format(self.id)
        table = APLArtifactTable(
            name=name,
            type_name=APLArtifactTable.OPERATION_LOG,
            apl_version=self._apl_version)
        return table

    def _create_train_config_table(self):
        """
        Creates a new APLArtifactTable object for TRAIN_OPERATION_LOG table.
        Returns:
        ------
            An APLArtifactTable object with data
        """
        if self._model_type is None:
            raise FitIncompleteError("Model type undefined.")
        train_config_ar = [(self.APL_ALIAS_KEYS['model_type'],
                            self._model_type, None)]
        # add params to OPERATION_CONFIG table
        for k in self.APL_ALIAS_KEYS:
            param_val = getattr(self, k, None)
            if param_val is not None:
                # Boolean values must be in lower case, except for target_key
                if k != 'target_key' and str(param_val) in ['True', 'False']:
                    param_val = str(param_val).lower()
                train_config_ar.append(
                    (self.APL_ALIAS_KEYS[k], str(param_val), None)
                    )
        train_config_df = pd.DataFrame(train_config_ar)
        if train_config_df is None:
            raise FitIncompleteError("Train configuration undefined.")
        train_config_table = self._create_aplartifact_table_with_data_frame(
            name='#CREATE_AND_TRAIN_CONFIG_{}'.format(self.id),
            type_name=APLArtifactTable.OPERATION_CONFIG_EXTENDED,
            data_df=train_config_df
            )
        return train_config_table

    def _create_var_roles_table(self,
                                data,
                                key,
                                label,
                                features):
        """
        Creates a new APLArtifactTable object for VARABLE_ROLES table.

        Returns
        -------
        An APLArtifactTable object with data
        """
        role_tuples = []
        if features is not None:
            role_tuples.extend(
                [(feat, 'input', None, None, None) for feat in features])
            # skip all those that are not in features, except the target
            for col_name in data.columns:
                if (col_name not in features) and (col_name != label):
                    role_tuples.extend([(col_name, 'skip', None, None, None)])

        if label is not None:
            role_tuples.append((label, 'target', None, None, None))

        # Unless features are explicitly specified, key will be skipped
        if key and not features:
            role_tuples.append((key, 'skip', None, None, None))

        var_roles_table = APLArtifactTable(
            name='#VARIABLE_ROLES_{}'.format(self.id),
            type_name=APLArtifactTable.VARIABLE_ROLES_WITH_COMPOSITES_OID,
            apl_version=self._apl_version,
            data=role_tuples)
        return var_roles_table

    def _create_var_desc_table(self, key, label,
                               data_view_name):
        """
        Creates a new APLArtifactTable object for VAR_DESC table.
        Returns:
        ------
            An APLArtifactTable object with data
        """
        vardesc_df = self._guess_var_description(data_view_name)
        if key is not None:
            # --- If key is given,
            # Overwrite the ID column by setting Key=1
            row_idx = vardesc_df[vardesc_df.NAME == key].index
            vardesc_df.loc[row_idx, 'KEYLEVEL'] = 1
        # Force target variable to nominal (classification)
        if label is not None:
            if self._force_target_var_type is not None:
                vardesc_df.loc[vardesc_df.NAME == label,
                               'VALUETYPE'] = self._force_target_var_type
        if self.variable_storages is not None:
            for var_name in self.variable_storages.keys():
                vardesc_df.loc[vardesc_df.NAME == var_name,
                               'STORAGE'] = self.variable_storages[var_name]
        if self.variable_value_types is not None:
            for var_name in self.variable_value_types.keys():
                vardesc_df.loc[vardesc_df.NAME == var_name,
                               'VALUETYPE'] = self.variable_value_types[var_name]
        if self.variable_missing_strings is not None:
            for var_name in self.variable_missing_strings.keys():
                vardesc_df.loc[vardesc_df.NAME == var_name,
                               'MISSINGSTRING'] = self.variable_missing_strings[var_name]
        # Create APLArtifactTable object for VARIABLE_DESC
        var_desc_table = self._create_aplartifact_table_with_data_frame(
            name='#VARIABLE_DESC_{}'.format(self.id),  # name of table
            type_name=APLArtifactTable.VARIABLE_DESC_OID,  # type
            data_df=vardesc_df
            )
        return var_desc_table

    def _create_aplartifact_table_with_data_frame(
            self,
            name,
            type_name,
            data_df):
        #pylint: disable=too-many-branches
        """
        Creates a new APLArtifactTable object.
        Parameter
        ---------
            name: str,
                The name of Hana table, example, '#VAR_DESC_GUESS'
            type_name: str,
                The predefined APL type , for example, APLArtifactTable.VARIABLE_DESC_OID
            data_df: Pandas dataframe or None
                The content of the table to be inserted.
        Returns:
        ------
            An APLArtifactTable object
        """
        artifact_table = APLArtifactTable(name=name,
                                          type_name=type_name,
                                          apl_version=self._apl_version)
        # Include data into the new APLArtifactTable object
        if data_df is None:
            artifact_table = APLArtifactTable(name,
                                              type_name=type_name,
                                              apl_version=self._apl_version)
        else:
            # Transform dataframe data_df into a list of tuples [(col1, col2, ...)*]
            data_tuples = [tuple(row) for _, row in data_df.iterrows()]
            artifact_table = APLArtifactTable(name,
                                              type_name=type_name,
                                              apl_version=self._apl_version,
                                              data=data_tuples)
        return artifact_table

    @staticmethod
    def _determine_target_var_names(
            input_df,
            label=None,
            vardesc_df=None,
            varroles_df=None):
        """
        Determines the target variable name.

        Arguments:
        --------
        - input_df: hana.ml.dataframe
          The input dataset
        - label: str, optional
            The column name of label
        - vardesc_df : pandas.DataFrame, optional
            The description of the dataset.
            For every column, provide the following:
            RANK, NAME, STORAGE, VALUETYPE, KEYLEVEL, ORDERLEVEL,
                MISSINGSTRING, GROUPNAME, DESCRIPTION, OID
            If it is given, it will be directly applied. Default setting will be skipped.
        - varroles_df: pandas.DataFrame, optional
            Description about the roles of the variables.
            For every column, provide the following:
            NAME, ROLE, COMPOSITION_TYPE, COMPONENT_NAME, OID
        Returns:
        ------
        a list of names
        """
        # if varroles_df is provided,
        # find first var name with role='target' in varroles_df
        if varroles_df is not None:
            return varroles_df.loc[
                varroles_df['ROLE'] == 'target',
                'NAME'].values
        if label is not None:
            return [label]
        if vardesc_df is not None:
            # var name with max rank in vardesc_df
            return [vardesc_df.loc[vardesc_df['RANK'].idxmax(), 'NAME']]
        return [input_df.columns[-1]]

    def _check_value_type_of_target_vars( #pylint: disable=too-many-arguments
            self,
            data,
            good_type,
            label=None,
            vardesc_df=None,
            varroles_df=None):
        """
        Checks the target variable name is correctly set in vardesc table.

        Arguments:
        --------
        - data: hana.ml.dataframe
          The input dataset
        - good_type: 'continuous' for regression or 'nominal' for classification
        - vardesc_df : pandas.DataFrame
            The description of the dataset.
            For every column, provide the following:
            RANK, NAME, STORAGE, VALUETYPE, KEYLEVEL, ORDERLEVEL,
                MISSINGSTRING, GROUPNAME, DESCRIPTION, OID
            If it is given, it will be directly applied. Default setting will be skipped.
        - varroles_df: pandas.DataFrame
            Description about the roles of the variables.
            For every column, provide the following:
            NAME, ROLE, COMPOSITION_TYPE, COMPONENT_NAME, OID
        """
        target_var_names = self._determine_target_var_names(
            input_df=data,
            label=label,
            vardesc_df=vardesc_df,
            varroles_df=varroles_df)
        for name in target_var_names:
            value_type_of_target = vardesc_df.loc[vardesc_df.NAME == name,
                                                  'VALUETYPE'].values
            value_type_of_target = value_type_of_target[0]
            if value_type_of_target is not good_type:
                err = "Incorrect VALUETYPE. {varName} must be {good_type} instead of {badType} ".\
                    format(varName=name,
                           good_type=good_type,
                           badType=value_type_of_target)
                raise FitIncompleteError(err)

    def _arg_dictionary(self, name,
                        dictionary,
                        possible_values,
                        required=None):
        """
        Checks and returns the value of a parameter.
        This method works similarly to the _arg method defined in the ML_BASE class.
        This is a helper for argument value checking.
        It verifies that the given dictionary contains correct values
        For example, check the given dictionary has values in
                 ['continuous,' 'nominal', 'ordinal']:
        self._arg_dictionary(
            name='variable_value_types',
            dictionary=variable_value_types,
            possible_values=['continuous,' 'nominal', 'ordinal']
            )
        Arguments:
        -------
        name : str
            The parameter name. Used for error messages.
        dictionary : dict
            A dictionary.
        possible_values : list of values
            Possible values in the dictionary.
        required : boolean, optional
            Whether the argument is a required argument. Default is False.
        Returns:
        -------
        The dictionary if values are correct.
        The TypeError Exception if not.
        """
        if dictionary is None:
            return None
        # Check whether dictionary of type dict
        self._arg(name, dictionary, dict, required)
        # Check all values in dictionary are among possible_values
        bools = [v in possible_values for v in dictionary.values()]
        if not all(bools):
            template = 'Parameter {!r} must contain value in {!r}'
            raise TypeError(template.format(
                name, possible_values))
        return dictionary

    # ----- materialization and type conversions

    @staticmethod
    def _get_select_w_type_conv(data):
        """
        Makes a SQL statement for select * from <dataframe.select_statement>.
        The select statement includes all required type conversions for ANY.
        Arguments
        --------
        data : hana_ml DataFrame
            The input dataset
        Returns
        ------
        select SQL statement : str
        """
        sql = 'SELECT '
        for i, (name, type_ori, _) in enumerate(data.dtypes()):
            if i > 0:
                sql += ', '
            conv_fct = TYPE_CONVERSION_MAP.get(type_ori, None)
            if conv_fct is None:
                # no conversion
                sql += quotename(name)
            else:
                sql += (conv_fct.format(NAME=quotename(name))
                        + ' as '
                        + quotename(name))
        sql += ' FROM ({select})'.format(select=data.select_statement)
        return sql

    def _materialize_w_type_conv(self, name, data, force=True):
        """
        Materializes a DataFrame into a table.
        Local temporary column tables only.
        includes the type conversion for ANY procedures.

        Parameters
        ----------
        name : str
            The new table name
        data: hana_ml DataFrame
            The input dataset
        force : boolean, optional
            Whether to delete any existing table with the same name
        """
        create_stmt = 'CREATE LOCAL TEMPORARY COLUMN TABLE {} AS ({})'.format(
            quotename(name), self._get_select_w_type_conv(data))
        with self.conn_context.connection.cursor() as cur:
            if force:
                try_drop(self.conn_context, name)
            execute_logged(cur, create_stmt)

    def _create_view(self,
                     view_name,
                     data,
                     order_by=None,
                     force=True):
        """
        Creates a view with the DataFrame.
        The view includes type conversions for ANY if required.

        Parameters
        ----------
        view_name : str
            The new table name
        data: hana_ml DataFrame
            The input dataset given for fit or predict
        order_by: str
            The ORDER BY clause
        force : boolean, optional
            Whether to delete any existing table with the same name
        """
        # get select statement on data with type conversions
        select_w_type_conv = self._get_select_w_type_conv(data)
        create_view_statement = (
            'CREATE VIEW {name} AS ({select}'.format(
                name=quotename(view_name),
                select=select_w_type_conv
                )
            )
        if order_by is None:
            create_view_statement += ')'
        else:
            create_view_statement += ' ORDER BY ' + order_by + ')'

        self._view_name = view_name
        if force:
            self._try_drop_view(view_name)
        with self.conn_context.connection.cursor() as cur:
            execute_logged(cur, create_view_statement)

    def _try_drop_view(self, names):
        """
        Drop the view or views.

        Parameters
        ----------
        conn : ConnectionContext
            The SAP HANA connection.
        names : str or list of str
            The name or names of the tables to drop.
        """
        if isinstance(names, _TEXT_TYPES):
            names = [names]
        with self.conn_context.connection.cursor() as cur:
            for name in names:
                sql = 'DROP VIEW {name}'.format(name=quotename(name))
                try:
                    execute_logged(cur, sql)
                except dbapi.Error:
                    pass

    @staticmethod
    def _get_new_column_name(old_col_re, old_col, new_col_re):
        r"""
        Returns the new name of a column provided in predictions.

        Parameters
        ---------
        old_col_re: str
            A regular expression like r'kts_(\d+)' to be applied on the old column
        old_col: str
            The old column name
        new_col_re: str
            Replacement string with the regular expression.

        Returns
        -------
        The new column name if old_col matches old_col_re.
        None if not

        Examples
        -------
        >>> _get_new_column_name(old_col_re=r'kts_(\d+)',
                                 old_col='kts_42',
                                 new_col_re=r'PREDICTED_\1')
        PREDICTED_44
        """
        new_col = None
        if re.search(old_col_re, old_col):
            new_col = re.sub(old_col_re, new_col_re, old_col)
        return new_col

    # --- get methods

    def get_summary(self):
        """
        Retrieves the summary table after model training.

        Returns
        -------
        The reference to the SUMMARY table : hana_ml DataFrame
        This contains execution summary of the last model training
        """
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        if not hasattr(self, 'summary_'):
            raise FitIncompleteError("Summary not initialized. Perform a fit first.")
        return self.summary_

    def get_indicators(self):
        """
        Retrieves the Indicator table after model training.

        Returns
        -------
        The reference to INDICATORS table : hana_ml DataFrame
        This table provides the performance metrics of the last model training
        """
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError("Indicators not initialized. Perform a fit first.")
        return self.indicators_


    def get_fit_operation_log(self):
        """
        Retrieves the operation log table after the model training.
        This table contains the log given by SAP HANA APL during the last fit operation.

        Returns
        -------
        The reference to OPERATION_LOG table : hana_ml DataFrame
        This table provides detailed logs of the last model training
        """
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        if not hasattr(self, 'fit_operation_log_'):
            raise FitIncompleteError("Fit operation_log not initialized. Perform a fit first.")
        return self.fit_operation_log_

    def _get_apply_out_table_spec(self, apply_in_table_name, apply_config_table): #pylint: disable=too-many-locals
        """
        Returns the table spec [(columnName, columnType)*] of the table apply-out
        (the output table of predict).

        Parameters
        ----------
        apply_in_table_name : str
            The table name of apply in dataset (String)
        apply_config_table : APLArtifactTable object
            The APPLY_CONFIG table that contains configuration for apply

        Returns
        -------
        spec = List of tuples = [(column_name: str, type: str), ...]
        """
        if getattr(self, 'model_table_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        # Header func
        func_header_table = self._create_func_header_table()

        # OPERATION_LOG,
        operation_log_table = self._create_operation_log_table(
            '#GET_TABLE_TYPE_FOR_APPLY_LOG_{}'.format(self.id))
        # SUMMARY,
        apply_out_table_type = APLArtifactTable(
            name='#APLYOUT_TABLE_TYPE_{}'.format(self.id),
            type_name=APLArtifactTable.TABLE_TYPE,
            apl_version=self._apl_version,
            )
        try:
            self._create_artifact_table(func_header_table)
            self._create_artifact_table(operation_log_table)
            self._create_artifact_table(apply_out_table_type)

            # Call procedure
            self._call_apl("APL_GET_TABLE_TYPE_FOR_APPLY",
                           func_header_table,
                           self.model_table_,
                           apply_config_table,
                           apply_in_table_name,
                           apply_out_table_type,
                           operation_log_table)
        except dbapi.Error as db_er:
            # clean up the table used in fit function
            logger.error("Fit failure, the error message: %s",
                         db_er, exc_info=True)
            self._drop_artifact_tables()
            raise

        # Read the output and translate into spec [(ColumnName, ColumnType)*]
        spec = []
        # get Dataframe mapped to table apply_out_table_type
        h_table_df = self.conn_context.table(apply_out_table_type.name)
        # sort by column no
        types = pd.DataFrame(h_table_df.collect()).sort_values(['POSITION'])
        for i in range(len(types)):
            col_name = types.NAME.iloc[i]
            kind = types.KIND.iloc[i]
            max_len = types.MAXIMUM_LENGTH.iloc[i]
            col_type = kind
            if (max_len is not None) and (not isnan(max_len)):
                col_type = col_type + '(' + str(int(max_len)) + ')'
            spec.append((col_name, col_type))
        logger.info("ApplyOut Table Spec: %s", str(spec))
        return spec

    def get_predict_operation_log(self):
        """
        Retrieves the operation log table after the model training.

        Returns
        -------
        The reference to the OPERATION_LOG table : hana_ml DataFrame
        This table provides detailed logs about the last prediction
        """
        if not hasattr(self, 'model_'):
            raise FitIncompleteError("Model not initialized. Perform a fit first.")
        if not hasattr(self, 'predict_operation_log_'):
            raise FitIncompleteError(
                "Predict operation_log not initialized. Perform predict first.")
        return self.predict_operation_log_

    def _guess_var_description(self, table_name):
        """
        Guesses a variable description.

        Parameters
        ----------
        table_name : str.
            The name of the table for which the variable description will be guessed

        Returns
        -------
        A pandas DataFrame object containing the variable description
        """
        # -- Prepare Input parameter tables
        func_header_table = self._create_func_header_table()

        # OPERATION_CONFIG
        config_table = APLArtifactTable(
            name='#GUESSVAR_CONFIG_{}'.format(self.id),
            type_name=APLArtifactTable.OPERATION_CONFIG_EXTENDED,
            apl_version=self._apl_version)
        config_table = config_table.with_data(
            [('APL/ModelType', 'regression/classification', None)]
            )

        # MODEL_TRAIN_BIN
        model_table = APLArtifactTable(
            name='#GUESSVAR_MODEL_TRAIN_BIN_{}'.format(self.id),
            type_name=APLArtifactTable.MODEL_BIN_OID,
            apl_version=self._apl_version)
        # VARIABLE_DESC (output)
        var_desc_table = APLArtifactTable(
            name='#VARIABLE_DESC_{}'.format(self.id),
            type_name=APLArtifactTable.VARIABLE_DESC_OID,
            apl_version=self._apl_version
            )
        try:
            self._create_artifact_table(func_header_table)
            self._create_artifact_table(config_table)
            self._create_artifact_table(model_table)
            self._create_artifact_table(var_desc_table)

            # --- Call procedure
            self._call_apl("APL_CREATE_MODEL",
                           func_header_table,
                           config_table,
                           table_name,
                           model_table,
                           var_desc_table)
        except dbapi.Error as db_er:
            # clean up the table used in fit function
            logger.error("Failed to guess variable description, the error message: %s",
                         db_er, exc_info=True)
            self._drop_artifact_tables()
            raise
        pd_df = self.conn_context.table(var_desc_table.name).collect()
        return pd_df

    def save_artifact(
            self,
            artifact_df,
            schema_name, table_name,
            if_exists='fail',
            new_oid=None):
        #pylint: disable=too-many-arguments
        #pylint: disable=too-many-locals
        """
        Saves an artifact, a temporary table, into a permanent table.
        The model has to be trained or fitted beforehand.

        Parameters
        ----------
        schema_name: str
            The schema name
        artifact_df : hana_ml DataFrame
            The artifact created after fit or predict methods are called
        table_name: str
            The table name
        if_exists: str. {'fail', 'replace', 'append'}, default 'fail'
            The behavior when the table already exists:
                - fail: Raises a ValueError
                - replace: Drops the table before inserting new values
                - append: Inserts new values to the existing table
        new_oid: str. Optional.
            If it is given, it will be inserted as a new OID value.
            It is useful when one like to save data into the same table.

        Examples
        --------
        >>> myModel.save_artifact(
        ...             artifactTable=myModel.indicators_,
        ...             schema_name='MySchema',
        ...             table_name='MyModel_Indicators',
        ...             if_exists='replace'
        ...             )
        """
        if artifact_df is None:
            raise FitIncompleteError(
                "Artifact is none. Call fit or predict first.")

        # Check if table exists
        with self.conn_context.connection.cursor() as cursor:
            # Try to drop table if already exists
            table_exists = False
            try:
                sql = 'select 1 from {schema_name}.{table_name} limit 1'.format(
                    schema_name=quotename(schema_name),
                    table_name=quotename(table_name))
                execute_logged(cursor, sql)
                table_exists = True
            except dbapi.Error:
                pass

            # Replace OID with new_oid
            colnames = '*'
            if new_oid is not None:
                # Replace OID by new_oid
                # colnames = 'col1, col2, ..., coln'
                colnames = functools.reduce(
                    (lambda x, y: x + ',' + y),
                    [quotename(colname) for colname in artifact_df.columns])
                # colnames = ', '.join(artifact_df.columns)
                colnames = colnames.replace(quotename('OID'),
                                            "'{}' OID".format(new_oid),
                                            1)
            if table_exists:
                if if_exists == 'fail':
                    raise FitIncompleteError(
                        "Table {schema_name}.{table_name} already exists.".format(
                            schema_name=quotename(schema_name),
                            table_name=quotename(table_name)))
                if if_exists == 'replace':
                    # drop table if already exists
                    sql = 'DROP TABLE {schema_name}.{table_name}'.format(
                        schema_name=quotename(schema_name),
                        table_name=quotename(table_name))
                    execute_logged(cursor, sql)

                    # Create table and insert data
                    # artifact_df.save(where=(schema_name, table_name),
                    #                 table_type='COLUMN')
                    sql = 'CREATE COLUMN TABLE {schema_name}.{table_name}'.format(
                        schema_name=quotename(schema_name),
                        table_name=quotename(table_name))
                    sql = sql + ' AS (SELECT {COLS} FROM ({SOURCE}))'.format(
                        SOURCE=artifact_df.select_statement,
                        COLS=colnames
                        )
                    execute_logged(cursor, sql)
                elif if_exists == 'append':
                    # append data of artifact_df into table
                    sql = 'INSERT INTO {schema_name}.{table_name}'.format(
                        schema_name=quotename(schema_name),
                        table_name=quotename(table_name))
                    sql = sql + ' (SELECT {COLS} FROM ({SOURCE}))'.format(
                        SOURCE=artifact_df.select_statement,
                        COLS=colnames
                        )
                    execute_logged(cursor, sql)
            else:
                # If table does exist
                sql = 'CREATE COLUMN TABLE {schema_name}.{table_name}'.format(
                    schema_name=quotename(schema_name),
                    table_name=quotename(table_name))
                sql = sql + ' AS (SELECT {COLS} FROM ({SOURCE}))'.format(
                    SOURCE=artifact_df.select_statement,
                    COLS=colnames
                    )
                execute_logged(cursor, sql)

        logger.info(
            'Artifact (%s) is saved into {%s}.{%s}',
            artifact_df.select_statement,
            schema_name,
            table_name)

    def save_model(self, schema_name, table_name, if_exists='fail',
                   new_oid=None):
        """
        Saves the model into a table.
        The model has to be trained beforehand.
        The model can be saved either into a new table (if_exists='replace'), or an existing table
        (if_exists='append').
        In the latter case, the user can provide an identifier value (new_oid).
        The oid must be unique.
        By default, this oid is set when the model is created in Python (model.id attribute).

        Parameters
        ----------
        schema_name: str
            The schema name
        table_name: str
            Table name
        if_exists: str. {'fail', 'replace', 'append'}, default 'fail'
            The behavior when the table already exists:
                - fail: Raises an Error
                - replace: Drops the table before inserting new values
                - append: Inserts new values to the existing table
        new_oid: str. Optional.
            If it is given, it will be inserted as a new OID value.
            It is useful when one wants to save data into the same table.

        Returns
        -------
        None
        The model is saved into a table with the following columns:
         - "OID" NVARCHAR(50),     -- Serve as ID
         - "FORMAT" NVARCHAR(50),  -- APL technical info
         - "LOB" CLOB MEMORY THRESHOLD NULL  -- binary content of the model
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError(
                "Model not initialized. Perform a fit first.")
        self.save_artifact(artifact_df=self.model_,
                           schema_name=schema_name,
                           table_name=table_name,
                           if_exists=if_exists,
                           new_oid=new_oid)

    def load_model(self, schema_name, table_name, oid=None):
        """
        Loads the model from a table.

        Parameters
        ----------
        schema_name: str
            The schema name
        table_name: str
            The table name
        oid : str. optional
            If the table contains several models,
            the oid must be given as an identifier.
            If it is not provided, the whole table is read.
        """
        # Create a temporary table
        temp_table_name = '#MODEL_TRAIN_BIN_{}'.format(self.id)
        model_table = APLArtifactTable(
            name=temp_table_name,
            type_name=APLArtifactTable.MODEL_BIN_OID,
            apl_version=self._apl_version)
        self._create_artifact_table(model_table)
        # Copy the content of the existing model
        with self.conn_context.connection.cursor() as cursor:
            sql = ('INSERT INTO {temp_table_name} '
                   + '(SELECT * FROM {schema_name}.{table_name}').format(
                       temp_table_name=quotename(temp_table_name),
                       schema_name=quotename(schema_name),
                       table_name=quotename(table_name))
            if oid is None:
                sql = sql + ')'
            else:
                sql = sql + " WHERE OID='{OID}')".format(OID=oid)
            execute_logged(cursor, sql)
        self.model_table_ = model_table
        self.model_ = self.conn_context.table(temp_table_name)

    @staticmethod
    def _check_valid(data, key, features, label):
        """
        Checks whether key, features and label are valid regarding to the given dataframe (data).
        Raises FitIncompleteError if problem.
        """
        if features is not None:
            if not features:
                raise FitIncompleteError("Features list is empty")
        # l_given = [key, *features, label] : not compatible with python 2.7
        l_given = [key, label]
        if features is not None:
            l_given.extend(features)
        # are they among column names of Dataframe or None?
        chk_list = [(c is None) or (c in data.columns) for c in l_given]
        if not all(chk_list):
            raise FitIncompleteError(
                ("Some of the values given in key/label/features parameters are unknown" +
                 " of the dataframe's columns"))

    def _fit(self,
             data,
             key=None,
             features=None,
             label=None):
        # pylint:disable=too-many-locals
        # take in all feature labels except label column
        """
        Fits the model when given training dataset and other attributes.

        Parameters
        ----------
        data : DataFrame
            The training dataset.
        key : str, optional
            The name of the ID column.
            If `key` is not provided, the input is supposed to have no ID.
            If 'key' is given, it will be set and skipped as feature column.
        features : list of str, optional
            The names of the feature columns.
            If `features` is not provided, it defaults to all non-ID,
            non-label columns.
        label : str, optional
            The name of the dependent variable. Defaults to the last column.

        """
        # -- get and check the validity of input params key, features, label
        key = self._arg('key', key, str)
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        self._check_valid(data, key, features, label)

        # -- prepare TRAIN_CONFIG artifact
        train_config_table = self._create_train_config_table()

        try:
            # ---- prepare train dataset
            # Before doing a guess variable, need to materialize the dataset.
            # We can either make a view or create a new table
            # View offers more performance
            # But view is impossible when the dataset is temporary table
            data_view_name = "#TRAIN_DATA_VIEW_{}".format(self.id)
            try:
                self._create_view(view_name=data_view_name, data=data)
            except dbapi.Error:
                # fall back if view can't be created
                # (because the original dataset is a temp table)
                self._materialize_w_type_conv(name=data_view_name, data=data)

            # ---- VARIABLE_DESCRIPTION
            var_desc_table = self._create_var_desc_table(key, label, data_view_name)

            # --- HEADER_FUNCTION
            func_header_table = self._create_func_header_table()

            # --- VARIABLE_ROLES
            var_roles_table = self._create_var_roles_table(
                data=data,
                key=key,
                label=label,
                features=features)
            # --- Prepare Output tables
            # MODEL_TRAIN_BIN
            model_table = APLArtifactTable(
                name='#MODEL_TRAIN_BIN_{}'.format(self.id),
                type_name=APLArtifactTable.MODEL_BIN_OID,
                apl_version=self._apl_version)
            # OPERATION_LOG,
            operation_log_table = self._create_operation_log_table(
                '#FIT_LOG_{}'.format(self.id))

            # SUMMARY,
            summary_table = APLArtifactTable(
                name='#SUMMARY_{}'.format(self.id),
                type_name=APLArtifactTable.SUMMARY,
                apl_version=self._apl_version)

            # INDICATORS
            indicators_table = APLArtifactTable(
                name='#INDICATORS_{}'.format(self.id),
                type_name=APLArtifactTable.INDICATORS,
                apl_version=self._apl_version)

            # Materialize artifacts prior to calling APL
            self._create_artifact_table(func_header_table)
            self._create_artifact_table(train_config_table)
            self._create_artifact_table(var_desc_table)
            self._create_artifact_table(var_roles_table)
            self._create_artifact_table(model_table)
            self._create_artifact_table(operation_log_table)
            self._create_artifact_table(summary_table)
            self._create_artifact_table(indicators_table)

            # --- Call procedure
            self._call_apl(
                "APL_CREATE_MODEL_AND_TRAIN",
                func_header_table,
                train_config_table,
                var_desc_table,
                var_roles_table,
                data_view_name,
                model_table,
                operation_log_table,
                summary_table,
                indicators_table
                )
        except dbapi.Error as db_er:
            # do stuff with the error, and also raise to the caller,
            # clean up the table used in fit function
            logger.error("Fit failure, the error message: %s",
                         db_er, exc_info=True)
            self._drop_artifact_tables()
            raise
        finally:
            # clean created view
            self._try_drop_view(data_view_name)

        # --- Save returned artifacts
        # --- Table containing the model
        self.model_table_ = model_table #pylint:disable=attribute-defined-outside-init
        # Convert the model Table into a Hana DataFrame
        self.model_ = self.conn_context.table(model_table.name) #pylint:disable=attribute-defined-outside-init
        # It is useless to keep the model as Dataframe, just for compatibility
        # --- capture the other output artifacts as hana Dataframes
        self.summary_ = self.conn_context.table(summary_table.name) #pylint:disable=attribute-defined-outside-init
        self.indicators_ = self.conn_context.table(indicators_table.name) #pylint:disable=attribute-defined-outside-init
        self.fit_operation_log_ = self.conn_context.table( #pylint:disable=attribute-defined-outside-init
            operation_log_table.name)
        self.var_desc_ = self.conn_context.table(var_desc_table.name) #pylint:disable=attribute-defined-outside-init

    def _predict(self, data, apply_config_data_df=None):
        """
        Predicts values based on the fitted model.

        Parameters
        ----------
        data : hana-ml DataFrame
            The dataset to be used for prediction
        apply_config_data_df: pandas DataFrame, mandatory
            APL Apply function parameters.
            It must be provided by the call.
            It defined the content expected in the output.

        Returns:
        Prediction output: hana_ml DataFrame
        """
        # -- get and check args
        if getattr(self, 'model_table_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        if apply_config_data_df is None:
            raise FitIncompleteError("Apply config undefined")
        # Converts to List of tuples
        apply_config_data_tuples = [tuple(row)
                                    for _, row in
                                    apply_config_data_df.iterrows()]

        try:
            # ApplyIn dataset
            apply_in_tbl = "#APPLYIN_TBL_{}".format(self.id)
            # We can either make a view or create a new table
            # View offers more performance
            # But view is impossible when the dataset is temporary table
            try:
                self._create_view(view_name=apply_in_tbl, data=data)
            except dbapi.Error:
                self._materialize_w_type_conv(name=apply_in_tbl, data=data)

            # -- Prepare Input parameter tables
            apply_config_table = APLArtifactTable(
                name='#APPLY_CONFIG_{}'.format(self.id),
                type_name=APLArtifactTable.OPERATION_CONFIG_EXTENDED,
                apl_version=self._apl_version,
                data=apply_config_data_tuples)
            # Need to materialize table prior to request applyout table definition
            self._create_artifact_table(apply_config_table)

            # call APL_GET_TABLE_TYPE_FOR_APPLY to get applyout table definition
            # Warning: this APL call will overwrite artifact table
            # such as FUNCT_HEADER, OPERATION_LOG_
            # So, do not create them beforehand
            applyout_tbl_specs = self._get_apply_out_table_spec(
                apply_in_tbl, apply_config_table)
            # Create Apply-out table
            applyout_tbl_name = '#APPLYOUT_%s' % (self.id)  # name of table
            applyout_table = Table(applyout_tbl_name, applyout_tbl_specs)

            # Header func
            func_header_table = self._create_func_header_table()

            # OPERATION_LOG
            apply_log_table = self._create_operation_log_table(
                name='#APPL_LOG_{}'.format(self.id))  # name of table)

            # Materialize artifacts prior to calling APL
            self._create_artifact_table(func_header_table)
            self._create(applyout_table)
            self._create_artifact_table(apply_log_table)

            # --- Call procedure
            self._call_apl(
                "APL_APPLY_MODEL",
                func_header_table,
                self.model_table_,
                apply_config_table,
                apply_in_tbl,
                applyout_table,
                apply_log_table
                )
            self.applyout_ = self.conn_context.table(applyout_table.name) #pylint:disable=attribute-defined-outside-init
            self.applyout_table_ = applyout_table #pylint:disable=attribute-defined-outside-init
            self.predict_operation_log_ = self.conn_context.table( #pylint:disable=attribute-defined-outside-init
                apply_log_table.name)
            return self.applyout_
        except dbapi.Error as db_er:
            # do stuff with the error, and also raise to the caller,
            # clean up the table used in fit function
            logger.error("Fit failure, the error message: %s", db_er,
                         exc_info=True)
            self._drop_artifact_tables()
            raise
        finally:
            self._try_drop_view(apply_in_tbl)
