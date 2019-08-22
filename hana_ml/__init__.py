"""
hana_ml provides Python bindings for the SAP HANA Predictive Analysis
Library and the SAP HANA Automated Predictive Library, and a DataFrame
class that represents HANA tables and queries.
"""

from .dataframe import DataFrame, ConnectionContext

__all__ = [
    'algorithms',
    'visualizers',
    'DataFrame',
    'ConnectionContext',
]

# This is duplicated from version.txt due to build system issues.
__version__ = '1.0.7'
