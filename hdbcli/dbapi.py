import time
import datetime
import decimal
import sys
import gc
import pyhdbcli

if sys.version_info >= (3,):
    long = int
    buffer = memoryview
    unicode = str
    xrange = range

#
# globals
#
apilevel = '2.0'
threadsafety = 1
paramstyle = ('qmark', 'named')

Connection = pyhdbcli.Connection
LOB = pyhdbcli.LOB
ResultRow = pyhdbcli.ResultRow
connect = Connection

#
# cursor class
#
class Cursor(object):
    def __init__(self, connection):
        if not isinstance(connection, Connection):
            raise ProgrammingError(0, "Connection object is required to initialize Cursor object")
        self.connection = connection
        self.__cursor = connection._internal_cursor()
        self.__column_labels = None
        self.description = None
        self.__rows_fetched = 0
        self.rowcount = -1
        self.arraysize = 32
        self.maxage = None
        self.refreshts = None
        self._scrollable = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.__cursor = None
        return

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                break
            yield row

    def __str__(self):
        return '<dbapi.Cursor instance>'

    def __parsenamedquery(self, operation, parameters):
        return self.__cursor.parsenamedquery(operation, parameters)

    def __execute(self, operation, parameters = None):
        # parameters is already checked as None or Tuple type.
        ret = self.__cursor.execute(operation, parameters=parameters, scrollable=self._scrollable)
        self.__cursor.setfetchsize(self.arraysize)
        return ret

    def __callproc(self, operation, parameters = None):
        if parameters is None:
            return self.__cursor.callproc(operation)
        elif isinstance(parameters, tuple):
            return self.__cursor.callproc(operation, parameters)
        elif isinstance(parameters, list):
            return self.__cursor.callproc(operation, tuple(parameters))
        else:
            raise ProgrammingError(0,"%s is not acceptable as parameters" % str(type(parameters)))

    def __metadata(self):
        if self.__cursor.has_result_set():
            self.rowcount = -1
            self.description = self.__cursor.description()
            self.__column_labels = []
            for column in self.description:
                self.__column_labels.append(column[0])
            self.maxage = self.__cursor.maxage
            self.refreshts = self.__cursor.refreshts
        else:
            self.rowcount = self.__cursor.get_rows_affected()
            self.description = None
            self.__column_labels = None
            self.maxage = None
            self.refreshts = None

    def close(self):
        if self.__cursor.isconnected() is True:
            self.__cursor.close()

    def execute(self, operation, parameters=None, **keywords):
        if self.__cursor.isconnected() is False:
            raise ProgrammingError(0, "Connection closed")
        self.rowcount = -1 # reset
        self.__rows_fetched = 0 # reset
        self.description = None # reset
        self.maxage = None # reset
        self.refreshts = None # reset
        if not isinstance(operation, (str, unicode)):
            raise ProgrammingError(0, "First parameter must be a string")
        if parameters is None and len(keywords) == 0:
            ret = self.__execute(operation)
            self.__metadata()
            return ret
        elif parameters is None and len(keywords) > 0:
            qmark_sql, param_values = self.__parsenamedquery(operation, keywords)
            ret = self.__execute(qmark_sql, param_values)
            self.__metadata()
            return ret
        elif isinstance(parameters, dict):
            parameters.update(keywords)
            qmark_sql, param_values = self.__parsenamedquery(operation, parameters)
            ret = self.__execute(qmark_sql, param_values)
            self.__metadata()
            return ret
        elif isinstance(parameters, tuple):
            ret = self.__execute(operation, parameters)
            self.__metadata()
            return ret
        elif isinstance(parameters, list):
            ret = self.__execute(operation, tuple(parameters))
            self.__metadata()
            return ret
        elif operation.count('?') == 1:
            ret = self.__execute(operation, (parameters,))
            self.__metadata()
            return ret
        else:
            raise ProgrammingError(0,"Invalid parameters : execute(%s, %s, %s)"
                                   % (operation, str(parameters), str(keywords)))

    def executemany(self, operation, list_of_parameters = None):
        if self.__cursor.isconnected() is False:
            raise ProgrammingError(0,"Connection closed")
        self.rowcount = -1 # reset
        self.description = None # reset
        self.maxage = None #reset
        self.refreshts = None #reset
        if isinstance(operation, (str, unicode)):
            if list_of_parameters is None or len(list_of_parameters) == 0:
                return self.execute(operation)
            elif isinstance(list_of_parameters, (tuple, list)):
                # execute in batch
                try:
                    return self.__cursor.executemany_in_batch(operation, list_of_parameters)
                finally:
                    self.rowcount = self.__cursor.get_rows_affected()
            else:
                raise ProgrammingError(0,"Second parameter should be a tuple or a list of parameters")
        elif list_of_parameters is None:
            try:
                return self.__cursor.executemany(operation)
            finally:
                self.rowcount = self.__cursor.get_rows_affected()

        else:
            raise ProgrammingError(0,"Invalid parameter : Cursor.executemany(operation[s][, list of parameters])")

    def fetchone(self, uselob = False):
        if self.__cursor.isconnected() is False:
            raise ProgrammingError(0,"Connection closed")
        if not self.__cursor.has_result_set():
            raise ProgrammingError(0,"No result set")
        ret = self.__cursor.fetchone(uselob)
        if ret is not None:
            self.__rows_fetched += 1
            return ResultRow(self.__column_labels, ret)
        else:
            return None

    def fetchmany(self, size = None):
        gc_on = gc.isenabled()
        if gc_on:
            gc.disable()
        try:
            if size == None:
                size = self.arraysize
            if self.__cursor.isconnected() is False:
                raise ProgrammingError(0,"Connection closed")
            if not self.__cursor.has_result_set():
                raise ProgrammingError(0,"No result set")
            ret = self.__cursor.fetchmany(size)
            if ret is not None:
                length = len(ret)
                for i in range(length):
                    ret[i] = ResultRow(self.__column_labels, ret[i])
                self.__rows_fetched += length
                return ret
            else:
                return None
        finally:
            if gc_on:
                gc.enable()

    def fetchall(self):
        gc_on = gc.isenabled()
        if gc_on:
            gc.disable()
        try:
            if self.__cursor.isconnected() is False:
                raise ProgrammingError(0,"Connection closed")
            if not self.__cursor.has_result_set():
                raise ProgrammingError(0,"No result set")
            ret = self.__cursor.fetchall()
            if ret is not None:
                length = len(ret)
                for i in range(length):
                    ret[i] = ResultRow(self.__column_labels, ret[i])
                self.__rows_fetched += length
                self.rowcount = self.__rows_fetched
                return ret
            else:
                self.rowcount = self.__rows_fetched
                return None
        finally:
            if gc_on:
                gc.enable()

    def setfetchsize(self, value):
        self.arraysize = value
        self.__cursor.setfetchsize(value)

    def scroll(self, value, mode="relative"):
        if self._scrollable == False:
            raise ProgrammingError(0,"should be a scrollable-enabled cursor")
        ret = None;
        if( mode == "relative" ):
            ret = self.__cursor.relative(value)
        else:
            ret = self.__cursor.absolute(value)
        if ret == None :
            raise IndexError(0, "cursor position out of range")
        return ret

    def setinputsizes(self, sizes):
        pass

    def setoutputsize(self, size, column = None):
        pass

    def callproc(self, procname, parameters = (), overview = False):
        if self.__cursor.isconnected() is False:
            raise ProgrammingError(0,"Connection closed")
        self.rowcount = -1 # reset
        self.description = None # reset
        self.maxage = None #reset
        self.refreshts = None #reset
        if isinstance(parameters, tuple) :
            qmarks = ','.join(tuple('?' * len(parameters)))
            callsql = "CALL %s(%s)" % (procname, qmarks)
            if overview:
                callsql = callsql + " WITH OVERVIEW"
            callproc = "{ %s }" % callsql
            ret = self.__callproc(callproc, parameters)
            self.__metadata()
            return ret
        else:
            raise ProgrammingError(0,"Second parameter should be a tuple")

    def nextset(self):
        if self.__cursor.isconnected() is True:
            ret = self.__cursor.nextset()
            self.__metadata()
            return ret
        else:
            raise ProgrammingError(0,"Connection closed")

    def get_resultset_holdability(self):
        if self.__cursor.isconnected() is True:
            return self.__cursor.get_resultset_holdability()
        else:
            raise ProgrammingError(0,"Connection closed")

    def set_resultset_holdability(self, holdability):
        if self.__cursor.isconnected() is True:
            self.__cursor.set_resultset_holdability(holdability)
        else:
            raise ProgrammingError(0,"Connection closed")

    def description_ext(self):
        if self.description is None:
            return None
        else:
            return self.__cursor.description_ext()

    def parameter_description(self):
        return self.__cursor.parameter_description()

    def haswarning(self):
        return self.__cursor.haswarning()

    def getwarning(self):
        return self.__cursor.getwarning()

    def server_processing_time(self):
        return self.__cursor.server_processing_time()

    def server_cpu_time(self):
        return self.__cursor.server_cpu_time()

    def server_memory_usage(self):
        return self.__cursor.server_memory_usage()

    def setquerytimeout(self, timeout):
        return self.__cursor.setquerytimeout(timeout)

#
# exceptions
#
from pyhdbcli import Warning
Warning.__module__ = __name__
from pyhdbcli import Error
Error.__module__ = __name__
def __errorinit(self, *args):
    super(Error, self).__init__(*args)
    argc = len(args)
    if argc == 1:
        if isinstance(args[0], Error):
            self.errorcode = args[0].errorcode
            self.errortext = args[0].errortext
        elif isinstance(args[0], (str, unicode)):
            self.errorcode = 0
            self.errortext = args[0]
    elif argc >= 2 and isinstance(args[0], (int, long)) and isinstance(args[1], (str, unicode)):
        self.errorcode = args[0]
        self.errortext = args[1]
Error.__init__ = __errorinit
from pyhdbcli import DatabaseError
DatabaseError.__module__ = __name__
from pyhdbcli import OperationalError
OperationalError.__module__ = __name__
from pyhdbcli import ProgrammingError
ProgrammingError.__module__ = __name__
from pyhdbcli import IntegrityError
IntegrityError.__module__ = __name__
from pyhdbcli import InterfaceError
InterfaceError.__module__ = __name__
from pyhdbcli import InternalError
InternalError.__module__ = __name__
from pyhdbcli import DataError
DataError.__module__ = __name__
from pyhdbcli import NotSupportedError
NotSupportedError.__module__ = __name__


#
# input conversions
#

def Date(year, month, day):
    return datetime.date(year, month, day)

def Time(hour, minute, second, millisecond = 0):
    return datetime.time(hour, minute, second, millisecond * 1000)

def Timestamp(year, month, day, hour, minute, second, millisecond = 0):
    return datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)

def DateFromTicks(ticks):
    localtime = time.localtime(ticks)
    year = localtime[0]
    month = localtime[1]
    day = localtime[2]
    return Date(year, month, day)

def TimeFromTicks(ticks):
    localtime = time.localtime(ticks)
    hour = localtime[3]
    minute = localtime[4]
    second = localtime[5]
    return Time(hour, minute, second)

def TimestampFromTicks(ticks):
    localtime = time.localtime(ticks)
    year = localtime[0]
    month = localtime[1]
    day = localtime[2]
    hour = localtime[3]
    minute = localtime[4]
    second = localtime[5]
    return Timestamp(year, month, day, hour, minute, second)

def Binary(data):
    return buffer(data)

#
# Decimal
#
Decimal = decimal.Decimal

#
# type objects
#
class _AbstractType:
    def __init__(self, name, typeobjects):
        self.name = name
        self.typeobjects = typeobjects

    def __str__(self):
        return self.name

    def __cmp__(self, other):
        if other in self.typeobjects:
            return 0
        else:
            return -1

    def __eq__(self, other):
        return (other in self.typeobjects)

    def __hash__(self):
        return hash(self.name)

NUMBER = _AbstractType('NUMBER', (int, long, float, complex))
DATETIME = _AbstractType('DATETIME', (type(datetime.time(0)), type(datetime.date(1,1,1)), type(datetime.datetime(1,1,1))))
STRING = str
BINARY = buffer
ROWID = int
