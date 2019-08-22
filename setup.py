"""setup for hana-ml
See:
https://github.com/alundesap/hana_ml
"""

# To use a consistent encoding
from codecs import open # pylint: disable=W0622
from os import path

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__)) # pylint: disable=invalid-name

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read() # pylint: disable=invalid-name

def get_version():
    """ get version """
    with open('version.txt') as ver_file:
        version_str = ver_file.readline().rstrip()
    return version_str


def get_install_requires():
    """ install requires """
    reqs = []
    with open('requirements.txt') as reqs_file:
        for line in iter(lambda: reqs_file.readline().rstrip(), ''):
            reqs.append(line)
    return reqs


def get_extras_require():
    """ extras """
    with open('test-requirements.txt') as reqs_file:
        reqs = [line.rstrip() for line in reqs_file.readlines()]
    return {'test': reqs}


setup(
      # 
      name="hana-ml",
      version=get_version(),
      entry_points={"distutils.commands":
                    ["whitesource_update = plugin.WssPythonPlugin:SetupToolsCommand"]},
      packages=['hdbcli'],
      description="SAP HANA Python ML Client Library",
      long_description=long_description,
      install_requires=get_install_requires(),
      #extras_require=get_extras_require(),
      keywords="sap hana-ml python",
      author="Andrew Lunde",
      author_email="andrew.lunde@sap.com",
      license="SAP Developer",
      url="https://github.com/alundesap/hana_ml",
      package=['hana-ml'],
      package_dir={'hana_ml': 'hana_ml'},
      package_data={'hana_ml': ['hana_ml',
                               '../pyhdbcli.abi3.so'
                              ]},
      classifiers=[
          'Development Status :: 3 - Stable',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Client Library',
          'License :: OSI Approved :: SAP SE',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ]
)
