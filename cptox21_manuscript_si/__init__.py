"""
cptox21_manuscript_si
supporting notebook to the cptox21 project
"""

# Add imports here
from .cptox21 import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
