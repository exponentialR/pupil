"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from . import RecordingInfoFile, Version
from .recording_info_2_2 import _RecordingInfoFile_2_2
from . import recording_info_utils as utils


class _RecordingInfoFile_2_3(_RecordingInfoFile_2_2):
    @property
    def meta_version(self) -> Version:
        return Version("2.3")

    @property
    def _private_key_schema(self) -> RecordingInfoFile._KeyValueSchema:
        return {
            **super()._private_key_schema,
            # overwrite meta_version key from parent
            "meta_version": (utils.validator_version_string, lambda _: "2.3"),
        }
