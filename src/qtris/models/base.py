"""Shared `keras.Model` base for QTris policy/value architectures.

Hosts `_tokenize_bcg`. The base class owns no Variables; it assumes the
subclass `__init__` creates the submodules it reads:
    self._bcg_proj_b2b / _bcg_proj_combo / _bcg_proj_garbage / _bcg_ln
"""

import keras

from qtris.models.encoders import tokenize_bcg


class QtrisModelBase(keras.Model):
    def _tokenize_bcg(self, b2b_combo_garbage, training=False):
        return tokenize_bcg(
            b2b_combo_garbage,
            proj_b2b=self._bcg_proj_b2b,
            proj_combo=self._bcg_proj_combo,
            proj_garbage=self._bcg_proj_garbage,
            ln=self._bcg_ln,
            training=training,
        )
