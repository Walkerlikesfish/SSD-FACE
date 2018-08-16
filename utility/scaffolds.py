# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

def get_init_fn_for_scaffold(model_dir, checkpoint_path, model_scope, checkpoint_model_scope, checkpoint_exclude_scopes, ignore_missing_vars, name_remap=None):
    """scaffold init function
    The logic is:
    1) Extract the TRAINABLE_VARIABLES in current map;
    2) Load ther checkpoint from pointed path; Meanwhile replace the variable name (scope) `current graph name scope` to the `ckpt namescope` 
    to facilitate the weight loding -> `model_scope`, `checkpoint_model_scope`, `name_remap`
    3) 
    :param model_dir: str: the model checkpoint saving dir
    :param checkpoint_path: str:  pre-load checkpoint path
    :param model_scope: str: 
    :param checkpoint_model_scope: 
    :param checkpoint_exclude_scopes: str,seperated with ',': exclude namescopes in checkpoints
    :param ignore_missing_vars: 
    :param name_remap:  dict: 
    :return:
    """
    if tf.train.latest_checkpoint(model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s.' % model_dir)
        return None

    exclusion_scopes = []
    if checkpoint_exclude_scopes:
        exclusion_scopes = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]

    # 1) collect the trainable varaibles from the loaded graph
    #    save it to variables_to_restore
    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        excluded = False
        for exclusion in exclusion_scopes:
            if exclusion in var.op.name:#.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    #2) replace the `current_model` scope to this `checkpoint_model_scope` to facilitate the varibale weight loading process
    # at the same time remap the var names following the `remap` dict
    if checkpoint_model_scope is not None:
        if checkpoint_model_scope.strip() == '':
            variables_to_restore = {var.op.name.replace(model_scope + '/', ''): var for var in variables_to_restore}
        else:
            variables_to_restore = {var.op.name.replace(model_scope, checkpoint_model_scope.strip()): var for var in variables_to_restore}
        if name_remap is not None:
            renamed_variables_to_restore = dict()
            for var_name, var in variables_to_restore.items():
                found = False
                for k, v in name_remap.items():
                    if k in var_name:
                        renamed_variables_to_restore[var_name.replace(k, v)] = var # replace the var name in `variable_to_restore` following the remap dict
                        found = True
                        break
                if not found:
                    renamed_variables_to_restore[var_name] = var
            variables_to_restore = renamed_variables_to_restore

    # judge if checkpoint_path is a single `ckpt` file or a folder of checkpoints
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path) if tf.gfile.IsDirectory(checkpoint_path) else checkpoint_path

    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s.' % (checkpoint_path, ignore_missing_vars))

    if not variables_to_restore:
        raise ValueError('variables_to_restore cannot be empty')

    if ignore_missing_vars: # rebuild the variables_to_restore
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s.', var, checkpoint_path)
        variables_to_restore = available_vars

    if variables_to_restore:
        saver = tf.train.Saver(variables_to_restore, reshape=False)
        saver.build()
        def callback(scaffold, session):
            saver.restore(session, checkpoint_path)
        return callback
    else:
        tf.logging.warning('No Variables to restore.')
        return None
