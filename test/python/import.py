# RUN: python %s

#
# Part of the Dataflow Scheduler MLIR Dialects project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This is a trivial test that checks whether all the packages we expected to
# produce can be imported.

import mlir_scheduler
import mlir_scheduler.ir

import mlir_scheduler.dialects.agen
import mlir_scheduler.dialects.dataflow
import mlir_scheduler.dialects.ktdf
import mlir_scheduler.dialects.ktdf_arch
import mlir_scheduler.dialects.uniform
import mlir_scheduler.dialects.vectorchain
