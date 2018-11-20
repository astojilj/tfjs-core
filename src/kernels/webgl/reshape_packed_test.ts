/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import {expectArraysClose, WEBGL_ENVS} from '../../test_util';

describeWithFlags('expensive reshape', WEBGL_ENVS, () => {
  let webglLazilyUnpackFlagSaved: boolean;

  beforeAll(() => {
    webglLazilyUnpackFlagSaved = tf.ENV.get('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
  });

  afterAll(() => {
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
  });

  const cValues =
      [46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250];
  let c: tf.Tensor;

  beforeEach(() => {
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
    const b = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);
    c = tf.matMul(a, b);
  });

  it('6d --> 1d', () => {
    const cAs6D = tf.reshape(c, [1, 1, 1, 3, 1, 5]);
    const cAs1D = tf.reshape(cAs6D, [-1, cValues.length]);
    expectArraysClose(cAs1D, cValues);
  });
  it('1d --> 2d', () => {
    const cAs1D = tf.reshape(c, [cValues.length]);
    const cAs2D = tf.reshape(cAs1D, [5, -1]);
    expectArraysClose(cAs2D, cValues);
  });
  it('2d --> 3d', () => {
    const cAs3D = tf.reshape(c, [3, 1, 5]);
    expectArraysClose(cAs3D, cValues);
  });
  it('2d --> 4d', () => {
    const cAs4D = tf.reshape(c, [5, 3, 1, 1]);
    expectArraysClose(cAs4D, cValues);
  });
  it('3d --> 4d', () => {
    const cAs3D = tf.reshape(c, [3, 1, 5]);
    const cAs4D = tf.reshape(cAs3D, [3, 5, 1, 1]);
    expectArraysClose(cAs4D, cValues);
  });
  it('4d --> 4d', () => {
    const cAs4D0 = tf.reshape(c, [1, 1, 15, 1]);
    const cAs4D1 = tf.reshape(cAs4D0, [1, 3, 5, 1]);
    expectArraysClose(cAs4D1, cValues);
  });
  it('4d --> 5d', () => {
    const cAs4D = tf.reshape(c, [3, 5, 1, 1]);
    const cAs5D = tf.reshape(cAs4D, [1, 1, 1, 5, 3]);
    expectArraysClose(cAs5D, cValues);
  });
  it('5d --> 6d', () => {
    const cAs5D = tf.reshape(c, [1, 1, 1, 5, 3]);
    const cAs6D = tf.reshape(cAs5D, [3, 5, 1, 1, 1, 1]);
    expectArraysClose(cAs6D, cValues);
  });
});

describeWithFlags('expensive reshape when texture shapes != physical shape',
                   WEBGL_ENVS, () => {
  let webglLazilyUnpackFlagSaved: boolean;

  beforeAll(() => {
    webglLazilyUnpackFlagSaved = tf.ENV.get('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
  });

  afterAll(() => {
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
  });

  it('texture shapes != physical shape 2d --> 2d', () => {
    const maxTextureSize = tf.ENV.get('WEBGL_MAX_TEXTURE_SIZE');
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 6);
    let values: number[] = new Array<number>(30).fill(0);
    values = values.map((d, i) => i + 1);
    const a = tf.tensor2d(values, [15, 2]);
    const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const c = tf.matMul(a, b);

    const result = [7, 10, 15, 22, 23, 34, 31, 46, 39, 58,
                    47, 70, 55, 82, 63, 94, 71, 106, 79, 118,
                    87, 130, 95, 142, 103, 154, 111, 166, 119, 178];

    // Setting WEBGL_MAX_TEXTURE_SIZE to 4 makes that [10, 3] tensor is packed
    // to texture of width 4 by height 3. Indices are packed as:
    /* -------------------------
       | 0 1 | 2 . | 6 7 |10 . |       // First row's four 
       | 3 4 | 5 . | 8 9 |11 . |       // pixels.
       -------------------------
       |12 13|16 . |18 19|22 . | 
       |14 15|17 . |20 21|23 . |
       -------------------------
       
       ...

    */
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 4);
    const cAs2D = tf.reshape(c, [10, 3]);
    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', maxTextureSize);
    
    expectArraysClose(cAs2D, result);
  });
});
