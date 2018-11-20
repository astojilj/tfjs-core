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

import {getChannels} from '../packing_util';

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class PackProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;

  constructor(
      outputShape:
          number[]) {  // TODO(https://github.com/tensorflow/tfjs/issues/893):
                       // Only input / output 3D tensors.
    this.outputShape = outputShape;
    const rank = outputShape.length;

    const channels = getChannels('rc', rank);
    const dtype = getCoordsDataType(rank);
    const outOfBoundsCondition =
        getOutOfBoundsCondition(rank, outputShape, channels);
    const setup = getSetup(outputShape, channels, dtype);
    const output = getOutput(outputShape, channels);

    this.userCode = `
      void main() {
        ${dtype} rc = getOutputCoords();

        if(${outOfBoundsCondition}) {
          gl_FragColor = vec4(0);
        } else {
          ${setup}

          setOutput(vec4(${output}));
        }
      }
    `;
  }
}

function getSourceCoordsArr(rank: number, dims: string[]): string[] {
  const dimsp1 = getChannels('rcp1', rank);
  const coords = [];
  for (let i = 0; i < 4; i++) {
    let coord = (i > 1 ? dimsp1[dims.length - 2] : dims[dims.length - 2]) + ', '
        + ((i % 2 === 0) ? dims[dims.length - 1] : dimsp1[dims.length - 1]);
    for (let d = rank - 3; d >= 0; d--) {
      coord = (i > 1 ? dimsp1[d] : dims[d]) + ', ' + coord;
    }
    coords.push(coord);
  }
  return coords;
}

function getOutOfBoundsCondition(
    rank: number, shape: number[], dims: string[]): string {
  if (rank === 1) {
    return `rc > ${shape[0]}`;
  }

  let cond = '';
  for (let i = 0; i < rank; i++) {
    cond += `${dims[i]} >= ${shape[i]}`;
    if (i < rank - 1) {
      cond += '||';
    }
  }

  return cond;
}

function getSetup(shape: number[], dims: string[], dtype: string): string {
  const rank = shape.length;
  if (rank === 1) {
    return '';
  }
  const rows = shape[shape.length - 2];
  const columns = shape[shape.length - 1];

  let src = `
      ${dtype} rcp1 = rc + ` + (rank === 2 ? 'ivec2(1, 1)' :
                                rank === 3 ? 'ivec3(0, 1, 1)' :
                                             'ivec4(0, 0, 1, 1)') + ';';
  // edge[0] and edge[1] refer to components below (row + 1) and to the right
  // (column + 1) being out of bounds, respectivelly. 
  src += `
      bvec2 edge = greaterThanEqual(` +
          (rank === 2 ? 'rcp1' : rank === 3 ? 'rcp1.gb' : 'rcp1.ba') +
              `, ivec2(${rows}, ${columns}));`;
  // Handle when row + 1 carries over to the next batch.
  for (let i = 3; i <= rank; i++) {
    src +=`
        if (rcp1[${shape.length - i + 1}] == ${shape[shape.length - i + 1]}) {
          rcp1[${shape.length - i + 1}] = 0;
          rcp1[${shape.length - i}]++;
          edge[0] = rcp1[${shape.length - i}] >= ${shape[shape.length - i]};
        }`;
  }
  return src;
}

function getOutput(shape: number[], dims: string[]): string {
  const rank = shape.length;
  const sourceCoords = getSourceCoordsArr(rank, dims);
  if (rank === 1) {
    return `getA(rc),
            rc + 1 >= ${shape[0]} ? 0. : getA(rc + 1),
            0, 0`;
  }
  return `getA(${sourceCoords[0]}),
          edge[1] ? 0. : getA(${sourceCoords[1]}),
          edge[0] ? 0. : getA(${sourceCoords[2]}),
          any(edge) ? 0. : getA(${sourceCoords[3]})`;
}