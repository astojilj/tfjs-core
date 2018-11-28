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

import * as util from '../../util';
import {GPGPUProgram} from './gpgpu_math';
import * as shader_util from './shader_compiler_util';

export class ReshapePackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;

  constructor(outputShape: [number, number, number], inputShape: [
    number, number, number
  ]) {
    this.outputShape = outputShape;

    let mainLoop = ``;
    const componentSetup = [
      'thisRC = rc;',
      `thisRC.z += 1;
       if(thisRC.z < cols) {
      `,
      `}
       thisRC = (rc.y + 1 == rows) ? ivec3(rc.x + 1, 0, rc.z)
                                   : ivec3(rc.x, rc.y + 1, rc.z);
       if(thisRC.x < batches && thisRC.y < rows) {`,
      `  thisRC.z += 1;
         if(thisRC.z < cols) {`
    ];
    for (let i = 0; i < 4; i++) {
      mainLoop += `
        ${componentSetup[i]}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          result[${i}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRC.yz);
      `;
    }
    mainLoop += `
         }
       }`;

    this.userCode = `
      ${getReshapedInputCoords(inputShape)}
      ${getFlatIndex(outputShape)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int batches = ${outputShape[0]};
        int rows = ${outputShape[1]};
        int cols = ${outputShape[2]};

        ${mainLoop}

        setOutput(result);
      }
    `;
  }
}

function getFlatIndex(shape: [number, number, number]): string {
  const dotCoordsWithStrides = shader_util.dotify(
      ['coords.x', 'coords.y', 'coords.z'],
      util.computeStrides(shape).map(d => d.toString()).concat(['1.']));

  return `
    int getFlatIndex(ivec3 coords) {
      return round(${dotCoordsWithStrides});
    }
  `;
}

function getReshapedInputCoords(shape: [number, number, number]): string {
  const coordsFromIndexSnippet =
      shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);

  return `
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
}