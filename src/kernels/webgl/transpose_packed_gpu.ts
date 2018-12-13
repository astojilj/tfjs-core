/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';
import { getVecChannels } from '../packing_util';

export class TransposePackedProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  rank: number;
  usesPackedTextures = true;

  constructor(aShape: number[], newDim: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[newDim[i]];
    }
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    if (this.rank > 6) {
      throw Error(
          `Packed transpose for rank ${this.rank} is not yet supported.`);
    }
    const dtype = getCoordsDataType(this.rank);

    const outputOrder = getVecChannels('rc', this.rank);
    const switchedOrder = new Array(this.rank);
    for (let i = 0; i < newDim.length; i++) {
      switchedOrder[newDim[i]] = outputOrder[i];
    }
    const innerDims = `vec2(${switchedOrder.slice(-2).join()})`;
    const cLimit =
        `${outputOrder[this.rank - 1]} < ${outputShape[this.rank - 1]}`;
    let main = '';
    
    const componentSetup = [
      `${dtype} rc = resRC;`,
      `${outputOrder[this.rank - 1]} += 1;
       if(${cLimit}) {
      `,
      `}
       rc = resRC;
       ${outputOrder[this.rank - 2]} += 1;
       if(${outputOrder[this.rank - 2]} < ${outputShape[this.rank - 2]}) {`,
      `  ${outputOrder[this.rank - 1]} += 1;
         if(${cLimit}) {`
    ];
    for (let i = 0; i < 4; i++) {
      main += `
        ${componentSetup[i]}
          result[${i}] =
            getChannel(getA(${switchedOrder.join()}), ${innerDims});
      `;
    }
    main += `
         }
       }`;

    this.userCode = `
    void main() {
      ${dtype} resRC = getOutputCoords();
      vec4 result = vec4(0.);       
      ${main}
      setOutput(result);
    }
    `;
  }
}