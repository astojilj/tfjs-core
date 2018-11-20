/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {Tensor} from '../../tensor';
import {DataType, DataTypeMap} from '../../types';

export enum TextureUsage {
  RENDER,
  UPLOAD,
  PIXELS,
  DOWNLOAD
}

export enum PhysicalTextureType {
  UNPACKED_FLOAT16,
  UNPACKED_FLOAT32,
  PACKED_4X1_UNSIGNED_BYTE,
  PACKED_2X2_FLOAT32,
  PACKED_2X2_FLOAT16
}

export interface TextureData {
  texture: WebGLTexture;
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensors, with a parent joining the two with the
  // complexTensors field. When this is defined, texture will be null.
  complexTensors?: {real: Tensor, imag: Tensor};

  shape: number[];
  /** [rows, columns] shape of the texture. */
  texShape: [number, number];
  dtype: DataType;
  values: DataTypeMap[DataType];
  usage: TextureUsage;
  isPacked: boolean;
}

export function getUnpackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [columns, rows];
}

export function getUnpackedArraySizeFromMatrixSize(
    matrixSize: number, channelsPerTexture: number): number {
  return matrixSize * channelsPerTexture;
}

export function getColorMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [columns * 4, rows];
}

export function getMatrixSizeFromUnpackedArraySize(
    unpackedSize: number, channelsPerTexture: number): number {
  if (unpackedSize % channelsPerTexture !== 0) {
    throw new Error(
        `unpackedSize (${unpackedSize}) must be a multiple of ` +
        `${channelsPerTexture}`);
  }
  return unpackedSize / channelsPerTexture;
}

export type TypedArray = Float32Array|Uint8Array;

export function encodeMatrixToUnpackedArray(
    matrix: TypedArray, unpackedArray: TypedArray, channelsPerTexture: number) {
  const requiredSize =
      getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture);
  if (unpackedArray.length < requiredSize) {
    throw new Error(
        `unpackedArray length (${unpackedArray.length}) must be >= ` +
        `${requiredSize}`);
  }
  let dst = 0;
  for (let src = 0; src < matrix.length; ++src) {
    unpackedArray[dst] = matrix[src];
    dst += channelsPerTexture;
  }
}

export function decodeMatrixFromUnpackedArray(
    unpackedArray: Float32Array, matrix: Float32Array,
    channelsPerTexture: number) {
  const requiredSize = getMatrixSizeFromUnpackedArraySize(
      unpackedArray.length, channelsPerTexture);
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }
  let dst = 0;
  for (let src = 0; src < unpackedArray.length; src += channelsPerTexture) {
    matrix[dst++] = unpackedArray[src];
  }
}

export function decodeMatrixFromUnpackedColorRGBAArray(
    unpackedArray: Float32Array, matrix: Float32Array, channels: number) {
  const requiredSize = unpackedArray.length * channels / 4;
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }
  let dst = 0;
  for (let src = 0; src < unpackedArray.length; src += 4) {
    for (let c = 0; c < channels; c++) {
      matrix[dst++] = unpackedArray[src + c];
    }
  }
}

export function getPackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [Math.ceil(columns / 2), Math.ceil(rows / 2)];
}

export function getPackedRGBAArraySizeFromMatrixShape(
    rows: number, columns: number): number {
  const [w, h] = getPackedMatrixTextureShapeWidthHeight(rows, columns);
  return w * h * 4;
}

/*
This is how encodeMatrixToPackedRGBA encodes a tensor with shape = [2, 3, 5]
(indices are [batch, row, col]).

000|001   002|003   004|xxx
-------   -------   -------
010|011   012|013   014|xxx

020|021   022|023   024|xxx
-------   -------   -------
100|101   102|103   104|xxx

110|111   112|113   114|xxx
-------   -------   -------
120|121   122|123   124|xxx

Single texels contain values from adjacent rows and columns. For the last row of
the batch, adjacent is the first row of the next batch.
*/

export function encodeMatrixToPackedRGBA(
    matrix: Float32Array, batches: number, rows: number, columns: number,
    packedRGBA: Float32Array) {
  const requiredSize = getPackedRGBAArraySizeFromMatrixShape(rows, columns);
  if (packedRGBA.length < requiredSize) {
    throw new Error(`packedRGBA length (${packedRGBA.length}) must be >=
        ${requiredSize}`);
  }
  
  const srcHeightInRows = batches * rows;
  const srcHeightInFullBlocks = Math.floor(srcHeightInRows / 2);
  const widthInFullBlocks = Math.floor(columns / 2);
  const oddWidth = (columns % 2) === 1;
  const oddHeight = (srcHeightInRows % 2) === 1;
  let srcRow1 = 0;
  let srcRow2 = columns;
  let dst = 0;
  for (let j = 0; j < srcHeightInFullBlocks; j++) {
    for (let i = 0; i < widthInFullBlocks; i++) {
      packedRGBA[dst++] = matrix[srcRow1++];
      packedRGBA[dst++] = matrix[srcRow1++];
      packedRGBA[dst++] = matrix[srcRow2++];
      packedRGBA[dst++] = matrix[srcRow2++];
    }
    if (oddWidth) {
      packedRGBA[dst++] = matrix[srcRow1++];
      packedRGBA[dst++] = 0;
      packedRGBA[dst++] = matrix[srcRow2++];
      packedRGBA[dst++] = 0;      
    }
    srcRow1 = srcRow2;
    srcRow2 += columns;
  }
  if (oddHeight) {
    for (let i = 0; i < widthInFullBlocks; i++) {
      packedRGBA[dst++] = matrix[srcRow1++];
      packedRGBA[dst++] = matrix[srcRow1++];
      packedRGBA[dst++] = 0;
      packedRGBA[dst++] = 0;
    }
    if (oddWidth) {
      packedRGBA[dst++] = matrix[srcRow1++];
      packedRGBA[dst++] = 0;
      packedRGBA[dst++] = 0;
      packedRGBA[dst++] = 0;      
    }
  }
  return packedRGBA;
}

export function decodeMatrixFromPackedRGBA(
    packedRGBA: Float32Array, batches: number, rows: number, columns: number,
    matrix: Float32Array): Float32Array {
  const requiredSize = rows * columns;
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }

  const srcWidthInFullBlocks = Math.floor(columns / 2);
  const dstHeightInRows = batches * rows; 
  const srcHeightInFullBlocks = Math.floor(dstHeightInRows / 2);
  const oddWidth = (columns % 2) === 1;
  const oddHeight = (dstHeightInRows % 2) === 1;

  let dstRow1 = 0;
  let dstRow2 = columns;
  let src = 0;
  for (let j = 0; j < srcHeightInFullBlocks; j++) {
    // loop over full 2x2 blocks
    for (let i = 0; i < srcWidthInFullBlocks; i++) {
      matrix[dstRow1++] = packedRGBA[src++];
      matrix[dstRow1++] = packedRGBA[src++];
      matrix[dstRow2++] = packedRGBA[src++];
      matrix[dstRow2++] = packedRGBA[src++];
    }    
    if (oddWidth) {
      matrix[dstRow1++] = packedRGBA[src++];
      src++;
      matrix[dstRow2++] = packedRGBA[src++];
      src++;
    }
    dstRow1 += columns;
    dstRow2 = dstRow1 + columns;
  }
  // loop across final row
  if (oddHeight) {
    for (let i = 0; i < srcWidthInFullBlocks; i++) {
      matrix[dstRow1++] = packedRGBA[src++];
      matrix[dstRow1++] = packedRGBA[src++];
      src += 2;
    }    
    if (oddHeight) {
      matrix[dstRow1++] = packedRGBA[src++];
    }
  }
  return matrix;
}
