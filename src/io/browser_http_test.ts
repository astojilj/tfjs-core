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

/**
 * Unit tests for browser_http.ts.
 */

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {BROWSER_ENVS, CHROME_ENVS, NODE_ENVS} from '../test_util';

import {BrowserHTTPRequest, httpRequestRouter, parseUrl} from './browser_http';

// Test data.
const modelTopology1: {} = {
  'class_name': 'Sequential',
  'keras_version': '2.1.4',
  'config': [{
    'class_name': 'Dense',
    'config': {
      'kernel_initializer': {
        'class_name': 'VarianceScaling',
        'config': {
          'distribution': 'uniform',
          'scale': 1.0,
          'seed': null,
          'mode': 'fan_avg'
        }
      },
      'name': 'dense',
      'kernel_constraint': null,
      'bias_regularizer': null,
      'bias_constraint': null,
      'dtype': 'float32',
      'activation': 'linear',
      'trainable': true,
      'kernel_regularizer': null,
      'bias_initializer': {'class_name': 'Zeros', 'config': {}},
      'units': 1,
      'batch_input_shape': [null, 3],
      'use_bias': true,
      'activity_regularizer': null
    }
  }],
  'backend': 'tensorflow'
};

describeWithFlags('browserHTTPRequest-load fetch-polyfill', NODE_ENVS, () => {
  let requestInits: RequestInit[];

  // simulate a fetch polyfill, this needs to be non-null for spyOn to work
  beforeEach(() => {
    // tslint:disable-next-line:no-any
    (global as any).fetch = () => {};
    requestInits = [];
  });

  afterAll(() => {
    // tslint:disable-next-line:no-any
    delete (global as any).fetch;
  });
  type TypedArrays = Float32Array|Int32Array|Uint8Array|Uint16Array;

  const fakeResponse = (body: string|TypedArrays|ArrayBuffer) => ({
    ok: true,
    json() {
      return Promise.resolve(JSON.parse(body as string));
    },
    arrayBuffer() {
      const buf: ArrayBuffer = (body as TypedArrays).buffer ?
          (body as TypedArrays).buffer :
          body as ArrayBuffer;
      return Promise.resolve(buf);
    }
  });

  const setupFakeWeightFiles = (fileBufferMap: {
    [filename: string]: string|Float32Array|Int32Array|ArrayBuffer|Uint8Array|
    Uint16Array
  }) => {
    // tslint:disable-next-line:no-any
    spyOn(global as any, 'fetch')
        .and.callFake((path: string, init: RequestInit) => {
          requestInits.push(init);
          return fakeResponse(fileBufferMap[path]);
        });
  };

  it('1 group, 2 weights, 1 path', (done: DoneFn) => {
    const weightManifest1: tf.io.WeightsManifestConfig = [{
      paths: ['weightfile0'],
      weights: [
        {
          name: 'dense/kernel',
          shape: [3, 1],
          dtype: 'float32',
        },
        {
          name: 'dense/bias',
          shape: [2],
          dtype: 'float32',
        }
      ]
    }];
    const floatData = new Float32Array([1, 3, 3, 7, 4]);
    setupFakeWeightFiles({
      './model.json': JSON.stringify(
          {modelTopology: modelTopology1, weightsManifest: weightManifest1}),
      './weightfile0': floatData,
    });

    const handler = tf.io.browserHTTPRequest('./model.json');
    handler.load()
        .then(modelArtifacts => {
          expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
          expect(modelArtifacts.weightSpecs)
              .toEqual(weightManifest1[0].weights);
          expect(new Float32Array(modelArtifacts.weightData))
              .toEqual(floatData);
          expect(requestInits).toEqual([{}, {}]);
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('throw exception if no fetch polyfill', () => {
    // tslint:disable-next-line:no-any
    delete (global as any).fetch;
    try {
      tf.io.browserHTTPRequest('./model.json');
    } catch (err) {
      expect(err.message)
          .toMatch(
              /not supported outside the web browser without a fetch polyfill/);
    }
  });
});

// Turned off for other browsers due to:
// https://github.com/tensorflow/tfjs/issues/426
describeWithFlags('browserHTTPRequest-save', CHROME_ENVS, () => {
  // Test data.
  const weightSpecs1: tf.io.WeightsManifestEntry[] = [
    {
      name: 'dense/kernel',
      shape: [3, 1],
      dtype: 'float32',
    },
    {
      name: 'dense/bias',
      shape: [1],
      dtype: 'float32',
    }
  ];
  const weightData1 = new ArrayBuffer(16);
  const artifacts1: tf.io.ModelArtifacts = {
    modelTopology: modelTopology1,
    weightSpecs: weightSpecs1,
    weightData: weightData1,
  };

  let requestInits: RequestInit[] = [];

  beforeEach(() => {
    requestInits = [];
    spyOn(window, 'fetch').and.callFake((path: string, init: RequestInit) => {
      if (path === 'model-upload-test' || path === 'http://model-upload-test') {
        requestInits.push(init);
        return new Response(null, {status: 200});
      } else {
        return new Response(null, {status: 404});
      }
    });
  });

  it('Save topology and weights, default POST method', (done: DoneFn) => {
    const testStartDate = new Date();
    const handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
    handler.save(artifacts1)
        .then(saveResult => {
          expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
              .toBeGreaterThanOrEqual(testStartDate.getTime());
          // Note: The following two assertions work only because there is no
          //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
          expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
              .toEqual(JSON.stringify(modelTopology1).length);
          expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
              .toEqual(JSON.stringify(weightSpecs1).length);
          expect(saveResult.modelArtifactsInfo.weightDataBytes)
              .toEqual(weightData1.byteLength);

          expect(requestInits.length).toEqual(1);
          const init = requestInits[0];
          expect(init.method).toEqual('POST');
          const body = init.body as FormData;
          const jsonFile = body.get('model.json') as File;
          const jsonFileReader = new FileReader();
          jsonFileReader.onload = (event: Event) => {
            // tslint:disable-next-line:no-any
            const modelJSON = JSON.parse((event.target as any).result);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            expect(modelJSON.weightsManifest.length).toEqual(1);
            expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);

            const weightsFile = body.get('model.weights.bin') as File;
            const weightsFileReader = new FileReader();
            weightsFileReader.onload = (event: Event) => {
              // tslint:disable-next-line:no-any
              const weightData = (event.target as any).result as ArrayBuffer;
              expect(new Uint8Array(weightData))
                  .toEqual(new Uint8Array(weightData1));
              done();
            };
            weightsFileReader.onerror = (error: FileReaderProgressEvent) => {
              done.fail(error.target.error.message);
            };
            weightsFileReader.readAsArrayBuffer(weightsFile);
          };
          jsonFileReader.onerror = (error: FileReaderProgressEvent) => {
            done.fail(error.target.error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Save topology only, default POST method', (done: DoneFn) => {
    const testStartDate = new Date();
    const handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
    const topologyOnlyArtifacts = {modelTopology: modelTopology1};
    handler.save(topologyOnlyArtifacts)
        .then(saveResult => {
          expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
              .toBeGreaterThanOrEqual(testStartDate.getTime());
          // Note: The following two assertions work only because there is no
          //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
          expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
              .toEqual(JSON.stringify(modelTopology1).length);
          expect(saveResult.modelArtifactsInfo.weightSpecsBytes).toEqual(0);
          expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(0);

          expect(requestInits.length).toEqual(1);
          const init = requestInits[0];
          expect(init.method).toEqual('POST');
          const body = init.body as FormData;
          const jsonFile = body.get('model.json') as File;
          const jsonFileReader = new FileReader();
          jsonFileReader.onload = (event: Event) => {
            // tslint:disable-next-line:no-any
            const modelJSON = JSON.parse((event.target as any).result);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            // No weights should have been sent to the server.
            expect(body.get('model.weights.bin')).toEqual(null);
            done();
          };
          jsonFileReader.onerror = (error: FileReaderProgressEvent) => {
            done.fail(error.target.error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('Save topology and weights, PUT method, extra headers', (done: DoneFn) => {
    const testStartDate = new Date();
    const handler = tf.io.browserHTTPRequest('model-upload-test', {
      method: 'PUT',
      headers: {
        'header_key_1': 'header_value_1',
        'header_key_2': 'header_value_2'
      }
    });
    handler.save(artifacts1)
        .then(saveResult => {
          expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
              .toBeGreaterThanOrEqual(testStartDate.getTime());
          // Note: The following two assertions work only because there is no
          //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
          expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
              .toEqual(JSON.stringify(modelTopology1).length);
          expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
              .toEqual(JSON.stringify(weightSpecs1).length);
          expect(saveResult.modelArtifactsInfo.weightDataBytes)
              .toEqual(weightData1.byteLength);

          expect(requestInits.length).toEqual(1);
          const init = requestInits[0];
          expect(init.method).toEqual('PUT');

          // Check headers.
          expect(init.headers).toEqual({
            'header_key_1': 'header_value_1',
            'header_key_2': 'header_value_2'
          });

          const body = init.body as FormData;
          const jsonFile = body.get('model.json') as File;
          const jsonFileReader = new FileReader();
          jsonFileReader.onload = (event: Event) => {
            // tslint:disable-next-line:no-any
            const modelJSON = JSON.parse((event.target as any).result);
            expect(modelJSON.modelTopology).toEqual(modelTopology1);
            expect(modelJSON.weightsManifest.length).toEqual(1);
            expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);

            const weightsFile = body.get('model.weights.bin') as File;
            const weightsFileReader = new FileReader();
            weightsFileReader.onload = (event: Event) => {
              // tslint:disable-next-line:no-any
              const weightData = (event.target as any).result as ArrayBuffer;
              expect(new Uint8Array(weightData))
                  .toEqual(new Uint8Array(weightData1));
              done();
            };
            weightsFileReader.onerror = (error: FileReaderProgressEvent) => {
              done.fail(error.target.error.message);
            };
            weightsFileReader.readAsArrayBuffer(weightsFile);
          };
          jsonFileReader.onerror = (error: FileReaderProgressEvent) => {
            done.fail(error.target.error.message);
          };
          jsonFileReader.readAsText(jsonFile);
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('404 response causes Error', (done: DoneFn) => {
    const handler = tf.io.getSaveHandlers('http://invalid/path')[0];
    handler.save(artifacts1)
        .then(saveResult => {
          done.fail(
              'Calling browserHTTPRequest at invalid URL succeeded ' +
              'unexpectedly');
        })
        .catch(err => {
          done();
        });
  });

  it('getLoadHandlers with one URL string', () => {
    const handlers = tf.io.getLoadHandlers('http://foo/model.json');
    expect(handlers.length).toEqual(1);
    expect(handlers[0] instanceof BrowserHTTPRequest).toEqual(true);
  });

  it('getLoadHandlers with two URL strings', () => {
    const handlers = tf.io.getLoadHandlers(
        ['https://foo/graph.pb', 'https://foo/weights_manifest.json']);
    expect(handlers.length).toEqual(1);
    expect(handlers[0] instanceof BrowserHTTPRequest).toEqual(true);
  });

  it('Existing body leads to Error', () => {
    expect(() => tf.io.browserHTTPRequest('model-upload-test', {
      body: 'existing body'
    })).toThrowError(/requestInit is expected to have no pre-existing body/);
  });

  it('Empty, null or undefined URL paths lead to Error', () => {
    expect(() => tf.io.browserHTTPRequest(null))
        .toThrowError(/must not be null, undefined or empty/);
    expect(() => tf.io.browserHTTPRequest(undefined))
        .toThrowError(/must not be null, undefined or empty/);
    expect(() => tf.io.browserHTTPRequest(''))
        .toThrowError(/must not be null, undefined or empty/);
  });

  it('router', () => {
    expect(httpRequestRouter('http://bar/foo') instanceof BrowserHTTPRequest)
        .toEqual(true);
    expect(
        httpRequestRouter('https://localhost:5000/upload') instanceof
        BrowserHTTPRequest)
        .toEqual(true);
    expect(httpRequestRouter('localhost://foo')).toBeNull();
    expect(httpRequestRouter('foo:5000/bar')).toBeNull();
  });
});

describeWithFlags('parseUrl', BROWSER_ENVS, () => {
  it('should parse url with no suffix', () => {
    const url = 'http://google.com/file';
    const [prefix, suffix] = parseUrl(url);
    expect(prefix).toEqual('http://google.com/');
    expect(suffix).toEqual('');
  });
  it('should parse url with suffix', () => {
    const url = 'http://google.com/file?param=1';
    const [prefix, suffix] = parseUrl(url);
    expect(prefix).toEqual('http://google.com/');
    expect(suffix).toEqual('?param=1');
  });
  it('should parse url with multiple serach params', () => {
    const url = 'http://google.com/a?x=1/file?param=1';
    const [prefix, suffix] = parseUrl(url);
    expect(prefix).toEqual('http://google.com/a?x=1/');
    expect(suffix).toEqual('?param=1');
  });
});

describeWithFlags('browserHTTPRequest-load', BROWSER_ENVS, () => {

  describe('JSON model', () => {
    let requestInits: RequestInit[];

    const setupFakeWeightFiles = (fileBufferMap: {
      [filename: string]: string|Float32Array|Int32Array|ArrayBuffer|Uint8Array|
      Uint16Array
    }) => {
      spyOn(window, 'fetch').and.callFake((path: string, init: RequestInit) => {
        requestInits.push(init);
        return new Response(fileBufferMap[path]);
      });
    };

    beforeEach(() => {
      requestInits = [];
    });

    it('1 group, 2 weights, 1 path', (done: DoneFn) => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);
      setupFakeWeightFiles({
        './model.json': JSON.stringify(
            {modelTopology: modelTopology1, weightsManifest: weightManifest1}),
        './weightfile0': floatData,
      });

      const handler = tf.io.browserHTTPRequest('./model.json');
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(floatData);
            expect(requestInits).toEqual([{}, {}]);
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('1 group, 2 weights, 1 path, with requestInit', (done: DoneFn) => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);
      setupFakeWeightFiles({
        './model.json': JSON.stringify(
            {modelTopology: modelTopology1, weightsManifest: weightManifest1}),
        './weightfile0': floatData,
      });

      const handler = tf.io.browserHTTPRequest(
          './model.json', {headers: {'header_key_1': 'header_value_1'}});
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(floatData);
            expect(requestInits).toEqual([
              {headers: {'header_key_1': 'header_value_1'}},
              {headers: {'header_key_1': 'header_value_1'}}
            ]);
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('1 group, 2 weight, 2 paths', (done: DoneFn) => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0', 'weightfile1'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData1 = new Float32Array([1, 3, 3]);
      const floatData2 = new Float32Array([7, 4]);
      setupFakeWeightFiles({
        './model.json': JSON.stringify(
            {modelTopology: modelTopology1, weightsManifest: weightManifest1}),
        './weightfile0': floatData1,
        './weightfile1': floatData2,
      });

      const handler = tf.io.browserHTTPRequest('./model.json');
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(new Float32Array([1, 3, 3, 7, 4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('2 groups, 2 weight, 2 paths', (done: DoneFn) => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }],
        }
      ];
      const floatData1 = new Float32Array([1, 3, 3]);
      const floatData2 = new Float32Array([7, 4]);
      setupFakeWeightFiles({
        './model.json':
            JSON.stringify({modelTopology: modelTopology1, weightsManifest}),
        './weightfile0': floatData1,
        './weightfile1': floatData2,
      });

      const handler = tf.io.browserHTTPRequest('./model.json');
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(
                    weightsManifest[1].weights));
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(new Float32Array([1, 3, 3, 7, 4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', (done: DoneFn) => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'fooWeight',
            shape: [3, 1],
            dtype: 'int32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'barWeight',
            shape: [2],
            dtype: 'bool',
          }],
        }
      ];
      const floatData1 = new Int32Array([1, 3, 3]);
      const floatData2 = new Uint8Array([7, 4]);
      setupFakeWeightFiles({
        'path1/model.json':
            JSON.stringify({modelTopology: modelTopology1, weightsManifest}),
        'path1/weightfile0': floatData1,
        'path1/weightfile1': floatData2,
      });

      const handler = tf.io.browserHTTPRequest('path1/model.json');
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(
                    weightsManifest[1].weights));
            expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                .toEqual(new Int32Array([1, 3, 3]));
            expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                .toEqual(new Uint8Array([7, 4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('topology only', (done: DoneFn) => {
      setupFakeWeightFiles({
        './model.json': JSON.stringify({modelTopology: modelTopology1}),
      });

      const handler = tf.io.browserHTTPRequest('./model.json');
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs).toBeUndefined();
            expect(modelArtifacts.weightData).toBeUndefined();
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('weights only', (done: DoneFn) => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'fooWeight',
            shape: [3, 1],
            dtype: 'int32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'barWeight',
            shape: [2],
            dtype: 'float32',
          }],
        }
      ];
      const floatData1 = new Int32Array([1, 3, 3]);
      const floatData2 = new Float32Array([-7, -4]);
      setupFakeWeightFiles({
        'path1/model.json': JSON.stringify({weightsManifest}),
        'path1/weightfile0': floatData1,
        'path1/weightfile1': floatData2,
      });

      const handler = tf.io.browserHTTPRequest('path1/model.json');
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toBeUndefined();
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(
                    weightsManifest[1].weights));
            expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                .toEqual(new Int32Array([1, 3, 3]));
            expect(new Float32Array(modelArtifacts.weightData.slice(12, 20)))
                .toEqual(new Float32Array([-7, -4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('Missing modelTopology and weightsManifest leads to error',
       (done: DoneFn) => {
         setupFakeWeightFiles({'path1/model.json': JSON.stringify({})});
         const handler = tf.io.browserHTTPRequest('path1/model.json');
         handler.load()
             .then(modelTopology1 => {
               done.fail(
                   'Loading from missing modelTopology and weightsManifest ' +
                   'succeeded expectedly.');
             })
             .catch(err => {
               expect(err.message)
                   .toMatch(/contains neither model topology or manifest/);
               done();
             });
       });
  });

  describe('Binary model', () => {
    let requestInits: RequestInit[];
    let modelData: ArrayBuffer;

    const setupFakeWeightFiles = (fileBufferMap: {
      [filename: string]: string|Float32Array|Int32Array|ArrayBuffer|Uint8Array|
      Uint16Array
    }) => {
      spyOn(window, 'fetch').and.callFake((path: string, init: RequestInit) => {
        requestInits.push(init);
        return new Response(fileBufferMap[path]);
      });
    };

    beforeEach(() => {
      requestInits = [];
      modelData = new ArrayBuffer(5);
    });

    it('1 group, 2 weights, 1 path', (done: DoneFn) => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);
      setupFakeWeightFiles({
        './model.pb': modelData,
        './weights_manifest.json': JSON.stringify(weightManifest1),
        './weightfile0': floatData,
      });

      const handler =
          tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json']);
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelData);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(floatData);
            expect(requestInits).toEqual([{}, {}, {}]);
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('1 group, 2 weights, 1 path, with requestInit', (done: DoneFn) => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData = new Float32Array([1, 3, 3, 7, 4]);

      setupFakeWeightFiles({
        './model.pb': modelData,
        './weights_manifest.json': JSON.stringify(weightManifest1),
        './weightfile0': floatData,
      });

      const handler = tf.io.browserHTTPRequest(
          ['./model.pb', './weights_manifest.json'],
          {headers: {'header_key_1': 'header_value_1'}});
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelData);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(floatData);
            expect(requestInits).toEqual([
              {headers: {'header_key_1': 'header_value_1'}},
              {headers: {'header_key_1': 'header_value_1'}},
              {headers: {'header_key_1': 'header_value_1'}},
            ]);
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('1 group, 2 weight, 2 paths', (done: DoneFn) => {
      const weightManifest1: tf.io.WeightsManifestConfig = [{
        paths: ['weightfile0', 'weightfile1'],
        weights: [
          {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          },
          {
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }
        ]
      }];
      const floatData1 = new Float32Array([1, 3, 3]);
      const floatData2 = new Float32Array([7, 4]);
      setupFakeWeightFiles({
        './model.pb': modelData,
        './weights_manifest.json': JSON.stringify(weightManifest1),
        './weightfile0': floatData1,
        './weightfile1': floatData2,
      });

      const handler =
          tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json']);
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelData);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(new Float32Array([1, 3, 3, 7, 4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('2 groups, 2 weight, 2 paths', (done: DoneFn) => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'dense/bias',
            shape: [2],
            dtype: 'float32',
          }],
        }
      ];
      const floatData1 = new Float32Array([1, 3, 3]);
      const floatData2 = new Float32Array([7, 4]);
      setupFakeWeightFiles({
        './model.pb': modelData,
        './weights_manifest.json': JSON.stringify(weightsManifest),
        './weightfile0': floatData1,
        './weightfile1': floatData2,
      });

      const handler =
          tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json']);
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelData);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(
                    weightsManifest[1].weights));
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(new Float32Array([1, 3, 3, 7, 4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', (done: DoneFn) => {
      const weightsManifest: tf.io.WeightsManifestConfig = [
        {
          paths: ['weightfile0'],
          weights: [{
            name: 'fooWeight',
            shape: [3, 1],
            dtype: 'int32',
          }]
        },
        {
          paths: ['weightfile1'],
          weights: [{
            name: 'barWeight',
            shape: [2],
            dtype: 'bool',
          }],
        }
      ];
      const floatData1 = new Int32Array([1, 3, 3]);
      const floatData2 = new Uint8Array([7, 4]);
      setupFakeWeightFiles({
        'path1/model.pb': modelData,
        'path2/weights_manifest.json': JSON.stringify(weightsManifest),
        'path2/weightfile0': floatData1,
        'path2/weightfile1': floatData2,
      });

      const handler = tf.io.browserHTTPRequest(
          ['path1/model.pb', 'path2/weights_manifest.json']);
      handler.load()
          .then(modelArtifacts => {
            expect(modelArtifacts.modelTopology).toEqual(modelData);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(
                    weightsManifest[1].weights));
            expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                .toEqual(new Int32Array([1, 3, 3]));
            expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                .toEqual(new Uint8Array([7, 4]));
            done();
          })
          .catch(err => done.fail(err.stack));
    });

    it('2 groups, 2 weight, weight path prefix, Int32 and Uint8 Data',
       (done: DoneFn) => {
         const weightsManifest: tf.io.WeightsManifestConfig = [
           {
             paths: ['weightfile0'],
             weights: [{
               name: 'fooWeight',
               shape: [3, 1],
               dtype: 'int32',
             }]
           },
           {
             paths: ['weightfile1'],
             weights: [{
               name: 'barWeight',
               shape: [2],
               dtype: 'bool',
             }],
           }
         ];
         const floatData1 = new Int32Array([1, 3, 3]);
         const floatData2 = new Uint8Array([7, 4]);
         setupFakeWeightFiles({
           'path1/model.pb': modelData,
           'path2/weights_manifest.json': JSON.stringify(weightsManifest),
           'path3/weightfile0': floatData1,
           'path3/weightfile1': floatData2,
         });

         const handler = tf.io.browserHTTPRequest(
             ['path1/model.pb', 'path2/weights_manifest.json'], {}, 'path3/');
         handler.load()
             .then(modelArtifacts => {
               expect(modelArtifacts.modelTopology).toEqual(modelData);
               expect(modelArtifacts.weightSpecs)
                   .toEqual(weightsManifest[0].weights.concat(
                       weightsManifest[1].weights));
               expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                   .toEqual(new Int32Array([1, 3, 3]));
               expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                   .toEqual(new Uint8Array([7, 4]));
               done();
             })
             .catch(err => done.fail(err.stack));
       });
    it('the url path length is not 2 should leads to error', () => {
      expect(() => tf.io.browserHTTPRequest(['path1/model.pb'])).toThrow();
    });
  });
});
