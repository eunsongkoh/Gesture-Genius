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
 * IOHandler implementations based on HTTP requests in the web browser.
 *
 * Uses [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 */
import { env } from '../environment';
import { assert } from '../util';
import { getModelArtifactsForJSON, getModelArtifactsInfoForJSON, getModelJSONForModelArtifacts, getWeightSpecs } from './io_utils';
import { CompositeArrayBuffer } from './composite_array_buffer';
import { IORouterRegistry } from './router_registry';
import { loadWeightsAsArrayBuffer } from './weights_loader';
const OCTET_STREAM_MIME_TYPE = 'application/octet-stream';
const JSON_TYPE = 'application/json';
class HTTPRequest {
    constructor(path, loadOptions) {
        this.DEFAULT_METHOD = 'POST';
        if (loadOptions == null) {
            loadOptions = {};
        }
        this.weightPathPrefix = loadOptions.weightPathPrefix;
        this.onProgress = loadOptions.onProgress;
        this.weightUrlConverter = loadOptions.weightUrlConverter;
        if (loadOptions.fetchFunc != null) {
            assert(typeof loadOptions.fetchFunc === 'function', () => 'Must pass a function that matches the signature of ' +
                '`fetch` (see ' +
                'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
            this.fetch = loadOptions.fetchFunc;
        }
        else {
            this.fetch = env().platform.fetch;
        }
        assert(path != null && path.length > 0, () => 'URL path for http must not be null, undefined or ' +
            'empty.');
        if (Array.isArray(path)) {
            assert(path.length === 2, () => 'URL paths for http must have a length of 2, ' +
                `(actual length is ${path.length}).`);
        }
        this.path = path;
        if (loadOptions.requestInit != null &&
            loadOptions.requestInit.body != null) {
            throw new Error('requestInit is expected to have no pre-existing body, but has one.');
        }
        this.requestInit = loadOptions.requestInit || {};
    }
    async save(modelArtifacts) {
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
            throw new Error('BrowserHTTPRequest.save() does not support saving model topology ' +
                'in binary formats yet.');
        }
        const init = Object.assign({ method: this.DEFAULT_METHOD }, this.requestInit);
        init.body = new FormData();
        const weightsManifest = [{
                paths: ['./model.weights.bin'],
                weights: modelArtifacts.weightSpecs,
            }];
        const modelTopologyAndWeightManifest = getModelJSONForModelArtifacts(modelArtifacts, weightsManifest);
        init.body.append('model.json', new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: JSON_TYPE }), 'model.json');
        if (modelArtifacts.weightData != null) {
            // TODO(mattsoulanille): Support saving models over 2GB that exceed
            // Chrome's ArrayBuffer size limit.
            const weightBuffer = CompositeArrayBuffer.join(modelArtifacts.weightData);
            init.body.append('model.weights.bin', new Blob([weightBuffer], { type: OCTET_STREAM_MIME_TYPE }), 'model.weights.bin');
        }
        const response = await this.fetch(this.path, init);
        if (response.ok) {
            return {
                modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts),
                responses: [response],
            };
        }
        else {
            throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ` +
                `${response.status}.`);
        }
    }
    /**
     * Load model artifacts via HTTP request(s).
     *
     * See the documentation to `tf.io.http` for details on the saved
     * artifacts.
     *
     * @returns The loaded model artifacts (if loading succeeds).
     */
    async load() {
        const modelConfigRequest = await this.fetch(this.path, this.requestInit);
        if (!modelConfigRequest.ok) {
            throw new Error(`Request to ${this.path} failed with status code ` +
                `${modelConfigRequest.status}. Please verify this URL points to ` +
                `the model JSON of the model to load.`);
        }
        let modelJSON;
        try {
            modelJSON = await modelConfigRequest.json();
        }
        catch (e) {
            let message = `Failed to parse model JSON of response from ${this.path}.`;
            // TODO(nsthorat): Remove this after some time when we're comfortable that
            // .pb files are mostly gone.
            if (this.path.endsWith('.pb')) {
                message += ' Your path contains a .pb file extension. ' +
                    'Support for .pb models have been removed in TensorFlow.js 1.0 ' +
                    'in favor of .json models. You can re-convert your Python ' +
                    'TensorFlow model using the TensorFlow.js 1.0 conversion scripts ' +
                    'or you can convert your.pb models with the \'pb2json\'' +
                    'NPM script in the tensorflow/tfjs-converter repository.';
            }
            else {
                message += ' Please make sure the server is serving valid ' +
                    'JSON for this request.';
            }
            throw new Error(message);
        }
        // We do not allow both modelTopology and weightsManifest to be missing.
        const modelTopology = modelJSON.modelTopology;
        const weightsManifest = modelJSON.weightsManifest;
        if (modelTopology == null && weightsManifest == null) {
            throw new Error(`The JSON from HTTP path ${this.path} contains neither model ` +
                `topology or manifest for weights.`);
        }
        return getModelArtifactsForJSON(modelJSON, (weightsManifest) => this.loadWeights(weightsManifest));
    }
    async loadWeights(weightsManifest) {
        const weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
        const [prefix, suffix] = parseUrl(weightPath);
        const pathPrefix = this.weightPathPrefix || prefix;
        const weightSpecs = getWeightSpecs(weightsManifest);
        const fetchURLs = [];
        const urlPromises = [];
        for (const weightsGroup of weightsManifest) {
            for (const path of weightsGroup.paths) {
                if (this.weightUrlConverter != null) {
                    urlPromises.push(this.weightUrlConverter(path));
                }
                else {
                    fetchURLs.push(pathPrefix + path + suffix);
                }
            }
        }
        if (this.weightUrlConverter) {
            fetchURLs.push(...await Promise.all(urlPromises));
        }
        const buffers = await loadWeightsAsArrayBuffer(fetchURLs, {
            requestInit: this.requestInit,
            fetchFunc: this.fetch,
            onProgress: this.onProgress
        });
        return [weightSpecs, buffers];
    }
}
HTTPRequest.URL_SCHEME_REGEX = /^https?:\/\//;
export { HTTPRequest };
/**
 * Extract the prefix and suffix of the url, where the prefix is the path before
 * the last file, and suffix is the search params after the last file.
 * ```
 * const url = 'http://tfhub.dev/model/1/tensorflowjs_model.pb?tfjs-format=file'
 * [prefix, suffix] = parseUrl(url)
 * // prefix = 'http://tfhub.dev/model/1/'
 * // suffix = '?tfjs-format=file'
 * ```
 * @param url the model url to be parsed.
 */
export function parseUrl(url) {
    const lastSlash = url.lastIndexOf('/');
    const lastSearchParam = url.lastIndexOf('?');
    const prefix = url.substring(0, lastSlash);
    const suffix = lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
    return [prefix + '/', suffix];
}
export function isHTTPScheme(url) {
    return url.match(HTTPRequest.URL_SCHEME_REGEX) != null;
}
export const httpRouter = (url, loadOptions) => {
    if (typeof fetch === 'undefined' &&
        (loadOptions == null || loadOptions.fetchFunc == null)) {
        // `http` uses `fetch` or `node-fetch`, if one wants to use it in
        // an environment that is not the browser or node they have to setup a
        // global fetch polyfill.
        return null;
    }
    else {
        let isHTTP = true;
        if (Array.isArray(url)) {
            isHTTP = url.every(urlItem => isHTTPScheme(urlItem));
        }
        else {
            isHTTP = isHTTPScheme(url);
        }
        if (isHTTP) {
            return http(url, loadOptions);
        }
    }
    return null;
};
IORouterRegistry.registerSaveRouter(httpRouter);
IORouterRegistry.registerLoadRouter(httpRouter);
/**
 * Creates an IOHandler subtype that sends model artifacts to HTTP server.
 *
 * An HTTP request of the `multipart/form-data` mime type will be sent to the
 * `path` URL. The form data includes artifacts that represent the topology
 * and/or weights of the model. In the case of Keras-style `tf.Model`, two
 * blobs (files) exist in form-data:
 *   - A JSON file consisting of `modelTopology` and `weightsManifest`.
 *   - A binary weights file consisting of the concatenated weight values.
 * These files are in the same format as the one generated by
 * [tfjs_converter](https://js.tensorflow.org/tutorials/import-keras.html).
 *
 * The following code snippet exemplifies the client-side code that uses this
 * function:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save(tf.io.http(
 *     'http://model-server:5000/upload', {requestInit: {method: 'PUT'}}));
 * console.log(saveResult);
 * ```
 *
 * If the default `POST` method is to be used, without any custom parameters
 * such as headers, you can simply pass an HTTP or HTTPS URL to `model.save`:
 *
 * ```js
 * const saveResult = await model.save('http://model-server:5000/upload');
 * ```
 *
 * The following GitHub Gist
 * https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
 * implements a server based on [flask](https://github.com/pallets/flask) that
 * can receive the request. Upon receiving the model artifacts via the requst,
 * this particular server reconstitutes instances of [Keras
 * Models](https://keras.io/models/model/) in memory.
 *
 *
 * @param path A URL path to the model.
 *   Can be an absolute HTTP path (e.g.,
 *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
 *   './model-upload').
 * @param requestInit Request configurations to be used when sending
 *    HTTP request to server using `fetch`. It can contain fields such as
 *    `method`, `credentials`, `headers`, `mode`, etc. See
 *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 *    for more information. `requestInit` must not have a body, because the
 * body will be set by TensorFlow.js. File blobs representing the model
 * topology (filename: 'model.json') and the weights of the model (filename:
 * 'model.weights.bin') will be appended to the body. If `requestInit` has a
 * `body`, an Error will be thrown.
 * @param loadOptions Optional configuration for the loading. It includes the
 *   following fields:
 *   - weightPathPrefix Optional, this specifies the path prefix for weight
 *     files, by default this is calculated from the path param.
 *   - fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
 *     the `fetch` from node-fetch can be used here.
 *   - onProgress Optional, progress callback function, fired periodically
 *     before the load is completed.
 * @returns An instance of `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
export function http(path, loadOptions) {
    return new HTTPRequest(path, loadOptions);
}
/**
 * Deprecated. Use `tf.io.http`.
 * @param path
 * @param loadOptions
 */
export function browserHTTPRequest(path, loadOptions) {
    return http(path, loadOptions);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaHR0cC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvaW8vaHR0cC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSDs7OztHQUlHO0FBRUgsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBRW5DLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDL0IsT0FBTyxFQUFDLHdCQUF3QixFQUFFLDRCQUE0QixFQUFFLDZCQUE2QixFQUFFLGNBQWMsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNqSSxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSwwQkFBMEIsQ0FBQztBQUM5RCxPQUFPLEVBQVcsZ0JBQWdCLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUU3RCxPQUFPLEVBQUMsd0JBQXdCLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUUxRCxNQUFNLHNCQUFzQixHQUFHLDBCQUEwQixDQUFDO0FBQzFELE1BQU0sU0FBUyxHQUFHLGtCQUFrQixDQUFDO0FBQ3JDLE1BQWEsV0FBVztJQWN0QixZQUFZLElBQVksRUFBRSxXQUF5QjtRQVAxQyxtQkFBYyxHQUFHLE1BQU0sQ0FBQztRQVEvQixJQUFJLFdBQVcsSUFBSSxJQUFJLEVBQUU7WUFDdkIsV0FBVyxHQUFHLEVBQUUsQ0FBQztTQUNsQjtRQUNELElBQUksQ0FBQyxnQkFBZ0IsR0FBRyxXQUFXLENBQUMsZ0JBQWdCLENBQUM7UUFDckQsSUFBSSxDQUFDLFVBQVUsR0FBRyxXQUFXLENBQUMsVUFBVSxDQUFDO1FBQ3pDLElBQUksQ0FBQyxrQkFBa0IsR0FBRyxXQUFXLENBQUMsa0JBQWtCLENBQUM7UUFFekQsSUFBSSxXQUFXLENBQUMsU0FBUyxJQUFJLElBQUksRUFBRTtZQUNqQyxNQUFNLENBQ0YsT0FBTyxXQUFXLENBQUMsU0FBUyxLQUFLLFVBQVUsRUFDM0MsR0FB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Q3pDLE9BQU8sSUFBSSxDQUFDLElBQUksRUFBRSxXQUFXLENBQUMsQ0FBQztBQUNqQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKipcbiAqIElPSGFuZGxlciBpbXBsZW1lbnRhdGlvbnMgYmFzZWQgb24gSFRUUCByZXF1ZXN0cyBpbiB0aGUgd2ViIGJyb3dzZXIuXG4gKlxuICogVXNlcyBbYGZldGNoYF0oaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL0ZldGNoX0FQSSkuXG4gKi9cblxuaW1wb3J0IHtlbnZ9IGZyb20gJy4uL2Vudmlyb25tZW50JztcblxuaW1wb3J0IHthc3NlcnR9IGZyb20gJy4uL3V0aWwnO1xuaW1wb3J0IHtnZXRNb2RlbEFydGlmYWN0c0ZvckpTT04sIGdldE1vZGVsQXJ0aWZhY3RzSW5mb0ZvckpTT04sIGdldE1vZGVsSlNPTkZvck1vZGVsQXJ0aWZhY3RzLCBnZXRXZWlnaHRTcGVjc30gZnJvbSAnLi9pb191dGlscyc7XG5pbXBvcnQge0NvbXBvc2l0ZUFycmF5QnVmZmVyfSBmcm9tICcuL2NvbXBvc2l0ZV9hcnJheV9idWZmZXInO1xuaW1wb3J0IHtJT1JvdXRlciwgSU9Sb3V0ZXJSZWdpc3RyeX0gZnJvbSAnLi9yb3V0ZXJfcmVnaXN0cnknO1xuaW1wb3J0IHtJT0hhbmRsZXIsIExvYWRPcHRpb25zLCBNb2RlbEFydGlmYWN0cywgTW9kZWxKU09OLCBPblByb2dyZXNzQ2FsbGJhY2ssIFNhdmVSZXN1bHQsIFdlaWdodERhdGEsIFdlaWdodHNNYW5pZmVzdENvbmZpZywgV2VpZ2h0c01hbmlmZXN0RW50cnl9IGZyb20gJy4vdHlwZXMnO1xuaW1wb3J0IHtsb2FkV2VpZ2h0c0FzQXJyYXlCdWZmZXJ9IGZyb20gJy4vd2VpZ2h0c19sb2FkZXInO1xuXG5jb25zdCBPQ1RFVF9TVFJFQU1fTUlNRV9UWVBFID0gJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSc7XG5jb25zdCBKU09OX1RZUEUgPSAnYXBwbGljYXRpb24vanNvbic7XG5leHBvcnQgY2xhc3MgSFRUUFJlcXVlc3QgaW1wbGVtZW50cyBJT0hhbmRsZXIge1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcGF0aDogc3RyaW5nO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgcmVxdWVzdEluaXQ6IFJlcXVlc3RJbml0O1xuXG4gIHByaXZhdGUgcmVhZG9ubHkgZmV0Y2g6IEZ1bmN0aW9uO1xuICBwcml2YXRlIHJlYWRvbmx5IHdlaWdodFVybENvbnZlcnRlcjogKHdlaWdodE5hbWU6IHN0cmluZykgPT4gUHJvbWlzZTxzdHJpbmc+O1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfTUVUSE9EID0gJ1BPU1QnO1xuXG4gIHN0YXRpYyByZWFkb25seSBVUkxfU0NIRU1FX1JFR0VYID0gL15odHRwcz86XFwvXFwvLztcblxuICBwcml2YXRlIHJlYWRvbmx5IHdlaWdodFBhdGhQcmVmaXg6IHN0cmluZztcbiAgcHJpdmF0ZSByZWFkb25seSBvblByb2dyZXNzOiBPblByb2dyZXNzQ2FsbGJhY2s7XG5cbiAgY29uc3RydWN0b3IocGF0aDogc3RyaW5nLCBsb2FkT3B0aW9ucz86IExvYWRPcHRpb25zKSB7XG4gICAgaWYgKGxvYWRPcHRpb25zID09IG51bGwpIHtcbiAgICAgIGxvYWRPcHRpb25zID0ge307XG4gICAgfVxuICAgIHRoaXMud2VpZ2h0UGF0aFByZWZpeCA9IGxvYWRPcHRpb25zLndlaWdodFBhdGhQcmVmaXg7XG4gICAgdGhpcy5vblByb2dyZXNzID0gbG9hZE9wdGlvbnMub25Qcm9ncmVzcztcbiAgICB0aGlzLndlaWdodFVybENvbnZlcnRlciA9IGxvYWRPcHRpb25zLndlaWdodFVybENvbnZlcnRlcjtcblxuICAgIGlmIChsb2FkT3B0aW9ucy5mZXRjaEZ1bmMgIT0gbnVsbCkge1xuICAgICAgYXNzZXJ0KFxuICAgICAgICAgIHR5cGVvZiBsb2FkT3B0aW9ucy5mZXRjaEZ1bmMgPT09ICdmdW5jdGlvbicsXG4gICAgICAgICAgKCkgPT4gJ011c3QgcGFzcyBhIGZ1bmN0aW9uIHRoYXQgbWF0Y2hlcyB0aGUgc2lnbmF0dXJlIG9mICcgK1xuICAgICAgICAgICAgICAnYGZldGNoYCAoc2VlICcgK1xuICAgICAgICAgICAgICAnaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL0ZldGNoX0FQSSknKTtcbiAgICAgIHRoaXMuZmV0Y2ggPSBsb2FkT3B0aW9ucy5mZXRjaEZ1bmM7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuZmV0Y2ggPSBlbnYoKS5wbGF0Zm9ybS5mZXRjaDtcbiAgICB9XG5cbiAgICBhc3NlcnQoXG4gICAgICAgIHBhdGggIT0gbnVsbCAmJiBwYXRoLmxlbmd0aCA+IDAsXG4gICAgICAgICgpID0+ICdVUkwgcGF0aCBmb3IgaHR0cCBtdXN0IG5vdCBiZSBudWxsLCB1bmRlZmluZWQgb3IgJyArXG4gICAgICAgICAgICAnZW1wdHkuJyk7XG5cbiAgICBpZiAoQXJyYXkuaXNBcnJheShwYXRoKSkge1xuICAgICAgYXNzZXJ0KFxuICAgICAgICAgIHBhdGgubGVuZ3RoID09PSAyLFxuICAgICAgICAgICgpID0+ICdVUkwgcGF0aHMgZm9yIGh0dHAgbXVzdCBoYXZlIGEgbGVuZ3RoIG9mIDIsICcgK1xuICAgICAgICAgICAgICBgKGFjdHVhbCBsZW5ndGggaXMgJHtwYXRoLmxlbmd0aH0pLmApO1xuICAgIH1cbiAgICB0aGlzLnBhdGggPSBwYXRoO1xuXG4gICAgaWYgKGxvYWRPcHRpb25zLnJlcXVlc3RJbml0ICE9IG51bGwgJiZcbiAgICAgICAgbG9hZE9wdGlvbnMucmVxdWVzdEluaXQuYm9keSAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ3JlcXVlc3RJbml0IGlzIGV4cGVjdGVkIHRvIGhhdmUgbm8gcHJlLWV4aXN0aW5nIGJvZHksIGJ1dCBoYXMgb25lLicpO1xuICAgIH1cbiAgICB0aGlzLnJlcXVlc3RJbml0ID0gbG9hZE9wdGlvbnMucmVxdWVzdEluaXQgfHwge307XG4gIH1cblxuICBhc3luYyBzYXZlKG1vZGVsQXJ0aWZhY3RzOiBNb2RlbEFydGlmYWN0cyk6IFByb21pc2U8U2F2ZVJlc3VsdD4ge1xuICAgIGlmIChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5IGluc3RhbmNlb2YgQXJyYXlCdWZmZXIpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQnJvd3NlckhUVFBSZXF1ZXN0LnNhdmUoKSBkb2VzIG5vdCBzdXBwb3J0IHNhdmluZyBtb2RlbCB0b3BvbG9neSAnICtcbiAgICAgICAgICAnaW4gYmluYXJ5IGZvcm1hdHMgeWV0LicpO1xuICAgIH1cblxuICAgIGNvbnN0IGluaXQgPSBPYmplY3QuYXNzaWduKHttZXRob2Q6IHRoaXMuREVGQVVMVF9NRVRIT0R9LCB0aGlzLnJlcXVlc3RJbml0KTtcbiAgICBpbml0LmJvZHkgPSBuZXcgRm9ybURhdGEoKTtcblxuICAgIGNvbnN0IHdlaWdodHNNYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgIHBhdGhzOiBbJy4vbW9kZWwud2VpZ2h0cy5iaW4nXSxcbiAgICAgIHdlaWdodHM6IG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzLFxuICAgIH1dO1xuICAgIGNvbnN0IG1vZGVsVG9wb2xvZ3lBbmRXZWlnaHRNYW5pZmVzdDogTW9kZWxKU09OID1cbiAgICAgICAgZ2V0TW9kZWxKU09ORm9yTW9kZWxBcnRpZmFjdHMobW9kZWxBcnRpZmFjdHMsIHdlaWdodHNNYW5pZmVzdCk7XG5cbiAgICBpbml0LmJvZHkuYXBwZW5kKFxuICAgICAgICAnbW9kZWwuanNvbicsXG4gICAgICAgIG5ldyBCbG9iKFxuICAgICAgICAgICAgW0pTT04uc3RyaW5naWZ5KG1vZGVsVG9wb2xvZ3lBbmRXZWlnaHRNYW5pZmVzdCldLFxuICAgICAgICAgICAge3R5cGU6IEpTT05fVFlQRX0pLFxuICAgICAgICAnbW9kZWwuanNvbicpO1xuXG4gICAgaWYgKG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEgIT0gbnVsbCkge1xuICAgICAgLy8gVE9ETyhtYXR0c291bGFuaWxsZSk6IFN1cHBvcnQgc2F2aW5nIG1vZGVscyBvdmVyIDJHQiB0aGF0IGV4Y2VlZFxuICAgICAgLy8gQ2hyb21lJ3MgQXJyYXlCdWZmZXIgc2l6ZSBsaW1pdC5cbiAgICAgIGNvbnN0IHdlaWdodEJ1ZmZlciA9IENvbXBvc2l0ZUFycmF5QnVmZmVyLmpvaW4obW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSk7XG5cbiAgICAgIGluaXQuYm9keS5hcHBlbmQoXG4gICAgICAgICAgJ21vZGVsLndlaWdodHMuYmluJyxcbiAgICAgICAgICBuZXcgQmxvYihbd2VpZ2h0QnVmZmVyXSwge3R5cGU6IE9DVEVUX1NUUkVBTV9NSU1FX1RZUEV9KSxcbiAgICAgICAgICAnbW9kZWwud2VpZ2h0cy5iaW4nKTtcbiAgICB9XG5cbiAgICBjb25zdCByZXNwb25zZSA9IGF3YWl0IHRoaXMuZmV0Y2godGhpcy5wYXRoLCBpbml0KTtcblxuICAgIGlmIChyZXNwb25zZS5vaykge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgbW9kZWxBcnRpZmFjdHNJbmZvOiBnZXRNb2RlbEFydGlmYWN0c0luZm9Gb3JKU09OKG1vZGVsQXJ0aWZhY3RzKSxcbiAgICAgICAgcmVzcG9uc2VzOiBbcmVzcG9uc2VdLFxuICAgICAgfTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBCcm93c2VySFRUUFJlcXVlc3Quc2F2ZSgpIGZhaWxlZCBkdWUgdG8gSFRUUCByZXNwb25zZSBzdGF0dXMgYCArXG4gICAgICAgICAgYCR7cmVzcG9uc2Uuc3RhdHVzfS5gKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogTG9hZCBtb2RlbCBhcnRpZmFjdHMgdmlhIEhUVFAgcmVxdWVzdChzKS5cbiAgICpcbiAgICogU2VlIHRoZSBkb2N1bWVudGF0aW9uIHRvIGB0Zi5pby5odHRwYCBmb3IgZGV0YWlscyBvbiB0aGUgc2F2ZWRcbiAgICogYXJ0aWZhY3RzLlxuICAgKlxuICAgKiBAcmV0dXJucyBUaGUgbG9hZGVkIG1vZGVsIGFydGlmYWN0cyAoaWYgbG9hZGluZyBzdWNjZWVkcykuXG4gICAqL1xuICBhc3luYyBsb2FkKCk6IFByb21pc2U8TW9kZWxBcnRpZmFjdHM+IHtcbiAgICBjb25zdCBtb2RlbENvbmZpZ1JlcXVlc3QgPSBhd2FpdCB0aGlzLmZldGNoKHRoaXMucGF0aCwgdGhpcy5yZXF1ZXN0SW5pdCk7XG5cbiAgICBpZiAoIW1vZGVsQ29uZmlnUmVxdWVzdC5vaykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBSZXF1ZXN0IHRvICR7dGhpcy5wYXRofSBmYWlsZWQgd2l0aCBzdGF0dXMgY29kZSBgICtcbiAgICAgICAgICBgJHttb2RlbENvbmZpZ1JlcXVlc3Quc3RhdHVzfS4gUGxlYXNlIHZlcmlmeSB0aGlzIFVSTCBwb2ludHMgdG8gYCArXG4gICAgICAgICAgYHRoZSBtb2RlbCBKU09OIG9mIHRoZSBtb2RlbCB0byBsb2FkLmApO1xuICAgIH1cbiAgICBsZXQgbW9kZWxKU09OOiBNb2RlbEpTT047XG4gICAgdHJ5IHtcbiAgICAgIG1vZGVsSlNPTiA9IGF3YWl0IG1vZGVsQ29uZmlnUmVxdWVzdC5qc29uKCk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgbGV0IG1lc3NhZ2UgPSBgRmFpbGVkIHRvIHBhcnNlIG1vZGVsIEpTT04gb2YgcmVzcG9uc2UgZnJvbSAke3RoaXMucGF0aH0uYDtcbiAgICAgIC8vIFRPRE8obnN0aG9yYXQpOiBSZW1vdmUgdGhpcyBhZnRlciBzb21lIHRpbWUgd2hlbiB3ZSdyZSBjb21mb3J0YWJsZSB0aGF0XG4gICAgICAvLyAucGIgZmlsZXMgYXJlIG1vc3RseSBnb25lLlxuICAgICAgaWYgKHRoaXMucGF0aC5lbmRzV2l0aCgnLnBiJykpIHtcbiAgICAgICAgbWVzc2FnZSArPSAnIFlvdXIgcGF0aCBjb250YWlucyBhIC5wYiBmaWxlIGV4dGVuc2lvbi4gJyArXG4gICAgICAgICAgICAnU3VwcG9ydCBmb3IgLnBiIG1vZGVscyBoYXZlIGJlZW4gcmVtb3ZlZCBpbiBUZW5zb3JGbG93LmpzIDEuMCAnICtcbiAgICAgICAgICAgICdpbiBmYXZvciBvZiAuanNvbiBtb2RlbHMuIFlvdSBjYW4gcmUtY29udmVydCB5b3VyIFB5dGhvbiAnICtcbiAgICAgICAgICAgICdUZW5zb3JGbG93IG1vZGVsIHVzaW5nIHRoZSBUZW5zb3JGbG93LmpzIDEuMCBjb252ZXJzaW9uIHNjcmlwdHMgJyArXG4gICAgICAgICAgICAnb3IgeW91IGNhbiBjb252ZXJ0IHlvdXIucGIgbW9kZWxzIHdpdGggdGhlIFxcJ3BiMmpzb25cXCcnICtcbiAgICAgICAgICAgICdOUE0gc2NyaXB0IGluIHRoZSB0ZW5zb3JmbG93L3RmanMtY29udmVydGVyIHJlcG9zaXRvcnkuJztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIG1lc3NhZ2UgKz0gJyBQbGVhc2UgbWFrZSBzdXJlIHRoZSBzZXJ2ZXIgaXMgc2VydmluZyB2YWxpZCAnICtcbiAgICAgICAgICAgICdKU09OIGZvciB0aGlzIHJlcXVlc3QuJztcbiAgICAgIH1cbiAgICAgIHRocm93IG5ldyBFcnJvcihtZXNzYWdlKTtcbiAgICB9XG5cbiAgICAvLyBXZSBkbyBub3QgYWxsb3cgYm90aCBtb2RlbFRvcG9sb2d5IGFuZCB3ZWlnaHRzTWFuaWZlc3QgdG8gYmUgbWlzc2luZy5cbiAgICBjb25zdCBtb2RlbFRvcG9sb2d5ID0gbW9kZWxKU09OLm1vZGVsVG9wb2xvZ3k7XG4gICAgY29uc3Qgd2VpZ2h0c01hbmlmZXN0ID0gbW9kZWxKU09OLndlaWdodHNNYW5pZmVzdDtcbiAgICBpZiAobW9kZWxUb3BvbG9neSA9PSBudWxsICYmIHdlaWdodHNNYW5pZmVzdCA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYFRoZSBKU09OIGZyb20gSFRUUCBwYXRoICR7dGhpcy5wYXRofSBjb250YWlucyBuZWl0aGVyIG1vZGVsIGAgK1xuICAgICAgICAgIGB0b3BvbG9neSBvciBtYW5pZmVzdCBmb3Igd2VpZ2h0cy5gKTtcbiAgICB9XG5cbiAgICByZXR1cm4gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OKFxuICAgICAgICBtb2RlbEpTT04sICh3ZWlnaHRzTWFuaWZlc3QpID0+IHRoaXMubG9hZFdlaWdodHMod2VpZ2h0c01hbmlmZXN0KSk7XG4gIH1cblxuICBwcml2YXRlIGFzeW5jIGxvYWRXZWlnaHRzKHdlaWdodHNNYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnKTpcbiAgICBQcm9taXNlPFtXZWlnaHRzTWFuaWZlc3RFbnRyeVtdLCBXZWlnaHREYXRhXT4ge1xuICAgIGNvbnN0IHdlaWdodFBhdGggPSBBcnJheS5pc0FycmF5KHRoaXMucGF0aCkgPyB0aGlzLnBhdGhbMV0gOiB0aGlzLnBhdGg7XG4gICAgY29uc3QgW3ByZWZpeCwgc3VmZml4XSA9IHBhcnNlVXJsKHdlaWdodFBhdGgpO1xuICAgIGNvbnN0IHBhdGhQcmVmaXggPSB0aGlzLndlaWdodFBhdGhQcmVmaXggfHwgcHJlZml4O1xuXG4gICAgY29uc3Qgd2VpZ2h0U3BlY3MgPSBnZXRXZWlnaHRTcGVjcyh3ZWlnaHRzTWFuaWZlc3QpO1xuXG4gICAgY29uc3QgZmV0Y2hVUkxzOiBzdHJpbmdbXSA9IFtdO1xuICAgIGNvbnN0IHVybFByb21pc2VzOiBBcnJheTxQcm9taXNlPHN0cmluZz4+ID0gW107XG4gICAgZm9yIChjb25zdCB3ZWlnaHRzR3JvdXAgb2Ygd2VpZ2h0c01hbmlmZXN0KSB7XG4gICAgICBmb3IgKGNvbnN0IHBhdGggb2Ygd2VpZ2h0c0dyb3VwLnBhdGhzKSB7XG4gICAgICAgIGlmICh0aGlzLndlaWdodFVybENvbnZlcnRlciAhPSBudWxsKSB7XG4gICAgICAgICAgdXJsUHJvbWlzZXMucHVzaCh0aGlzLndlaWdodFVybENvbnZlcnRlcihwYXRoKSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgZmV0Y2hVUkxzLnB1c2gocGF0aFByZWZpeCArIHBhdGggKyBzdWZmaXgpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKHRoaXMud2VpZ2h0VXJsQ29udmVydGVyKSB7XG4gICAgICBmZXRjaFVSTHMucHVzaCguLi5hd2FpdCBQcm9taXNlLmFsbCh1cmxQcm9taXNlcykpO1xuICAgIH1cblxuICAgIGNvbnN0IGJ1ZmZlcnMgPSBhd2FpdCBsb2FkV2VpZ2h0c0FzQXJyYXlCdWZmZXIoZmV0Y2hVUkxzLCB7XG4gICAgICByZXF1ZXN0SW5pdDogdGhpcy5yZXF1ZXN0SW5pdCxcbiAgICAgIGZldGNoRnVuYzogdGhpcy5mZXRjaCxcbiAgICAgIG9uUHJvZ3Jlc3M6IHRoaXMub25Qcm9ncmVzc1xuICAgIH0pO1xuICAgIHJldHVybiBbd2VpZ2h0U3BlY3MsIGJ1ZmZlcnNdO1xuICB9XG59XG5cbi8qKlxuICogRXh0cmFjdCB0aGUgcHJlZml4IGFuZCBzdWZmaXggb2YgdGhlIHVybCwgd2hlcmUgdGhlIHByZWZpeCBpcyB0aGUgcGF0aCBiZWZvcmVcbiAqIHRoZSBsYXN0IGZpbGUsIGFuZCBzdWZmaXggaXMgdGhlIHNlYXJjaCBwYXJhbXMgYWZ0ZXIgdGhlIGxhc3QgZmlsZS5cbiAqIGBgYFxuICogY29uc3QgdXJsID0gJ2h0dHA6Ly90Zmh1Yi5kZXYvbW9kZWwvMS90ZW5zb3JmbG93anNfbW9kZWwucGI/dGZqcy1mb3JtYXQ9ZmlsZSdcbiAqIFtwcmVmaXgsIHN1ZmZpeF0gPSBwYXJzZVVybCh1cmwpXG4gKiAvLyBwcmVmaXggPSAnaHR0cDovL3RmaHViLmRldi9tb2RlbC8xLydcbiAqIC8vIHN1ZmZpeCA9ICc/dGZqcy1mb3JtYXQ9ZmlsZSdcbiAqIGBgYFxuICogQHBhcmFtIHVybCB0aGUgbW9kZWwgdXJsIHRvIGJlIHBhcnNlZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHBhcnNlVXJsKHVybDogc3RyaW5nKTogW3N0cmluZywgc3RyaW5nXSB7XG4gIGNvbnN0IGxhc3RTbGFzaCA9IHVybC5sYXN0SW5kZXhPZignLycpO1xuICBjb25zdCBsYXN0U2VhcmNoUGFyYW0gPSB1cmwubGFzdEluZGV4T2YoJz8nKTtcbiAgY29uc3QgcHJlZml4ID0gdXJsLnN1YnN0cmluZygwLCBsYXN0U2xhc2gpO1xuICBjb25zdCBzdWZmaXggPVxuICAgICAgbGFzdFNlYXJjaFBhcmFtID4gbGFzdFNsYXNoID8gdXJsLnN1YnN0cmluZyhsYXN0U2VhcmNoUGFyYW0pIDogJyc7XG4gIHJldHVybiBbcHJlZml4ICsgJy8nLCBzdWZmaXhdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNIVFRQU2NoZW1lKHVybDogc3RyaW5nKTogYm9vbGVhbiB7XG4gIHJldHVybiB1cmwubWF0Y2goSFRUUFJlcXVlc3QuVVJMX1NDSEVNRV9SRUdFWCkgIT0gbnVsbDtcbn1cblxuZXhwb3J0IGNvbnN0IGh0dHBSb3V0ZXI6IElPUm91dGVyID1cbiAgICAodXJsOiBzdHJpbmcsIGxvYWRPcHRpb25zPzogTG9hZE9wdGlvbnMpID0+IHtcbiAgICAgIGlmICh0eXBlb2YgZmV0Y2ggPT09ICd1bmRlZmluZWQnICYmXG4gICAgICAgICAgKGxvYWRPcHRpb25zID09IG51bGwgfHwgbG9hZE9wdGlvbnMuZmV0Y2hGdW5jID09IG51bGwpKSB7XG4gICAgICAgIC8vIGBodHRwYCB1c2VzIGBmZXRjaGAgb3IgYG5vZGUtZmV0Y2hgLCBpZiBvbmUgd2FudHMgdG8gdXNlIGl0IGluXG4gICAgICAgIC8vIGFuIGVudmlyb25tZW50IHRoYXQgaXMgbm90IHRoZSBicm93c2VyIG9yIG5vZGUgdGhleSBoYXZlIHRvIHNldHVwIGFcbiAgICAgICAgLy8gZ2xvYmFsIGZldGNoIHBvbHlmaWxsLlxuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGxldCBpc0hUVFAgPSB0cnVlO1xuICAgICAgICBpZiAoQXJyYXkuaXNBcnJheSh1cmwpKSB7XG4gICAgICAgICAgaXNIVFRQID0gdXJsLmV2ZXJ5KHVybEl0ZW0gPT4gaXNIVFRQU2NoZW1lKHVybEl0ZW0pKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBpc0hUVFAgPSBpc0hUVFBTY2hlbWUodXJsKTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoaXNIVFRQKSB7XG4gICAgICAgICAgcmV0dXJuIGh0dHAodXJsLCBsb2FkT3B0aW9ucyk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIHJldHVybiBudWxsO1xuICAgIH07XG5JT1JvdXRlclJlZ2lzdHJ5LnJlZ2lzdGVyU2F2ZVJvdXRlcihodHRwUm91dGVyKTtcbklPUm91dGVyUmVnaXN0cnkucmVnaXN0ZXJMb2FkUm91dGVyKGh0dHBSb3V0ZXIpO1xuXG4vKipcbiAqIENyZWF0ZXMgYW4gSU9IYW5kbGVyIHN1YnR5cGUgdGhhdCBzZW5kcyBtb2RlbCBhcnRpZmFjdHMgdG8gSFRUUCBzZXJ2ZXIuXG4gKlxuICogQW4gSFRUUCByZXF1ZXN0IG9mIHRoZSBgbXVsdGlwYXJ0L2Zvcm0tZGF0YWAgbWltZSB0eXBlIHdpbGwgYmUgc2VudCB0byB0aGVcbiAqIGBwYXRoYCBVUkwuIFRoZSBmb3JtIGRhdGEgaW5jbHVkZXMgYXJ0aWZhY3RzIHRoYXQgcmVwcmVzZW50IHRoZSB0b3BvbG9neVxuICogYW5kL29yIHdlaWdodHMgb2YgdGhlIG1vZGVsLiBJbiB0aGUgY2FzZSBvZiBLZXJhcy1zdHlsZSBgdGYuTW9kZWxgLCB0d29cbiAqIGJsb2JzIChmaWxlcykgZXhpc3QgaW4gZm9ybS1kYXRhOlxuICogICAtIEEgSlNPTiBmaWxlIGNvbnNpc3Rpbmcgb2YgYG1vZGVsVG9wb2xvZ3lgIGFuZCBgd2VpZ2h0c01hbmlmZXN0YC5cbiAqICAgLSBBIGJpbmFyeSB3ZWlnaHRzIGZpbGUgY29uc2lzdGluZyBvZiB0aGUgY29uY2F0ZW5hdGVkIHdlaWdodCB2YWx1ZXMuXG4gKiBUaGVzZSBmaWxlcyBhcmUgaW4gdGhlIHNhbWUgZm9ybWF0IGFzIHRoZSBvbmUgZ2VuZXJhdGVkIGJ5XG4gKiBbdGZqc19jb252ZXJ0ZXJdKGh0dHBzOi8vanMudGVuc29yZmxvdy5vcmcvdHV0b3JpYWxzL2ltcG9ydC1rZXJhcy5odG1sKS5cbiAqXG4gKiBUaGUgZm9sbG93aW5nIGNvZGUgc25pcHBldCBleGVtcGxpZmllcyB0aGUgY2xpZW50LXNpZGUgY29kZSB0aGF0IHVzZXMgdGhpc1xuICogZnVuY3Rpb246XG4gKlxuICogYGBganNcbiAqIGNvbnN0IG1vZGVsID0gdGYuc2VxdWVudGlhbCgpO1xuICogbW9kZWwuYWRkKFxuICogICAgIHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDEsIGlucHV0U2hhcGU6IFsxMDBdLCBhY3RpdmF0aW9uOiAnc2lnbW9pZCd9KSk7XG4gKlxuICogY29uc3Qgc2F2ZVJlc3VsdCA9IGF3YWl0IG1vZGVsLnNhdmUodGYuaW8uaHR0cChcbiAqICAgICAnaHR0cDovL21vZGVsLXNlcnZlcjo1MDAwL3VwbG9hZCcsIHtyZXF1ZXN0SW5pdDoge21ldGhvZDogJ1BVVCd9fSkpO1xuICogY29uc29sZS5sb2coc2F2ZVJlc3VsdCk7XG4gKiBgYGBcbiAqXG4gKiBJZiB0aGUgZGVmYXVsdCBgUE9TVGAgbWV0aG9kIGlzIHRvIGJlIHVzZWQsIHdpdGhvdXQgYW55IGN1c3RvbSBwYXJhbWV0ZXJzXG4gKiBzdWNoIGFzIGhlYWRlcnMsIHlvdSBjYW4gc2ltcGx5IHBhc3MgYW4gSFRUUCBvciBIVFRQUyBVUkwgdG8gYG1vZGVsLnNhdmVgOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBzYXZlUmVzdWx0ID0gYXdhaXQgbW9kZWwuc2F2ZSgnaHR0cDovL21vZGVsLXNlcnZlcjo1MDAwL3VwbG9hZCcpO1xuICogYGBgXG4gKlxuICogVGhlIGZvbGxvd2luZyBHaXRIdWIgR2lzdFxuICogaHR0cHM6Ly9naXN0LmdpdGh1Yi5jb20vZHNtaWxrb3YvMWI2MDQ2ZmQ2MTMyZDc0MDhkNTI1N2IwOTc2Zjc4NjRcbiAqIGltcGxlbWVudHMgYSBzZXJ2ZXIgYmFzZWQgb24gW2ZsYXNrXShodHRwczovL2dpdGh1Yi5jb20vcGFsbGV0cy9mbGFzaykgdGhhdFxuICogY2FuIHJlY2VpdmUgdGhlIHJlcXVlc3QuIFVwb24gcmVjZWl2aW5nIHRoZSBtb2RlbCBhcnRpZmFjdHMgdmlhIHRoZSByZXF1c3QsXG4gKiB0aGlzIHBhcnRpY3VsYXIgc2VydmVyIHJlY29uc3RpdHV0ZXMgaW5zdGFuY2VzIG9mIFtLZXJhc1xuICogTW9kZWxzXShodHRwczovL2tlcmFzLmlvL21vZGVscy9tb2RlbC8pIGluIG1lbW9yeS5cbiAqXG4gKlxuICogQHBhcmFtIHBhdGggQSBVUkwgcGF0aCB0byB0aGUgbW9kZWwuXG4gKiAgIENhbiBiZSBhbiBhYnNvbHV0ZSBIVFRQIHBhdGggKGUuZy4sXG4gKiAgICdodHRwOi8vbG9jYWxob3N0OjgwMDAvbW9kZWwtdXBsb2FkKScpIG9yIGEgcmVsYXRpdmUgcGF0aCAoZS5nLixcbiAqICAgJy4vbW9kZWwtdXBsb2FkJykuXG4gKiBAcGFyYW0gcmVxdWVzdEluaXQgUmVxdWVzdCBjb25maWd1cmF0aW9ucyB0byBiZSB1c2VkIHdoZW4gc2VuZGluZ1xuICogICAgSFRUUCByZXF1ZXN0IHRvIHNlcnZlciB1c2luZyBgZmV0Y2hgLiBJdCBjYW4gY29udGFpbiBmaWVsZHMgc3VjaCBhc1xuICogICAgYG1ldGhvZGAsIGBjcmVkZW50aWFsc2AsIGBoZWFkZXJzYCwgYG1vZGVgLCBldGMuIFNlZVxuICogICAgaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL1JlcXVlc3QvUmVxdWVzdFxuICogICAgZm9yIG1vcmUgaW5mb3JtYXRpb24uIGByZXF1ZXN0SW5pdGAgbXVzdCBub3QgaGF2ZSBhIGJvZHksIGJlY2F1c2UgdGhlXG4gKiBib2R5IHdpbGwgYmUgc2V0IGJ5IFRlbnNvckZsb3cuanMuIEZpbGUgYmxvYnMgcmVwcmVzZW50aW5nIHRoZSBtb2RlbFxuICogdG9wb2xvZ3kgKGZpbGVuYW1lOiAnbW9kZWwuanNvbicpIGFuZCB0aGUgd2VpZ2h0cyBvZiB0aGUgbW9kZWwgKGZpbGVuYW1lOlxuICogJ21vZGVsLndlaWdodHMuYmluJykgd2lsbCBiZSBhcHBlbmRlZCB0byB0aGUgYm9keS4gSWYgYHJlcXVlc3RJbml0YCBoYXMgYVxuICogYGJvZHlgLCBhbiBFcnJvciB3aWxsIGJlIHRocm93bi5cbiAqIEBwYXJhbSBsb2FkT3B0aW9ucyBPcHRpb25hbCBjb25maWd1cmF0aW9uIGZvciB0aGUgbG9hZGluZy4gSXQgaW5jbHVkZXMgdGhlXG4gKiAgIGZvbGxvd2luZyBmaWVsZHM6XG4gKiAgIC0gd2VpZ2h0UGF0aFByZWZpeCBPcHRpb25hbCwgdGhpcyBzcGVjaWZpZXMgdGhlIHBhdGggcHJlZml4IGZvciB3ZWlnaHRcbiAqICAgICBmaWxlcywgYnkgZGVmYXVsdCB0aGlzIGlzIGNhbGN1bGF0ZWQgZnJvbSB0aGUgcGF0aCBwYXJhbS5cbiAqICAgLSBmZXRjaEZ1bmMgT3B0aW9uYWwsIGN1c3RvbSBgZmV0Y2hgIGZ1bmN0aW9uLiBFLmcuLCBpbiBOb2RlLmpzLFxuICogICAgIHRoZSBgZmV0Y2hgIGZyb20gbm9kZS1mZXRjaCBjYW4gYmUgdXNlZCBoZXJlLlxuICogICAtIG9uUHJvZ3Jlc3MgT3B0aW9uYWwsIHByb2dyZXNzIGNhbGxiYWNrIGZ1bmN0aW9uLCBmaXJlZCBwZXJpb2RpY2FsbHlcbiAqICAgICBiZWZvcmUgdGhlIGxvYWQgaXMgY29tcGxldGVkLlxuICogQHJldHVybnMgQW4gaW5zdGFuY2Ugb2YgYElPSGFuZGxlcmAuXG4gKlxuICogQGRvYyB7XG4gKiAgIGhlYWRpbmc6ICdNb2RlbHMnLFxuICogICBzdWJoZWFkaW5nOiAnTG9hZGluZycsXG4gKiAgIG5hbWVzcGFjZTogJ2lvJyxcbiAqICAgaWdub3JlQ0k6IHRydWVcbiAqIH1cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGh0dHAocGF0aDogc3RyaW5nLCBsb2FkT3B0aW9ucz86IExvYWRPcHRpb25zKTogSU9IYW5kbGVyIHtcbiAgcmV0dXJuIG5ldyBIVFRQUmVxdWVzdChwYXRoLCBsb2FkT3B0aW9ucyk7XG59XG5cbi8qKlxuICogRGVwcmVjYXRlZC4gVXNlIGB0Zi5pby5odHRwYC5cbiAqIEBwYXJhbSBwYXRoXG4gKiBAcGFyYW0gbG9hZE9wdGlvbnNcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGJyb3dzZXJIVFRQUmVxdWVzdChcbiAgICBwYXRoOiBzdHJpbmcsIGxvYWRPcHRpb25zPzogTG9hZE9wdGlvbnMpOiBJT0hhbmRsZXIge1xuICByZXR1cm4gaHR0cChwYXRoLCBsb2FkT3B0aW9ucyk7XG59XG4iXX0=