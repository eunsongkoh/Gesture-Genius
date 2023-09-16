/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original source: keras/engine/topology.py */
import { serialization, tidy, util } from '@tensorflow/tfjs-core';
import { getNextUniqueTensorId, getUid } from '../backend/state';
import { getScopedTensorName, getUniqueTensorName, nameScope } from '../common';
import { AttributeError, NotImplementedError, RuntimeError, ValueError } from '../errors';
import { getInitializer } from '../initializers';
import * as generic_utils from '../utils/generic_utils';
import * as types_utils from '../utils/types_utils';
import * as variable_utils from '../utils/variable_utils';
import { batchGetValue, batchSetValue, LayerVariable } from '../variables';
/**
 * Specifies the ndim, dtype and shape of every input to a layer.
 *
 * Every layer should expose (if appropriate) an `inputSpec` attribute:
 * a list of instances of InputSpec (one per input tensor).
 *
 * A null entry in a shape is compatible with any dimension,
 * a null shape is compatible with any shape.
 */
export class InputSpec {
    constructor(args) {
        this.dtype = args.dtype;
        this.shape = args.shape;
        /*
          TODO(michaelterry): Could throw error if ndim and shape are both defined
            (then backport).
        */
        if (args.shape != null) {
            this.ndim = args.shape.length;
        }
        else {
            this.ndim = args.ndim;
        }
        this.maxNDim = args.maxNDim;
        this.minNDim = args.minNDim;
        this.axes = args.axes || {};
    }
}
/**
 * `tf.SymbolicTensor` is a placeholder for a Tensor without any concrete value.
 *
 * They are most often encountered when building a graph of `Layer`s for a
 * `tf.LayersModel` and the input data's shape, but not values are known.
 *
 * @doc {heading: 'Models', 'subheading': 'Classes'}
 */
export class SymbolicTensor {
    /**
     *
     * @param dtype
     * @param shape
     * @param sourceLayer The Layer that produced this symbolic tensor.
     * @param inputs The inputs passed to sourceLayer's __call__() method.
     * @param nodeIndex
     * @param tensorIndex
     * @param callArgs The keyword arguments passed to the __call__() method.
     * @param name
     * @param outputTensorIndex The index of this tensor in the list of outputs
     *   returned by apply().
     */
    constructor(dtype, shape, sourceLayer, inputs, callArgs, name, outputTensorIndex) {
        this.dtype = dtype;
        this.shape = shape;
        this.sourceLayer = sourceLayer;
        this.inputs = inputs;
        this.callArgs = callArgs;
        this.outputTensorIndex = outputTensorIndex;
        this.id = getNextUniqueTensorId();
        if (name != null) {
            this.originalName = getScopedTensorName(name);
            this.name = getUniqueTensorName(this.originalName);
        }
        this.rank = shape.length;
    }
}
let _nextNodeID = 0;
/**
 * A `Node` describes the connectivity between two layers.
 *
 * Each time a layer is connected to some new input,
 * a node is added to `layer.inboundNodes`.
 *
 * Each time the output of a layer is used by another layer,
 * a node is added to `layer.outboundNodes`.
 *
 * `nodeIndices` and `tensorIndices` are basically fine-grained coordinates
 * describing the origin of the `inputTensors`, verifying the following:
 *
 * `inputTensors[i] ==
 * inboundLayers[i].inboundNodes[nodeIndices[i]].outputTensors[
 *   tensorIndices[i]]`
 *
 * A node from layer A to layer B is added to:
 *     A.outboundNodes
 *     B.inboundNodes
 */
export class Node {
    constructor(args, 
    // TODO(michaelterry): Define actual type for this.
    callArgs) {
        this.callArgs = callArgs;
        this.id = _nextNodeID++;
        /*
          Layer instance (NOT a list).
          this is the layer that takes a list of input tensors
          and turns them into a list of output tensors.
          the current node will be added to
          the inboundNodes of outboundLayer.
        */
        this.outboundLayer = args.outboundLayer;
        /*
            The following 3 properties describe where
            the input tensors come from: which layers,
            and for each layer, which node and which
            tensor output of each node.
        */
        // List of layer instances.
        this.inboundLayers = args.inboundLayers;
        // List of integers, 1:1 mapping with inboundLayers.
        this.nodeIndices = args.nodeIndices;
        // List of integers, 1:1 mapping with inboundLayers.
        this.tensorIndices = args.tensorIndices;
        /*
            Following 2 properties:
            tensor inputs and outputs of outboundLayer.
        */
        // List of tensors. 1:1 mapping with inboundLayers.
        this.inputTensors = args.inputTensors;
        // List of tensors, created by outboundLayer.call().
        this.outputTensors = args.outputTensors;
        /*
            Following 2 properties: input and output masks.
            List of tensors, 1:1 mapping with inputTensor.
        */
        this.inputMasks = args.inputMasks;
        // List of tensors, created by outboundLayer.computeMask().
        this.outputMasks = args.outputMasks;
        // Following 2 properties: input and output shapes.
        // List of shape tuples, shapes of inputTensors.
        this.inputShapes = args.inputShapes;
        // List of shape tuples, shapes of outputTensors.
        this.outputShapes = args.outputShapes;
        // Add nodes to all layers involved.
        for (const layer of args.inboundLayers) {
            if (layer != null) {
                layer.outboundNodes.push(this);
            }
        }
        args.outboundLayer.inboundNodes.push(this);
    }
    getConfig() {
        const inboundNames = [];
        for (const layer of this.inboundLayers) {
            if (layer != null) {
                inboundNames.push(layer.name);
            }
            else {
                inboundNames.push(null);
            }
        }
        return {
            outboundLayer: this.outboundLayer ? this.outboundLayer.name : null,
            inboundLayers: inboundNames,
            nodeIndices: this.nodeIndices,
            tensorIndices: this.tensorIndices
        };
    }
}
let _nextLayerID = 0;
/**
 * A layer is a grouping of operations and weights that can be composed to
 * create a `tf.LayersModel`.
 *
 * Layers are constructed by using the functions under the
 * [tf.layers](#Layers-Basic) namespace.
 *
 * @doc {heading: 'Layers', subheading: 'Classes', namespace: 'layers'}
 */
export class Layer extends serialization.Serializable {
    constructor(args = {}) {
        super();
        this._callHook = null;
        this._addedWeightNames = [];
        // Porting Notes: PyKeras does not have this property in this base Layer
        //   class. Instead lets Layer subclass set it dynamically and checks the
        //   value with `hasattr`. In tfjs-layers, we let this be a member of this
        //   base class.
        this._stateful = false;
        this.id = _nextLayerID++;
        this.activityRegularizer = null;
        this.inputSpec = null;
        this.supportsMasking = false;
        // These properties will be set upon call of this.build()
        this._trainableWeights = [];
        this._nonTrainableWeights = [];
        this._losses = [];
        this._updates = [];
        this._built = false;
        /*
          These lists will be filled via successive calls
          to this.addInboundNode().
         */
        this.inboundNodes = [];
        this.outboundNodes = [];
        let name = args.name;
        if (!name) {
            const prefix = this.getClassName();
            name = generic_utils.toSnakeCase(prefix) + '_' + getUid(prefix);
        }
        this.name = name;
        this.trainable_ = args.trainable == null ? true : args.trainable;
        if (args.inputShape != null || args.batchInputShape != null) {
            /*
              In this case we will later create an input layer
              to insert before the current layer
             */
            let batchInputShape;
            if (args.batchInputShape != null) {
                batchInputShape = args.batchInputShape;
            }
            else if (args.inputShape != null) {
                let batchSize = null;
                if (args.batchSize != null) {
                    batchSize = args.batchSize;
                }
                batchInputShape = [batchSize].concat(args.inputShape);
            }
            this.batchInputShape = batchInputShape;
            // Set dtype.
            let dtype = args.dtype;
            if (dtype == null) {
                dtype = args.inputDType;
            }
            if (dtype == null) {
                dtype = 'float32';
            }
            this.dtype = dtype;
        }
        if (args.weights != null) {
            this.initialWeights = args.weights;
        }
        else {
            this.initialWeights = null;
        }
        // The value of `_refCount` is initialized to null. When the layer is used
        // in a symbolic way for the first time, it will be set to 1.
        this._refCount = null;
        this.fastWeightInitDuringBuild = false;
    }
    /**
     * Converts a layer and its index to a unique (immutable type) name.
     * This function is used internally with `this.containerNodes`.
     * @param layer The layer.
     * @param nodeIndex The layer's position (e.g. via enumerate) in a list of
     *   nodes.
     *
     * @returns The unique name.
     */
    static nodeKey(layer, nodeIndex) {
        return layer.name + '_ib-' + nodeIndex.toString();
    }
    /**
     * Returns this.inboundNode at index nodeIndex.
     *
     * Porting note: This is a replacement for _get_node_attribute_at_index()
     * @param nodeIndex
     * @param attrName The name of the attribute related to request for this node.
     */
    getNodeAtIndex(nodeIndex, attrName) {
        if (this.inboundNodes.length === 0) {
            throw new RuntimeError('The layer has never been called ' +
                `and thus has no defined ${attrName}.`);
        }
        if (this.inboundNodes.length <= nodeIndex) {
            throw new ValueError(`Asked to get ${attrName} at node ${nodeIndex}, ` +
                `but the layer has only ${this.inboundNodes.length} inbound nodes.`);
        }
        return this.inboundNodes[nodeIndex];
    }
    /**
     * Retrieves the input tensor(s) of a layer at a given node.
     *
     * @param nodeIndex Integer, index of the node from which to retrieve the
     *   attribute. E.g. `nodeIndex=0` will correspond to the first time the layer
     *   was called.
     *
     * @return A tensor (or list of tensors if the layer has multiple inputs).
     */
    getInputAt(nodeIndex) {
        return generic_utils.singletonOrArray(this.getNodeAtIndex(nodeIndex, 'input').inputTensors);
    }
    /**
     * Retrieves the output tensor(s) of a layer at a given node.
     *
     * @param nodeIndex Integer, index of the node from which to retrieve the
     *   attribute. E.g. `nodeIndex=0` will correspond to the first time the layer
     *   was called.
     *
     * @return A tensor (or list of tensors if the layer has multiple outputs).
     */
    getOutputAt(nodeIndex) {
        return generic_utils.singletonOrArray(this.getNodeAtIndex(nodeIndex, 'output').outputTensors);
    }
    // Properties
    /**
     * Retrieves the input tensor(s) of a layer.
     *
     * Only applicable if the layer has exactly one inbound node,
     * i.e. if it is connected to one incoming layer.
     *
     * @return Input tensor or list of input tensors.
     *
     * @exception AttributeError if the layer is connected to more than one
     *   incoming layers.
     */
    get input() {
        if (this.inboundNodes.length > 1) {
            throw new AttributeError(`Layer ${this.name}` +
                ' has multiple inbound nodes, ' +
                'hence the notion of "layer input" ' +
                'is ill-defined. ' +
                'Use `getInputAt(nodeIndex)` instead.');
        }
        else if (this.inboundNodes.length === 0) {
            throw new AttributeError(`Layer ${this.name}` +
                ' is not connected, no input to return.');
        }
        return generic_utils.singletonOrArray(this.getNodeAtIndex(0, 'input').inputTensors);
    }
    /**
     * Retrieves the output tensor(s) of a layer.
     *
     * Only applicable if the layer has exactly one inbound node,
     * i.e. if it is connected to one incoming layer.
     *
     * @return Output tensor or list of output tensors.
     *
     * @exception AttributeError if the layer is connected to more than one
     *   incoming layers.
     */
    get output() {
        if (this.inboundNodes.length === 0) {
            throw new AttributeError(`Layer ${this.name}` +
                ' has no inbound nodes.');
        }
        if (this.inboundNodes.length > 1) {
            throw new AttributeError(`Layer ${this.name}` +
                ' has multiple inbound nodes, ' +
                'hence the notion of "layer output" ' +
                'is ill-defined. ' +
                'Use `getOutputAt(nodeIndex)` instead.');
        }
        return generic_utils.singletonOrArray(this.getNodeAtIndex(0, 'output').outputTensors);
    }
    get losses() {
        return this._losses;
    }
    /**
     * Retrieves the Layer's current loss values.
     *
     * Used for regularizers during training.
     */
    calculateLosses() {
        // Porting Node: This is an augmentation to Layer.loss in PyKeras.
        //   In PyKeras, Layer.loss returns symbolic tensors. Here a concrete
        //   Tensor (specifically Scalar) values are returned. This is due to the
        //   imperative backend.
        return this.losses.map(lossFn => lossFn());
    }
    get updates() {
        return this._updates;
    }
    get built() {
        return this._built;
    }
    set built(built) {
        this._built = built;
    }
    get trainable() {
        return this.trainable_;
    }
    set trainable(trainable) {
        this._trainableWeights.forEach(w => w.trainable = trainable);
        this.trainable_ = trainable;
    }
    get trainableWeights() {
        if (this.trainable_) {
            return this._trainableWeights.filter(w => w.trainable);
        }
        else {
            return [];
        }
    }
    set trainableWeights(weights) {
        this._trainableWeights = weights;
    }
    get nonTrainableWeights() {
        if (this.trainable) {
            return this._trainableWeights.filter(w => !w.trainable)
                .concat(this._nonTrainableWeights);
        }
        else {
            return this._trainableWeights.concat(this._nonTrainableWeights);
        }
    }
    set nonTrainableWeights(weights) {
        this._nonTrainableWeights = weights;
    }
    /**
     * The concatenation of the lists trainableWeights and nonTrainableWeights
     * (in this order).
     */
    get weights() {
        return this.trainableWeights.concat(this.nonTrainableWeights);
    }
    get stateful() {
        return this._stateful;
    }
    /**
     * Reset the states of the layer.
     *
     * This method of the base Layer class is essentially a no-op.
     * Subclasses that are stateful (e.g., stateful RNNs) should override this
     * method.
     */
    resetStates() {
        if (!this.stateful) {
            throw new Error('Cannot call the resetStates() method of a non-stateful Layer ' +
                'object.');
        }
    }
    /**
     * Checks compatibility between the layer and provided inputs.
     *
     * This checks that the tensor(s) `input`
     * verify the input assumptions of the layer
     * (if any). If not, exceptions are raised.
     *
     * @param inputs Input tensor or list of input tensors.
     *
     * @exception ValueError in case of mismatch between
     *   the provided inputs and the expectations of the layer.
     */
    assertInputCompatibility(inputs) {
        const inputsList = generic_utils.toList(inputs);
        if (this.inputSpec == null || this.inputSpec.length === 0) {
            return;
        }
        const inputSpec = generic_utils.toList(this.inputSpec);
        if (inputsList.length !== inputSpec.length) {
            throw new ValueError(`Layer ${this.name} expects ${inputSpec.length} inputs, ` +
                `but it received ${inputsList.length} input tensors. ` +
                `Input received: ${inputs}`);
        }
        for (let inputIndex = 0; inputIndex < inputsList.length; inputIndex++) {
            const x = inputsList[inputIndex];
            const spec = inputSpec[inputIndex];
            if (spec == null) {
                continue;
            }
            // Check ndim.
            const ndim = x.rank;
            if (spec.ndim != null) {
                if (ndim !== spec.ndim) {
                    throw new ValueError(`Input ${inputIndex} is incompatible with layer ${this.name}: ` +
                        `expected ndim=${spec.ndim}, found ndim=${ndim}`);
                }
            }
            if (spec.maxNDim != null) {
                if (ndim > spec.maxNDim) {
                    throw new ValueError(`Input ${inputIndex} is incompatible with layer ${this.name}` +
                        `: expected max_ndim=${spec.maxNDim}, found ndim=${ndim}`);
                }
            }
            if (spec.minNDim != null) {
                if (ndim < spec.minNDim) {
                    throw new ValueError(`Input ${inputIndex} is incompatible with layer ${this.name}` +
                        `: expected min_ndim=${spec.minNDim}, found ndim=${ndim}.`);
                }
            }
            // Check dtype.
            if (spec.dtype != null) {
                if (x.dtype !== spec.dtype) {
                    throw new ValueError(`Input ${inputIndex} is incompatible with layer ${this.name} ` +
                        `: expected dtype=${spec.dtype}, found dtype=${x.dtype}.`);
                }
            }
            // Check specific shape axes.
            if (spec.axes) {
                const xShape = x.shape;
                for (const key in spec.axes) {
                    const axis = Number(key);
                    const value = spec.axes[key];
                    // Perform Python-style slicing in case axis < 0;
                    // TODO(cais): Use https://github.com/alvivi/typescript-underscore to
                    // ensure type safety through Underscore calls.
                    const xShapeAtAxis = axis >= 0 ? xShape[axis] : xShape[xShape.length + axis];
                    if (value != null && [value, null].indexOf(xShapeAtAxis) === -1) {
                        throw new ValueError(`Input ${inputIndex} is incompatible with layer ` +
                            `${this.name}: expected axis ${axis} of input shape to ` +
                            `have value ${value} but got shape ${xShape}.`);
                    }
                }
            }
            // Check shape.
            if (spec.shape != null) {
                for (let i = 0; i < spec.shape.length; ++i) {
                    const specDim = spec.shape[i];
                    const dim = x.shape[i];
                    if (specDim != null && dim != null) {
                        if (specDim !== dim) {
                            throw new ValueError(`Input ${inputIndex} is incompatible with layer ` +
                                `${this.name}: expected shape=${spec.shape}, ` +
                                `found shape=${x.shape}.`);
                        }
                    }
                }
            }
        }
    }
    /**
     * This is where the layer's logic lives.
     *
     * @param inputs Input tensor, or list/tuple of input tensors.
     * @param kwargs Additional keyword arguments.
     *
     * @return A tensor or list/tuple of tensors.
     */
    call(inputs, kwargs) {
        return inputs;
    }
    invokeCallHook(inputs, kwargs) {
        if (this._callHook != null) {
            this._callHook(inputs, kwargs);
        }
    }
    /**
     * Set call hook.
     * This is currently used for testing only.
     * @param callHook
     */
    setCallHook(callHook) {
        this._callHook = callHook;
    }
    /**
     * Clear call hook.
     * This is currently used for testing only.
     */
    clearCallHook() {
        this._callHook = null;
    }
    /**
     * Builds or executes a `Layer`'s logic.
     *
     * When called with `tf.Tensor`(s), execute the `Layer`'s computation and
     * return Tensor(s). For example:
     *
     * ```js
     * const denseLayer = tf.layers.dense({
     *   units: 1,
     *   kernelInitializer: 'zeros',
     *   useBias: false
     * });
     *
     * // Invoke the layer's apply() method with a `tf.Tensor` (with concrete
     * // numeric values).
     * const input = tf.ones([2, 2]);
     * const output = denseLayer.apply(input);
     *
     * // The output's value is expected to be [[0], [0]], due to the fact that
     * // the dense layer has a kernel initialized to all-zeros and does not have
     * // a bias.
     * output.print();
     * ```
     *
     * When called with `tf.SymbolicTensor`(s), this will prepare the layer for
     * future execution.  This entails internal book-keeping on shapes of
     * expected Tensors, wiring layers together, and initializing weights.
     *
     * Calling `apply` with `tf.SymbolicTensor`s are typically used during the
     * building of non-`tf.Sequential` models. For example:
     *
     * ```js
     * const flattenLayer = tf.layers.flatten();
     * const denseLayer = tf.layers.dense({units: 1});
     *
     * // Use tf.layers.input() to obtain a SymbolicTensor as input to apply().
     * const input = tf.input({shape: [2, 2]});
     * const output1 = flattenLayer.apply(input);
     *
     * // output1.shape is [null, 4]. The first dimension is the undetermined
     * // batch size. The second dimension comes from flattening the [2, 2]
     * // shape.
     * console.log(JSON.stringify(output1.shape));
     *
     * // The output SymbolicTensor of the flatten layer can be used to call
     * // the apply() of the dense layer:
     * const output2 = denseLayer.apply(output1);
     *
     * // output2.shape is [null, 1]. The first dimension is the undetermined
     * // batch size. The second dimension matches the number of units of the
     * // dense layer.
     * console.log(JSON.stringify(output2.shape));
     *
     * // The input and output can be used to construct a model that consists
     * // of the flatten and dense layers.
     * const model = tf.model({inputs: input, outputs: output2});
     * ```
     *
     * @param inputs a `tf.Tensor` or `tf.SymbolicTensor` or an Array of them.
     * @param kwargs Additional keyword arguments to be passed to `call()`.
     *
     * @return Output of the layer's `call` method.
     *
     * @exception ValueError error in case the layer is missing shape information
     *   for its `build` call.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    // Porting Note: This is a replacement for __call__() in Python.
    apply(inputs, kwargs) {
        kwargs = kwargs || {};
        this.assertNotDisposed();
        // Ensure inputs are all the same type.
        const inputsList = generic_utils.toList(inputs);
        const allAreSymbolic = checkAllSymbolic(inputs);
        const noneAreSymbolic = checkNoneSymbolic(inputs);
        if (allAreSymbolic === noneAreSymbolic) {
            throw new ValueError('Arguments to apply() must be all ' +
                'SymbolicTensors or all Tensors');
        }
        // TODO(michaelterry): nameScope() may not be necessary.
        return nameScope(this.name, () => {
            // Handle laying building (weight creating, input spec locking).
            if (!this.built) {
                /*
                  Throw exceptions in case the input is not compatible
                  with the inputSpec specified in the layer constructor.
                 */
                this.assertInputCompatibility(inputs);
                // Collect input shapes to build layer.
                const inputShapes = [];
                for (const xElem of generic_utils.toList(inputs)) {
                    inputShapes.push(xElem.shape);
                }
                this.build(generic_utils.singletonOrArray(inputShapes));
                this.built = true;
                // Load weights that were specified at layer instantiation.
                if (this.initialWeights) {
                    this.setWeights(this.initialWeights);
                }
                if (this._refCount === null && noneAreSymbolic) {
                    // The first use of this layer is a non-symbolic call, set ref count
                    // to 1 so the Layer can be properly disposed if its dispose() method
                    // is called.
                    this._refCount = 1;
                }
            }
            /*
              Throw exceptions in case the input is not compatible
              with the inputSpec set at build time.
            */
            this.assertInputCompatibility(inputs);
            // Handle mask propagation.
            // TODO(michaelterry): Mask propagation not currently implemented.
            // Actually call the layer, collecting output(s), mask(s), and shape(s).
            if (noneAreSymbolic) {
                let output = this.call(inputs, kwargs);
                // Apply masks to the output tensors if the layer supports it.
                if (this.supportsMasking) {
                    // TODO(mattsoulanille): pass the input tensors' masks to computeMask
                    this.setMaskMetadata(inputs, output);
                }
                // If the layer returns tensors from its inputs, unmodified,
                // we copy them to avoid loss of tensor metadata.
                const outputList = generic_utils.toList(output);
                const outputListCopy = [];
                // TODO(michaelterry): This copying may not be necessary given our eager
                // backend.
                for (let x of outputList) {
                    if (inputsList.indexOf(x) !== -1) {
                        x = x.clone();
                    }
                    outputListCopy.push(x);
                }
                output = generic_utils.singletonOrArray(outputListCopy);
                if (this.activityRegularizer != null) {
                    throw new NotImplementedError('Layer invocation in the presence of activity ' +
                        'regularizer(s) is not supported yet.');
                }
                // TODO(michaelterry): Call addInboundNode()?
                return output;
            }
            else {
                const inputShape = collectInputShape(inputs);
                const outputShape = this.computeOutputShape(inputShape);
                let output;
                const outputDType = guessOutputDType(inputs);
                this.warnOnIncompatibleInputShape(Array.isArray(inputs) ? inputShape[0] :
                    inputShape);
                if (outputShape != null && outputShape.length > 0 &&
                    Array.isArray(outputShape[0])) {
                    // We have multiple output shapes. Create multiple output tensors.
                    output = outputShape
                        .map((shape, index) => new SymbolicTensor(outputDType, shape, this, generic_utils.toList(inputs), kwargs, this.name, index));
                }
                else {
                    output = new SymbolicTensor(outputDType, outputShape, this, generic_utils.toList(inputs), kwargs, this.name);
                }
                /*
                  Add an inbound node to the layer, so that it keeps track
                  of the call and of all new variables created during the call.
                  This also updates the layer history of the output tensor(s).
                  If the input tensor(s) had no previous history,
                  this does nothing.
                */
                this.addInboundNode(inputs, output, null, null, inputShape, outputShape, kwargs);
                this._refCount++;
                if (this.activityRegularizer != null) {
                    throw new NotImplementedError('Layer invocation in the presence of activity ' +
                        'regularizer(s) is not supported yet.');
                }
                return output;
            }
        });
    }
    /**
     * Check compatibility between input shape and this layer's batchInputShape.
     *
     * Print warning if any incompatibility is found.
     *
     * @param inputShape Input shape to be checked.
     */
    warnOnIncompatibleInputShape(inputShape) {
        if (this.batchInputShape == null) {
            return;
        }
        else if (inputShape.length !== this.batchInputShape.length) {
            console.warn(`The rank of the input tensor provided (shape: ` +
                `${JSON.stringify(inputShape)}) does not match that of the ` +
                `batchInputShape (${JSON.stringify(this.batchInputShape)}) ` +
                `of the layer ${this.name}`);
        }
        else {
            let dimMismatch = false;
            this.batchInputShape.forEach((dimension, i) => {
                if (dimension != null && inputShape[i] != null &&
                    inputShape[i] !== dimension) {
                    dimMismatch = true;
                }
            });
            if (dimMismatch) {
                console.warn(`The shape of the input tensor ` +
                    `(${JSON.stringify(inputShape)}) does not ` +
                    `match the expectation of layer ${this.name}: ` +
                    `${JSON.stringify(this.batchInputShape)}`);
            }
        }
    }
    /**
     * Retrieves the output shape(s) of a layer.
     *
     * Only applicable if the layer has only one inbound node, or if all inbound
     * nodes have the same output shape.
     *
     * @returns Output shape or shapes.
     * @throws AttributeError: if the layer is connected to more than one incoming
     *   nodes.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    get outputShape() {
        if (this.inboundNodes == null || this.inboundNodes.length === 0) {
            throw new AttributeError(`The layer ${this.name} has never been called and thus has no ` +
                `defined output shape.`);
        }
        const allOutputShapes = [];
        for (const node of this.inboundNodes) {
            const shapeString = JSON.stringify(node.outputShapes);
            if (allOutputShapes.indexOf(shapeString) === -1) {
                allOutputShapes.push(shapeString);
            }
        }
        if (allOutputShapes.length === 1) {
            const outputShapes = this.inboundNodes[0].outputShapes;
            if (Array.isArray(outputShapes) && Array.isArray(outputShapes[0]) &&
                outputShapes.length === 1) {
                return outputShapes[0];
            }
            else {
                return outputShapes;
            }
        }
        else {
            throw new AttributeError(`The layer ${this.name} has multiple inbound nodes with different ` +
                `output shapes. Hence the notion of "output shape" is ill-defined ` +
                `for the layer.`);
            // TODO(cais): Implement getOutputShapeAt().
        }
    }
    /**
     * Counts the total number of numbers (e.g., float32, int32) in the
     * weights.
     *
     * @returns An integer count.
     * @throws RuntimeError: If the layer is not built yet (in which case its
     *   weights are not defined yet.)
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    countParams() {
        if (!this.built) {
            throw new RuntimeError(`You tried to call countParams() on ${this.name}, ` +
                `but the layer is not built yet. Build it first by calling ` +
                `build(batchInputShape).`);
        }
        return variable_utils.countParamsInWeights(this.weights);
    }
    /**
     * Creates the layer weights.
     *
     * Must be implemented on all layers that have weights.
     *
     * Called when apply() is called to construct the weights.
     *
     * @param inputShape A `Shape` or array of `Shape` (unused).
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    build(inputShape) {
        this.built = true;
    }
    /**
     * Returns the current values of the weights of the layer.
     *
     * @param trainableOnly Whether to get the values of only trainable weights.
     * @returns Weight values as an `Array` of `tf.Tensor`s.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    getWeights(trainableOnly = false) {
        return batchGetValue(trainableOnly ? this.trainableWeights : this.weights);
    }
    /**
     * Sets the weights of the layer, from Tensors.
     *
     * @param weights a list of Tensors. The number of arrays and their shape
     *   must match number of the dimensions of the weights of the layer (i.e.
     *   it should match the output of `getWeights`).
     *
     * @exception ValueError If the provided weights list does not match the
     *   layer's specifications.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    setWeights(weights) {
        tidy(() => {
            const params = this.weights;
            if (params.length !== weights.length) {
                // TODO(cais): Restore the following and use `providedWeights`, instead
                // of `weights` in the error message, once the deeplearn.js bug is
                // fixed: https://github.com/PAIR-code/deeplearnjs/issues/498 const
                // providedWeights = JSON.stringify(weights).slice(0, 50);
                throw new ValueError(`You called setWeights(weights) on layer "${this.name}" ` +
                    `with a weight list of length ${weights.length}, ` +
                    `but the layer was expecting ${params.length} weights. ` +
                    `Provided weights: ${weights}...`);
            }
            if (params.length === 0) {
                return;
            }
            const weightValueTuples = [];
            const paramValues = batchGetValue(params);
            for (let i = 0; i < paramValues.length; ++i) {
                const pv = paramValues[i];
                const p = params[i];
                const w = weights[i];
                if (!util.arraysEqual(pv.shape, w.shape)) {
                    throw new ValueError(`Layer weight shape ${pv.shape} ` +
                        `not compatible with provided weight shape ${w.shape}`);
                }
                weightValueTuples.push([p, w]);
            }
            batchSetValue(weightValueTuples);
        });
    }
    /**
     * Adds a weight variable to the layer.
     *
     * @param name Name of the new weight variable.
     * @param shape The shape of the weight.
     * @param dtype The dtype of the weight.
     * @param initializer An initializer instance.
     * @param regularizer A regularizer instance.
     * @param trainable Whether the weight should be trained via backprop or not
     *   (assuming that the layer itself is also trainable).
     * @param constraint An optional trainable.
     * @return The created weight variable.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    addWeight(name, shape, dtype, initializer, regularizer, trainable, constraint, getInitializerFunc) {
        // Reject duplicate weight names.
        if (this._addedWeightNames.indexOf(name) !== -1) {
            throw new ValueError(`Duplicate weight name ${name} for layer ${this.name}`);
        }
        this._addedWeightNames.push(name);
        if (dtype == null) {
            dtype = 'float32';
        }
        if (this.fastWeightInitDuringBuild) {
            initializer = getInitializerFunc != null ? getInitializerFunc() :
                getInitializer('zeros');
        }
        const initValue = initializer.apply(shape, dtype);
        const weight = new LayerVariable(initValue, dtype, name, trainable, constraint);
        initValue.dispose();
        // Request backend not to dispose the weights of the model on scope() exit.
        if (regularizer != null) {
            this.addLoss(() => regularizer.apply(weight.read()));
        }
        if (trainable == null) {
            trainable = true;
        }
        if (trainable) {
            this._trainableWeights.push(weight);
        }
        else {
            this._nonTrainableWeights.push(weight);
        }
        return weight;
    }
    /**
     * Set the fast-weight-initialization flag.
     *
     * In cases where the initialized weight values will be immediately
     * overwritten by loaded weight values during model loading, setting
     * the flag to `true` saves unnecessary calls to potentially expensive
     * initializers and speeds up the loading process.
     *
     * @param value Target value of the flag.
     */
    setFastWeightInitDuringBuild(value) {
        this.fastWeightInitDuringBuild = value;
    }
    /**
     * Add losses to the layer.
     *
     * The loss may potentially be conditional on some inputs tensors,
     * for instance activity losses are conditional on the layer's inputs.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    addLoss(losses) {
        if (losses == null || Array.isArray(losses) && losses.length === 0) {
            return;
        }
        // Update this.losses
        losses = generic_utils.toList(losses);
        if (this._losses !== undefined && this._losses !== null) {
            this.losses.push(...losses);
        }
    }
    /**
     * Computes the output shape of the layer.
     *
     * Assumes that the layer will be built to match that input shape provided.
     *
     * @param inputShape A shape (tuple of integers) or a list of shape tuples
     *   (one per output tensor of the layer). Shape tuples can include null for
     *   free dimensions, instead of an integer.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    computeOutputShape(inputShape) {
        return inputShape;
    }
    /**
     * Computes an output mask tensor.
     *
     * @param inputs Tensor or list of tensors.
     * @param mask Tensor or list of tensors.
     *
     * @return null or a tensor (or list of tensors, one per output tensor of the
     * layer).
     */
    computeMask(inputs, mask) {
        if (!this.supportsMasking) {
            if (mask != null) {
                if (Array.isArray(mask)) {
                    mask.forEach(maskElement => {
                        if (maskElement != null) {
                            throw new TypeError(`Layer ${this.name} does not support masking, ` +
                                'but was passed an inputMask.');
                        }
                    });
                }
                else {
                    throw new TypeError(`Layer ${this.name} does not support masking, ` +
                        'but was passed an inputMask.');
                }
            }
            // masking not explicitly supported: return null as mask
            return null;
        }
        // if masking is explictly supported, by default
        // carry over the input mask
        return mask;
    }
    setMaskMetadata(inputs, outputs, previousMask) {
        if (!this.supportsMasking) {
            return;
        }
        const outputMasks = this.computeMask(inputs, previousMask);
        if (outputs instanceof Array && outputMasks instanceof Array) {
            if (outputs.length !== outputMasks.length) {
                throw new Error(`${this.name} outputs ${outputs.length} tensors `
                    + `but ${outputMasks.length} masks for those tensors`);
            }
            for (let i = 0; i < outputs.length; i++) {
                outputs[i].kerasMask = outputMasks[i];
            }
        }
        else if (outputMasks instanceof Array) {
            throw new Error(`{this.name} outputs a single tensor `
                + `but ${outputMasks.length} masks`);
        }
        else if (outputs instanceof Array) {
            throw new Error(`{this.name} outputs ${outputs.length} tensors `
                + `but only one mask`);
        }
        else {
            outputs.kerasMask = outputMasks;
        }
    }
    /**
     * Internal method to create an inbound node for the layer.
     *
     * @param inputTensors List of input tensors.
     * @param outputTensors List of output tensors.
     * @param inputMasks List of input masks (a mask can be a tensor, or null).
     * @param outputMasks List of output masks (a mask can be a tensor, or null).
     * @param inputShapes List of input shape tuples.
     * @param outputShapes List of output shape tuples.
     * @param kwargs Dictionary of keyword arguments that were passed to the
     *   `call` method of the layer at the call that created the node.
     */
    addInboundNode(inputTensors, outputTensors, inputMasks, outputMasks, inputShapes, outputShapes, kwargs = null) {
        const inputTensorList = generic_utils.toList(inputTensors);
        outputTensors = generic_utils.toList(outputTensors);
        inputMasks = generic_utils.toList(inputMasks);
        outputMasks = generic_utils.toList(outputMasks);
        inputShapes = types_utils.normalizeShapeList(inputShapes);
        outputShapes = types_utils.normalizeShapeList(outputShapes);
        // Collect input tensor(s) coordinates.
        const inboundLayers = [];
        const nodeIndices = [];
        const tensorIndices = [];
        for (const x of inputTensorList) {
            /*
             * TODO(michaelterry): Keras adds this value to tensors; it's not
             * clear whether we'll use this or not.
             */
            inboundLayers.push(x.sourceLayer);
            nodeIndices.push(x.nodeIndex);
            tensorIndices.push(x.tensorIndex);
        }
        // Create node, add it to inbound nodes.
        // (This call has side effects.)
        // tslint:disable-next-line:no-unused-expression
        new Node({
            outboundLayer: this,
            inboundLayers,
            nodeIndices,
            tensorIndices,
            inputTensors: inputTensorList,
            outputTensors,
            inputMasks,
            outputMasks,
            inputShapes,
            outputShapes
        }, kwargs);
        // Update tensor history
        for (let i = 0; i < outputTensors.length; i++) {
            // TODO(michaelterry: _uses_learning_phase not tracked.
            outputTensors[i].sourceLayer = this;
            outputTensors[i].nodeIndex = this.inboundNodes.length - 1;
            outputTensors[i].tensorIndex = i;
        }
    }
    /**
     * Returns the config of the layer.
     *
     * A layer config is a TS dictionary (serializable)
     * containing the configuration of a layer.
     * The same layer can be reinstantiated later
     * (without its trained weights) from this configuration.
     *
     * The config of a layer does not include connectivity
     * information, nor the layer class name.  These are handled
     * by 'Container' (one layer of abstraction above).
     *
     * Porting Note: The TS dictionary follows TS naming standards for
     * keys, and uses tfjs-layers type-safe Enums.  Serialization methods
     * should use a helper function to convert to the pythonic storage
     * standard. (see serialization_utils.convertTsToPythonic)
     *
     * @returns TS dictionary of configuration.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    getConfig() {
        const config = { name: this.name, trainable: this.trainable };
        if (this.batchInputShape != null) {
            config['batchInputShape'] = this.batchInputShape;
        }
        if (this.dtype != null) {
            config['dtype'] = this.dtype;
        }
        return config;
    }
    /**
     * Dispose the weight variables that this Layer instance holds.
     *
     * @returns {number} Number of disposed variables.
     */
    disposeWeights() {
        this.weights.forEach(weight => weight.dispose());
        return this.weights.length;
    }
    assertNotDisposed() {
        if (this._refCount === 0) {
            throw new Error(`Layer '${this.name}' is already disposed.`);
        }
    }
    /**
     * Attempt to dispose layer's weights.
     *
     * This method decreases the reference count of the Layer object by 1.
     *
     * A Layer is reference-counted. Its reference count is incremented by 1
     * the first item its `apply()` method is called and when it becomes a part
     * of a new `Node` (through calling the `apply()` method on a
     * `tf.SymbolicTensor`).
     *
     * If the reference count of a Layer becomes 0, all the weights will be
     * disposed and the underlying memory (e.g., the textures allocated in WebGL)
     * will be freed.
     *
     * Note: If the reference count is greater than 0 after the decrement, the
     * weights of the Layer will *not* be disposed.
     *
     * After a Layer is disposed, it cannot be used in calls such as `apply()`,
     * `getWeights()` or `setWeights()` anymore.
     *
     * @returns A DisposeResult Object with the following fields:
     *   - refCountAfterDispose: The reference count of the Container after this
     *     `dispose()` call.
     *   - numDisposedVariables: Number of `tf.Variable`s (i.e., weights) disposed
     *     during this `dispose()` call.
     * @throws {Error} If the layer is not built yet, or if the layer has already
     *   been disposed.
     *
     * @doc {heading: 'Models', 'subheading': 'Classes'}
     */
    dispose() {
        if (!this.built) {
            throw new Error(`Cannot dispose Layer ${this.name} because it has not been ` +
                `built yet.`);
        }
        if (this._refCount === null) {
            throw new Error(`Cannot dispose Layer ${this.name} because it has not been used ` +
                `yet.`);
        }
        this.assertNotDisposed();
        let numDisposedVariables = 0;
        if (--this._refCount === 0) {
            numDisposedVariables = this.disposeWeights();
        }
        return { refCountAfterDispose: this._refCount, numDisposedVariables };
    }
}
/**
 * Collects the input shape(s) of a list of `tf.Tensor`s or
 * `tf.SymbolicTensor`s.
 *
 * TODO(michaelterry): Update PyKeras docs (backport).
 *
 * @param inputTensors List of input tensors (or single input tensor).
 *
 * @return List of shape tuples (or single tuple), one tuple per input.
 */
function collectInputShape(inputTensors) {
    inputTensors =
        generic_utils.toList(inputTensors);
    const shapes = [];
    for (const x of inputTensors) {
        shapes.push(x.shape);
    }
    return generic_utils.singletonOrArray(shapes);
}
/**
 * Guesses output dtype based on inputs.
 *
 * At present, just returns 'float32' for any input.
 *
 * @param inputTensors List of input tensors (or single input tensor).
 *
 * @return The guessed DType. At present, always returns 'float32'.
 */
function guessOutputDType(inputTensors) {
    return 'float32';
}
/**
 * Returns the list of input tensors necessary to compute `tensor`.
 *
 * Output will always be a list of tensors (potentially with 1 element).
 *
 * @param tensor The tensor to start from.
 * @param layer Origin layer of the tensor.
 * @param nodeIndex Origin node index of the tensor.
 *
 * @return Array of input tensors.
 */
export function getSourceInputs(tensor, layer, nodeIndex) {
    if (layer == null || (nodeIndex != null && nodeIndex > 0)) {
        layer = tensor.sourceLayer;
        nodeIndex = tensor.nodeIndex;
    }
    if (layer.inboundNodes.length === 0) {
        return [tensor];
    }
    else {
        const node = layer.inboundNodes[nodeIndex];
        if (node.inboundLayers.length === 0) {
            return node.inputTensors;
        }
        else {
            const sourceTensors = [];
            for (let i = 0; i < node.inboundLayers.length; i++) {
                const x = node.inputTensors[i];
                const layer = node.inboundLayers[i];
                const nodeIndex = node.nodeIndices[i];
                const previousSources = getSourceInputs(x, layer, nodeIndex);
                // Avoid input redundancy.
                for (const x of previousSources) {
                    if (sourceTensors.indexOf(x) === -1) {
                        sourceTensors.push(x);
                    }
                }
            }
            return sourceTensors;
        }
    }
}
function checkAllSymbolic(tensors) {
    let allAreSymbolic = true;
    for (const tensor of generic_utils.toList(tensors)) {
        if (!(tensor instanceof SymbolicTensor)) {
            allAreSymbolic = false;
            break;
        }
    }
    return allAreSymbolic;
}
function checkNoneSymbolic(tensors) {
    let noneAreSymbolic = true;
    for (const tensor of generic_utils.toList(tensors)) {
        if (tensor instanceof SymbolicTensor) {
            noneAreSymbolic = false;
            break;
        }
    }
    return noneAreSymbolic;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidG9wb2xvZ3kuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZW5naW5lL3RvcG9sb2d5LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUgsK0NBQStDO0FBRS9DLE9BQU8sRUFBbUIsYUFBYSxFQUFVLElBQUksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUUxRixPQUFPLEVBQUMscUJBQXFCLEVBQUUsTUFBTSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixFQUFFLG1CQUFtQixFQUFFLFNBQVMsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUU5RSxPQUFPLEVBQUMsY0FBYyxFQUFFLG1CQUFtQixFQUFFLFlBQVksRUFBRSxVQUFVLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDeEYsT0FBTyxFQUFDLGNBQWMsRUFBYyxNQUFNLGlCQUFpQixDQUFDO0FBSTVELE9BQU8sS0FBSyxhQUFhLE1BQU0sd0JBQXdCLENBQUM7QUFDeEQsT0FBTyxLQUFLLFdBQVcsTUFBTSxzQkFBc0IsQ0FBQztBQUNwRCxPQUFPLEtBQUssY0FBYyxNQUFNLHlCQUF5QixDQUFDO0FBQzFELE9BQU8sRUFBQyxhQUFhLEVBQUUsYUFBYSxFQUFFLGFBQWEsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQXVCekU7Ozs7Ozs7O0dBUUc7QUFDSCxNQUFNLE9BQU8sU0FBUztJQWNwQixZQUFZLElBQW1CO1FBQzdCLElBQUksQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQztRQUN4QixJQUFJLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUM7UUFDeEI7OztVQUdFO1FBQ0YsSUFBSSxJQUFJLENBQUMsS0FBSyxJQUFJLElBQUksRUFBRTtZQUN0QixJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFD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QUM7QUFDekIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qIE9yaWdpbmFsIHNvdXJjZToga2VyYXMvZW5naW5lL3RvcG9sb2d5LnB5ICovXG5cbmltcG9ydCB7RGF0YVR5cGUsIFNjYWxhciwgc2VyaWFsaXphdGlvbiwgVGVuc29yLCB0aWR5LCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldE5leHRVbmlxdWVUZW5zb3JJZCwgZ2V0VWlkfSBmcm9tICcuLi9iYWNrZW5kL3N0YXRlJztcbmltcG9ydCB7Z2V0U2NvcGVkVGVuc29yTmFtZSwgZ2V0VW5pcXVlVGVuc29yTmFtZSwgbmFtZVNjb3BlfSBmcm9tICcuLi9jb21tb24nO1xuaW1wb3J0IHtDb25zdHJhaW50fSBmcm9tICcuLi9jb25zdHJhaW50cyc7XG5pbXBvcnQge0F0dHJpYnV0ZUVycm9yLCBOb3RJbXBsZW1lbnRlZEVycm9yLCBSdW50aW1lRXJyb3IsIFZhbHVlRXJyb3J9IGZyb20gJy4uL2Vycm9ycyc7XG5pbXBvcnQge2dldEluaXRpYWxpemVyLCBJbml0aWFsaXplcn0gZnJvbSAnLi4vaW5pdGlhbGl6ZXJzJztcbmltcG9ydCB7U2hhcGV9IGZyb20gJy4uL2tlcmFzX2Zvcm1hdC9jb21tb24nO1xuaW1wb3J0IHtSZWd1bGFyaXplcn0gZnJvbSAnLi4vcmVndWxhcml6ZXJzJztcbmltcG9ydCB7S3dhcmdzLCBSZWd1bGFyaXplckZufSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyBnZW5lcmljX3V0aWxzIGZyb20gJy4uL3V0aWxzL2dlbmVyaWNfdXRpbHMnO1xuaW1wb3J0ICogYXMgdHlwZXNfdXRpbHMgZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuaW1wb3J0ICogYXMgdmFyaWFibGVfdXRpbHMgZnJvbSAnLi4vdXRpbHMvdmFyaWFibGVfdXRpbHMnO1xuaW1wb3J0IHtiYXRjaEdldFZhbHVlLCBiYXRjaFNldFZhbHVlLCBMYXllclZhcmlhYmxlfSBmcm9tICcuLi92YXJpYWJsZXMnO1xuXG4vLyBUT0RPKG1pY2hhZWx0ZXJyeSk6IFRoaXMgaXMgYSBzdHViIHVudGlsIGl0J3MgZGVmaW5lZC5cbmV4cG9ydCB0eXBlIE9wID0gKHg6IExheWVyVmFyaWFibGUpID0+IExheWVyVmFyaWFibGU7XG5cbi8qKlxuICogQ29uc3RydWN0b3IgYXJndW1lbnRzIGZvciBJbnB1dFNwZWMuXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgSW5wdXRTcGVjQXJncyB7XG4gIC8qKiBFeHBlY3RlZCBkYXRhdHlwZSBvZiB0aGUgaW5wdXQuICovXG4gIGR0eXBlPzogRGF0YVR5cGU7XG4gIC8qKiBFeHBlY3RlZCBzaGFwZSBvZiB0aGUgaW5wdXQgKG1heSBpbmNsdWRlIG51bGwgZm9yIHVuY2hlY2tlZCBheGVzKS4gKi9cbiAgc2hhcGU/OiBTaGFwZTtcbiAgLyoqIEV4cGVjdGVkIHJhbmsgb2YgdGhlIGlucHV0LiAqL1xuICBuZGltPzogbnVtYmVyO1xuICAvKiogTWF4aW11bSByYW5rIG9mIHRoZSBpbnB1dC4gKi9cbiAgbWF4TkRpbT86IG51bWJlcjtcbiAgLyoqIE1pbmltdW0gcmFuayBvZiB0aGUgaW5wdXQuICovXG4gIG1pbk5EaW0/OiBudW1iZXI7XG4gIC8qKiBEaWN0aW9uYXJ5IG1hcHBpbmcgaW50ZWdlciBheGVzIHRvIGEgc3BlY2lmaWMgZGltZW5zaW9uIHZhbHVlLiAqL1xuICBheGVzPzoge1theGlzOiBudW1iZXJdOiBudW1iZXJ9O1xufVxuXG4vKipcbiAqIFNwZWNpZmllcyB0aGUgbmRpbSwgZHR5cGUgYW5kIHNoYXBlIG9mIGV2ZXJ5IGlucHV0IHRvIGEgbGF5ZXIuXG4gKlxuICogRXZlcnkgbGF5ZXIgc2hvdWxkIGV4cG9zZSAoaWYgYXBwcm9wcmlhdGUpIGFuIGBpbnB1dFNwZWNgIGF0dHJpYnV0ZTpcbiAqIGEgbGlzdCBvZiBpbnN0YW5jZXMgb2YgSW5wdXRTcGVjIChvbmUgcGVyIGlucHV0IHRlbnNvcikuXG4gKlxuICogQSBudWxsIGVudHJ5IGluIGEgc2hhcGUgaXMgY29tcGF0aWJsZSB3aXRoIGFueSBkaW1lbnNpb24sXG4gKiBhIG51bGwgc2hhcGUgaXMgY29tcGF0aWJsZSB3aXRoIGFueSBzaGFwZS5cbiAqL1xuZXhwb3J0IGNsYXNzIElucHV0U3BlYyB7XG4gIC8qKiBFeHBlY3RlZCBkYXRhdHlwZSBvZiB0aGUgaW5wdXQuICovXG4gIGR0eXBlPzogRGF0YVR5cGU7XG4gIC8qKiBFeHBlY3RlZCBzaGFwZSBvZiB0aGUgaW5wdXQgKG1heSBpbmNsdWRlIG51bGwgZm9yIHVuY2hlY2tlZCBheGVzKS4gKi9cbiAgc2hhcGU/OiBTaGFwZTtcbiAgLyoqIEV4cGVjdGVkIHJhbmsgb2YgdGhlIGlucHV0LiAqL1xuICBuZGltPzogbnVtYmVyO1xuICAvKiogTWF4aW11bSByYW5rIG9mIHRoZSBpbnB1dC4gKi9cbiAgbWF4TkRpbT86IG51bWJlcjtcbiAgLyoqIE1pbmltdW0gcmFuayBvZiB0aGUgaW5wdXQuICovXG4gIG1pbk5EaW0/OiBudW1iZXI7XG4gIC8qKiBEaWN0aW9uYXJ5IG1hcHBpbmcgaW50ZWdlciBheGVzIHRvIGEgc3BlY2lmaWMgZGltZW5zaW9uIHZhbHVlLiAqL1xuICBheGVzPzoge1theGlzOiBudW1iZXJdOiBudW1iZXJ9O1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IElucHV0U3BlY0FyZ3MpIHtcbiAgICB0aGlzLmR0eXBlID0gYXJncy5kdHlwZTtcbiAgICB0aGlzLnNoYXBlID0gYXJncy5zaGFwZTtcbiAgICAvKlxuICAgICAgVE9ETyhtaWNoYWVsdGVycnkpOiBDb3VsZCB0aHJvdyBlcnJvciBpZiBuZGltIGFuZCBzaGFwZSBhcmUgYm90aCBkZWZpbmVkXG4gICAgICAgICh0aGVuIGJhY2twb3J0KS5cbiAgICAqL1xuICAgIGlmIChhcmdzLnNoYXBlICE9IG51bGwpIHtcbiAgICAgIHRoaXMubmRpbSA9IGFyZ3Muc2hhcGUubGVuZ3RoO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLm5kaW0gPSBhcmdzLm5kaW07XG4gICAgfVxuICAgIHRoaXMubWF4TkRpbSA9IGFyZ3MubWF4TkRpbTtcbiAgICB0aGlzLm1pbk5EaW0gPSBhcmdzLm1pbk5EaW07XG4gICAgdGhpcy5heGVzID0gYXJncy5heGVzIHx8IHt9O1xuICB9XG59XG5cbi8qKlxuICogYHRmLlN5bWJvbGljVGVuc29yYCBpcyBhIHBsYWNlaG9sZGVyIGZvciBhIFRlbnNvciB3aXRob3V0IGFueSBjb25jcmV0ZSB2YWx1ZS5cbiAqXG4gKiBUaGV5IGFyZSBtb3N0IG9mdGVuIGVuY291bnRlcmVkIHdoZW4gYnVpbGRpbmcgYSBncmFwaCBvZiBgTGF5ZXJgcyBmb3IgYVxuICogYHRmLkxheWVyc01vZGVsYCBhbmQgdGhlIGlucHV0IGRhdGEncyBzaGFwZSwgYnV0IG5vdCB2YWx1ZXMgYXJlIGtub3duLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCAnc3ViaGVhZGluZyc6ICdDbGFzc2VzJ31cbiAqL1xuZXhwb3J0IGNsYXNzIFN5bWJvbGljVGVuc29yIHtcbiAgLyogQSB1bmlxdWUgSUQgZm9yIHRoZSB0ZW5zb3IgdG8gYmUgYWJsZSB0byBkaWZmZXJlbnRpYXRlIHRlbnNvcnMuICovXG4gIHJlYWRvbmx5IGlkOiBudW1iZXI7XG4gIC8vIFRoZSBmdWxseSBzY29wZWQgbmFtZSBvZiB0aGlzIFZhcmlhYmxlLCBpbmNsdWRpbmcgYSB1bmlxdWUgc3VmZml4IGlmIG5lZWRlZFxuICByZWFkb25seSBuYW1lOiBzdHJpbmc7XG4gIC8vIFRoZSBvcmlnaW5hbGx5IHJlcXVlc3RlZCBmdWxseSBzY29wZWQgbmFtZSBvZiB0aGlzIFZhcmlhYmxlLCBub3QgaW5jbHVkaW5nXG4gIC8vIGFueSB1bmlxdWUgc3VmZml4LiAgVGhpcyBtYXkgYmUgbmVlZGVkIHdoZW4gcmVzdG9yaW5nIHdlaWdodHMgYmVjYXVzZSB0aGlzXG4gIC8vIG9yaWdpbmFsIG5hbWUgaXMgdXNlZCBhcyBhIGtleS5cbiAgcmVhZG9ubHkgb3JpZ2luYWxOYW1lPzogc3RyaW5nO1xuICAvKipcbiAgICogUmFuay9kaW1lbnNpb25hbGl0eSBvZiB0aGUgdGVuc29yLlxuICAgKi9cbiAgcmVhZG9ubHkgcmFuazogbnVtYmVyO1xuICAvKipcbiAgICogUmVwbGFjZW1lbnQgZm9yIF9rZXJhc19oaXN0b3J5LlxuICAgKi9cbiAgbm9kZUluZGV4OiBudW1iZXI7XG4gIC8qKlxuICAgKiBSZXBsYWNlbWVudCBmb3IgX2tlcmFzX2hpc3RvcnkuXG4gICAqL1xuICB0ZW5zb3JJbmRleDogbnVtYmVyO1xuXG4gIC8qKlxuICAgKlxuICAgKiBAcGFyYW0gZHR5cGVcbiAgICogQHBhcmFtIHNoYXBlXG4gICAqIEBwYXJhbSBzb3VyY2VMYXllciBUaGUgTGF5ZXIgdGhhdCBwcm9kdWNlZCB0aGlzIHN5bWJvbGljIHRlbnNvci5cbiAgICogQHBhcmFtIGlucHV0cyBUaGUgaW5wdXRzIHBhc3NlZCB0byBzb3VyY2VMYXllcidzIF9fY2FsbF9fKCkgbWV0aG9kLlxuICAgKiBAcGFyYW0gbm9kZUluZGV4XG4gICAqIEBwYXJhbSB0ZW5zb3JJbmRleFxuICAgKiBAcGFyYW0gY2FsbEFyZ3MgVGhlIGtleXdvcmQgYXJndW1lbnRzIHBhc3NlZCB0byB0aGUgX19jYWxsX18oKSBtZXRob2QuXG4gICAqIEBwYXJhbSBuYW1lXG4gICAqIEBwYXJhbSBvdXRwdXRUZW5zb3JJbmRleCBUaGUgaW5kZXggb2YgdGhpcyB0ZW5zb3IgaW4gdGhlIGxpc3Qgb2Ygb3V0cHV0c1xuICAgKiAgIHJldHVybmVkIGJ5IGFwcGx5KCkuXG4gICAqL1xuICBjb25zdHJ1Y3RvcihcbiAgICAgIHJlYWRvbmx5IGR0eXBlOiBEYXRhVHlwZSwgcmVhZG9ubHkgc2hhcGU6IFNoYXBlLFxuICAgICAgcHVibGljIHNvdXJjZUxheWVyOiBMYXllciwgcmVhZG9ubHkgaW5wdXRzOiBTeW1ib2xpY1RlbnNvcltdLFxuICAgICAgcmVhZG9ubHkgY2FsbEFyZ3M6IEt3YXJncywgbmFtZT86IHN0cmluZyxcbiAgICAgIHJlYWRvbmx5IG91dHB1dFRlbnNvckluZGV4PzogbnVtYmVyKSB7XG4gICAgdGhpcy5pZCA9IGdldE5leHRVbmlxdWVUZW5zb3JJZCgpO1xuICAgIGlmIChuYW1lICE9IG51bGwpIHtcbiAgICAgIHRoaXMub3JpZ2luYWxOYW1lID0gZ2V0U2NvcGVkVGVuc29yTmFtZShuYW1lKTtcbiAgICAgIHRoaXMubmFtZSA9IGdldFVuaXF1ZVRlbnNvck5hbWUodGhpcy5vcmlnaW5hbE5hbWUpO1xuICAgIH1cbiAgICB0aGlzLnJhbmsgPSBzaGFwZS5sZW5ndGg7XG4gIH1cbn1cblxuLyoqXG4gKiBDb25zdHJ1Y3RvciBhcmd1bWVudHMgZm9yIE5vZGUuXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTm9kZUFyZ3Mge1xuICAvKipcbiAgICogVGhlIGxheWVyIHRoYXQgdGFrZXMgYGlucHV0VGVuc29yc2AgYW5kIHR1cm5zIHRoZW0gaW50byBgb3V0cHV0VGVuc29yc2AuXG4gICAqICh0aGUgbm9kZSBnZXRzIGNyZWF0ZWQgd2hlbiB0aGUgYGNhbGxgIG1ldGhvZCBvZiB0aGUgbGF5ZXIgaXMgY2FsbGVkKS5cbiAgICovXG4gIG91dGJvdW5kTGF5ZXI6IExheWVyO1xuICAvKipcbiAgICogQSBsaXN0IG9mIGxheWVycywgdGhlIHNhbWUgbGVuZ3RoIGFzIGBpbnB1dFRlbnNvcnNgLCB0aGUgbGF5ZXJzIGZyb20gd2hlcmVcbiAgICogYGlucHV0VGVuc29yc2Agb3JpZ2luYXRlLlxuICAgKi9cbiAgaW5ib3VuZExheWVyczogTGF5ZXJbXTtcbiAgLyoqXG4gICAqIEEgbGlzdCBvZiBpbnRlZ2VycywgdGhlIHNhbWUgbGVuZ3RoIGFzIGBpbmJvdW5kTGF5ZXJzYC4gYG5vZGVJbmRpY2VzW2ldYCBpc1xuICAgKiB0aGUgb3JpZ2luIG5vZGUgb2YgYGlucHV0VGVuc29yc1tpXWAgKG5lY2Vzc2FyeSBzaW5jZSBlYWNoIGluYm91bmQgbGF5ZXJcbiAgICogbWlnaHQgaGF2ZSBzZXZlcmFsIG5vZGVzLCBlLmcuIGlmIHRoZSBsYXllciBpcyBiZWluZyBzaGFyZWQgd2l0aCBhXG4gICAqIGRpZmZlcmVudCBkYXRhIHN0cmVhbSkuXG4gICAqL1xuICBub2RlSW5kaWNlczogbnVtYmVyW107XG4gIC8qKlxuICAgKiBBIGxpc3Qgb2YgaW50ZWdlcnMsIHRoZSBzYW1lIGxlbmd0aCBhcyBgaW5ib3VuZExheWVyc2AuIGB0ZW5zb3JJbmRpY2VzW2ldYFxuICAgKiBpcyB0aGUgaW5kZXggb2YgYGlucHV0VGVuc29yc1tpXWAgd2l0aGluIHRoZSBvdXRwdXQgb2YgdGhlIGluYm91bmQgbGF5ZXJcbiAgICogKG5lY2Vzc2FyeSBzaW5jZSBlYWNoIGluYm91bmQgbGF5ZXIgbWlnaHQgaGF2ZSBtdWx0aXBsZSB0ZW5zb3Igb3V0cHV0cyxcbiAgICogd2l0aCBlYWNoIG9uZSBiZWluZyBpbmRlcGVuZGVudGx5IG1hbmlwdWxhYmxlKS5cbiAgICovXG4gIHRlbnNvckluZGljZXM6IG51bWJlcltdO1xuICAvKiogTGlzdCBvZiBpbnB1dCB0ZW5zb3JzLiAqL1xuICBpbnB1dFRlbnNvcnM6IFN5bWJvbGljVGVuc29yW107XG4gIC8qKiBMaXN0IG9mIG91dHB1dCB0ZW5zb3JzLiAqL1xuICBvdXRwdXRUZW5zb3JzOiBTeW1ib2xpY1RlbnNvcltdO1xuICAvKiogTGlzdCBvZiBpbnB1dCBtYXNrcyAoYSBtYXNrIGNhbiBiZSBhIHRlbnNvciwgb3IgbnVsbCkuICovXG4gIGlucHV0TWFza3M6IFRlbnNvcltdO1xuICAvKiogTGlzdCBvZiBvdXRwdXQgbWFza3MgKGEgbWFzayBjYW4gYmUgYSB0ZW5zb3IsIG9yIG51bGwpLiAqL1xuICBvdXRwdXRNYXNrczogVGVuc29yW107XG4gIC8qKiBMaXN0IG9mIGlucHV0IHNoYXBlIHR1cGxlcy4gKi9cbiAgaW5wdXRTaGFwZXM6IFNoYXBlfFNoYXBlW107XG4gIC8qKiBMaXN0IG9mIG91dHB1dCBzaGFwZSB0dXBsZXMuICovXG4gIG91dHB1dFNoYXBlczogU2hhcGV8U2hhcGVbXTtcbn1cblxuLyoqXG4gKiBUaGUgdHlwZSBvZiB0aGUgcmV0dXJuIHZhbHVlIG9mIExheWVyLmRpc3Bvc2UoKSBhbmQgQ29udGFpbmVyLmRpc3Bvc2UoKS5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBEaXNwb3NlUmVzdWx0IHtcbiAgLyoqXG4gICAqIFJlZmVyZW5jZSBjb3VudCBhZnRlciB0aGUgZGlzcG9zZSBjYWxsLlxuICAgKi9cbiAgcmVmQ291bnRBZnRlckRpc3Bvc2U6IG51bWJlcjtcblxuICAvKipcbiAgICogTnVtYmVyIG9mIHZhcmlhYmxlcyBkaXNwb3NlIGluIHRoaXMgZGlzcG9zZSBjYWxsLlxuICAgKi9cbiAgbnVtRGlzcG9zZWRWYXJpYWJsZXM6IG51bWJlcjtcbn1cblxubGV0IF9uZXh0Tm9kZUlEID0gMDtcblxuLyoqXG4gKiBBIGBOb2RlYCBkZXNjcmliZXMgdGhlIGNvbm5lY3Rpdml0eSBiZXR3ZWVuIHR3byBsYXllcnMuXG4gKlxuICogRWFjaCB0aW1lIGEgbGF5ZXIgaXMgY29ubmVjdGVkIHRvIHNvbWUgbmV3IGlucHV0LFxuICogYSBub2RlIGlzIGFkZGVkIHRvIGBsYXllci5pbmJvdW5kTm9kZXNgLlxuICpcbiAqIEVhY2ggdGltZSB0aGUgb3V0cHV0IG9mIGEgbGF5ZXIgaXMgdXNlZCBieSBhbm90aGVyIGxheWVyLFxuICogYSBub2RlIGlzIGFkZGVkIHRvIGBsYXllci5vdXRib3VuZE5vZGVzYC5cbiAqXG4gKiBgbm9kZUluZGljZXNgIGFuZCBgdGVuc29ySW5kaWNlc2AgYXJlIGJhc2ljYWxseSBmaW5lLWdyYWluZWQgY29vcmRpbmF0ZXNcbiAqIGRlc2NyaWJpbmcgdGhlIG9yaWdpbiBvZiB0aGUgYGlucHV0VGVuc29yc2AsIHZlcmlmeWluZyB0aGUgZm9sbG93aW5nOlxuICpcbiAqIGBpbnB1dFRlbnNvcnNbaV0gPT1cbiAqIGluYm91bmRMYXllcnNbaV0uaW5ib3VuZE5vZGVzW25vZGVJbmRpY2VzW2ldXS5vdXRwdXRUZW5zb3JzW1xuICogICB0ZW5zb3JJbmRpY2VzW2ldXWBcbiAqXG4gKiBBIG5vZGUgZnJvbSBsYXllciBBIHRvIGxheWVyIEIgaXMgYWRkZWQgdG86XG4gKiAgICAgQS5vdXRib3VuZE5vZGVzXG4gKiAgICAgQi5pbmJvdW5kTm9kZXNcbiAqL1xuZXhwb3J0IGNsYXNzIE5vZGUge1xuICAvKipcbiAgICogVGhlIGxheWVyIHRoYXQgdGFrZXMgYGlucHV0VGVuc29yc2AgYW5kIHR1cm5zIHRoZW0gaW50byBgb3V0cHV0VGVuc29yc2BcbiAgICogKHRoZSBub2RlIGdldHMgY3JlYXRlZCB3aGVuIHRoZSBgY2FsbGAgbWV0aG9kIG9mIHRoZSBsYXllciBpcyBjYWxsZWQpLlxuICAgKi9cbiAgb3V0Ym91bmRMYXllcjogTGF5ZXI7XG4gIC8qKlxuICAgKiBBIGxpc3Qgb2YgbGF5ZXJzLCB0aGUgc2FtZSBsZW5ndGggYXMgYGlucHV0VGVuc29yc2AsIHRoZSBsYXllcnMgZnJvbSB3aGVyZVxuICAgKiBgaW5wdXRUZW5zb3JzYCBvcmlnaW5hdGUuXG4gICAqL1xuICBpbmJvdW5kTGF5ZXJzOiBMYXllcltdO1xuICAvKipcbiAgICogQSBsaXN0IG9mIGludGVnZXJzLCB0aGUgc2FtZSBsZW5ndGggYXMgYGluYm91bmRMYXllcnNgLiBgbm9kZUluZGljZXNbaV1gIGlzXG4gICAqIHRoZSBvcmlnaW4gbm9kZSBvZiBgaW5wdXRUZW5zb3JzW2ldYCAobmVjZXNzYXJ5IHNpbmNlIGVhY2ggaW5ib3VuZCBsYXllclxuICAgKiBtaWdodCBoYXZlIHNldmVyYWwgbm9kZXMsIGUuZy4gaWYgdGhlIGxheWVyIGlzIGJlaW5nIHNoYXJlZCB3aXRoIGFcbiAgICogZGlmZmVyZW50IGRhdGEgc3RyZWFtKS5cbiAgICovXG4gIG5vZGVJbmRpY2VzOiBudW1iZXJbXTtcbiAgLyoqXG4gICAqIEEgbGlzdCBvZiBpbnRlZ2VycywgdGhlIHNhbWUgbGVuZ3RoIGFzIGBpbmJvdW5kTGF5ZXJzYC4gYHRlbnNvckluZGljZXNbaV1gXG4gICAqIGlzIHRoZSBpbmRleCBvZiBgaW5wdXRUZW5zb3JzW2ldYCB3aXRoaW4gdGhlIG91dHB1dCBvZiB0aGUgaW5ib3VuZCBsYXllclxuICAgKiAobmVjZXNzYXJ5IHNpbmNlIGVhY2ggaW5ib3VuZCBsYXllciBtaWdodCBoYXZlIG11bHRpcGxlIHRlbnNvciBvdXRwdXRzLFxuICAgKiB3aXRoIGVhY2ggb25lIGJlaW5nIGluZGVwZW5kZW50bHkgbWFuaXB1bGFibGUpLlxuICAgKi9cbiAgdGVuc29ySW5kaWNlczogbnVtYmVyW107XG4gIC8qKiBMaXN0IG9mIGlucHV0IHRlbnNvcnMuICovXG4gIGlucHV0VGVuc29yczogU3ltYm9saWNUZW5zb3JbXTtcbiAgLyoqIExpc3Qgb2Ygb3V0cHV0IHRlbnNvcnMuICovXG4gIG91dHB1dFRlbnNvcnM6IFN5bWJvbGljVGVuc29yW107XG4gIC8qKiBMaXN0IG9mIGlucHV0IG1hc2tzIChhIG1hc2sgY2FuIGJlIGEgdGVuc29yLCBvciBudWxsKS4gKi9cbiAgaW5wdXRNYXNrczogVGVuc29yW107XG4gIC8qKiBMaXN0IG9mIG91dHB1dCBtYXNrcyAoYSBtYXNrIGNhbiBiZSBhIHRlbnNvciwgb3IgbnVsbCkuICovXG4gIG91dHB1dE1hc2tzOiBUZW5zb3JbXTtcbiAgLyoqIExpc3Qgb2YgaW5wdXQgc2hhcGUgdHVwbGVzLiAqL1xuICBpbnB1dFNoYXBlczogU2hhcGV8U2hhcGVbXTtcbiAgLyoqIExpc3Qgb2Ygb3V0cHV0IHNoYXBlIHR1cGxlcy4gKi9cbiAgb3V0cHV0U2hhcGVzOiBTaGFwZXxTaGFwZVtdO1xuXG4gIHJlYWRvbmx5IGlkOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBhcmdzOiBOb2RlQXJncyxcbiAgICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogRGVmaW5lIGFjdHVhbCB0eXBlIGZvciB0aGlzLlxuICAgICAgcHVibGljIGNhbGxBcmdzPzogS3dhcmdzKSB7XG4gICAgdGhpcy5pZCA9IF9uZXh0Tm9kZUlEKys7XG4gICAgLypcbiAgICAgIExheWVyIGluc3RhbmNlIChOT1QgYSBsaXN0KS5cbiAgICAgIHRoaXMgaXMgdGhlIGxheWVyIHRoYXQgdGFrZXMgYSBsaXN0IG9mIGlucHV0IHRlbnNvcnNcbiAgICAgIGFuZCB0dXJucyB0aGVtIGludG8gYSBsaXN0IG9mIG91dHB1dCB0ZW5zb3JzLlxuICAgICAgdGhlIGN1cnJlbnQgbm9kZSB3aWxsIGJlIGFkZGVkIHRvXG4gICAgICB0aGUgaW5ib3VuZE5vZGVzIG9mIG91dGJvdW5kTGF5ZXIuXG4gICAgKi9cbiAgICB0aGlzLm91dGJvdW5kTGF5ZXIgPSBhcmdzLm91dGJvdW5kTGF5ZXI7XG5cbiAgICAvKlxuICAgICAgICBUaGUgZm9sbG93aW5nIDMgcHJvcGVydGllcyBkZXNjcmliZSB3aGVyZVxuICAgICAgICB0aGUgaW5wdXQgdGVuc29ycyBjb21lIGZyb206IHdoaWNoIGxheWVycyxcbiAgICAgICAgYW5kIGZvciBlYWNoIGxheWVyLCB3aGljaCBub2RlIGFuZCB3aGljaFxuICAgICAgICB0ZW5zb3Igb3V0cHV0IG9mIGVhY2ggbm9kZS5cbiAgICAqL1xuXG4gICAgLy8gTGlzdCBvZiBsYXllciBpbnN0YW5jZXMuXG4gICAgdGhpcy5pbmJvdW5kTGF5ZXJzID0gYXJncy5pbmJvdW5kTGF5ZXJzO1xuICAgIC8vIExpc3Qgb2YgaW50ZWdlcnMsIDE6MSBtYXBwaW5nIHdpdGggaW5ib3VuZExheWVycy5cbiAgICB0aGlzLm5vZGVJbmRpY2VzID0gYXJncy5ub2RlSW5kaWNlcztcbiAgICAvLyBMaXN0IG9mIGludGVnZXJzLCAxOjEgbWFwcGluZyB3aXRoIGluYm91bmRMYXllcnMuXG4gICAgdGhpcy50ZW5zb3JJbmRpY2VzID0gYXJncy50ZW5zb3JJbmRpY2VzO1xuXG4gICAgLypcbiAgICAgICAgRm9sbG93aW5nIDIgcHJvcGVydGllczpcbiAgICAgICAgdGVuc29yIGlucHV0cyBhbmQgb3V0cHV0cyBvZiBvdXRib3VuZExheWVyLlxuICAgICovXG5cbiAgICAvLyBMaXN0IG9mIHRlbnNvcnMuIDE6MSBtYXBwaW5nIHdpdGggaW5ib3VuZExheWVycy5cbiAgICB0aGlzLmlucHV0VGVuc29ycyA9IGFyZ3MuaW5wdXRUZW5zb3JzO1xuICAgIC8vIExpc3Qgb2YgdGVuc29ycywgY3JlYXRlZCBieSBvdXRib3VuZExheWVyLmNhbGwoKS5cbiAgICB0aGlzLm91dHB1dFRlbnNvcnMgPSBhcmdzLm91dHB1dFRlbnNvcnM7XG5cbiAgICAvKlxuICAgICAgICBGb2xsb3dpbmcgMiBwcm9wZXJ0aWVzOiBpbnB1dCBhbmQgb3V0cHV0IG1hc2tzLlxuICAgICAgICBMaXN0IG9mIHRlbnNvcnMsIDE6MSBtYXBwaW5nIHdpdGggaW5wdXRUZW5zb3IuXG4gICAgKi9cbiAgICB0aGlzLmlucHV0TWFza3MgPSBhcmdzLmlucHV0TWFza3M7XG4gICAgLy8gTGlzdCBvZiB0ZW5zb3JzLCBjcmVhdGVkIGJ5IG91dGJvdW5kTGF5ZXIuY29tcHV0ZU1hc2soKS5cbiAgICB0aGlzLm91dHB1dE1hc2tzID0gYXJncy5vdXRwdXRNYXNrcztcblxuICAgIC8vIEZvbGxvd2luZyAyIHByb3BlcnRpZXM6IGlucHV0IGFuZCBvdXRwdXQgc2hhcGVzLlxuXG4gICAgLy8gTGlzdCBvZiBzaGFwZSB0dXBsZXMsIHNoYXBlcyBvZiBpbnB1dFRlbnNvcnMuXG4gICAgdGhpcy5pbnB1dFNoYXBlcyA9IGFyZ3MuaW5wdXRTaGFwZXM7XG4gICAgLy8gTGlzdCBvZiBzaGFwZSB0dXBsZXMsIHNoYXBlcyBvZiBvdXRwdXRUZW5zb3JzLlxuICAgIHRoaXMub3V0cHV0U2hhcGVzID0gYXJncy5vdXRwdXRTaGFwZXM7XG5cbiAgICAvLyBBZGQgbm9kZXMgdG8gYWxsIGxheWVycyBpbnZvbHZlZC5cbiAgICBmb3IgKGNvbnN0IGxheWVyIG9mIGFyZ3MuaW5ib3VuZExheWVycykge1xuICAgICAgaWYgKGxheWVyICE9IG51bGwpIHtcbiAgICAgICAgbGF5ZXIub3V0Ym91bmROb2Rlcy5wdXNoKHRoaXMpO1xuICAgICAgfVxuICAgIH1cbiAgICBhcmdzLm91dGJvdW5kTGF5ZXIuaW5ib3VuZE5vZGVzLnB1c2godGhpcyk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBpbmJvdW5kTmFtZXM6IHN0cmluZ1tdID0gW107XG4gICAgZm9yIChjb25zdCBsYXllciBvZiB0aGlzLmluYm91bmRMYXllcnMpIHtcbiAgICAgIGlmIChsYXllciAhPSBudWxsKSB7XG4gICAgICAgIGluYm91bmROYW1lcy5wdXNoKGxheWVyLm5hbWUpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgaW5ib3VuZE5hbWVzLnB1c2gobnVsbCk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB7XG4gICAgICBvdXRib3VuZExheWVyOiB0aGlzLm91dGJvdW5kTGF5ZXIgPyB0aGlzLm91dGJvdW5kTGF5ZXIubmFtZSA6IG51bGwsXG4gICAgICBpbmJvdW5kTGF5ZXJzOiBpbmJvdW5kTmFtZXMsXG4gICAgICBub2RlSW5kaWNlczogdGhpcy5ub2RlSW5kaWNlcyxcbiAgICAgIHRlbnNvckluZGljZXM6IHRoaXMudGVuc29ySW5kaWNlc1xuICAgIH07XG4gIH1cbn1cblxuLyoqIENvbnN0cnVjdG9yIGFyZ3VtZW50cyBmb3IgTGF5ZXIuICovXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIElmIGRlZmluZWQsIHdpbGwgYmUgdXNlZCB0byBjcmVhdGUgYW4gaW5wdXQgbGF5ZXIgdG8gaW5zZXJ0IGJlZm9yZSB0aGlzXG4gICAqIGxheWVyLiBJZiBib3RoIGBpbnB1dFNoYXBlYCBhbmQgYGJhdGNoSW5wdXRTaGFwZWAgYXJlIGRlZmluZWQsXG4gICAqIGBiYXRjaElucHV0U2hhcGVgIHdpbGwgYmUgdXNlZC4gVGhpcyBhcmd1bWVudCBpcyBvbmx5IGFwcGxpY2FibGUgdG8gaW5wdXRcbiAgICogbGF5ZXJzICh0aGUgZmlyc3QgbGF5ZXIgb2YgYSBtb2RlbCkuXG4gICAqL1xuICBpbnB1dFNoYXBlPzogU2hhcGU7XG4gIC8qKlxuICAgKiBJZiBkZWZpbmVkLCB3aWxsIGJlIHVzZWQgdG8gY3JlYXRlIGFuIGlucHV0IGxheWVyIHRvIGluc2VydCBiZWZvcmUgdGhpc1xuICAgKiBsYXllci4gSWYgYm90aCBgaW5wdXRTaGFwZWAgYW5kIGBiYXRjaElucHV0U2hhcGVgIGFyZSBkZWZpbmVkLFxuICAgKiBgYmF0Y2hJbnB1dFNoYXBlYCB3aWxsIGJlIHVzZWQuIFRoaXMgYXJndW1lbnQgaXMgb25seSBhcHBsaWNhYmxlIHRvIGlucHV0XG4gICAqIGxheWVycyAodGhlIGZpcnN0IGxheWVyIG9mIGEgbW9kZWwpLlxuICAgKi9cbiAgYmF0Y2hJbnB1dFNoYXBlPzogU2hhcGU7XG4gIC8qKlxuICAgKiBJZiBgaW5wdXRTaGFwZWAgaXMgc3BlY2lmaWVkIGFuZCBgYmF0Y2hJbnB1dFNoYXBlYCBpcyAqbm90KiBzcGVjaWZpZWQsXG4gICAqIGBiYXRjaFNpemVgIGlzIHVzZWQgdG8gY29uc3RydWN0IHRoZSBgYmF0Y2hJbnB1dFNoYXBlYDogYFtiYXRjaFNpemUsXG4gICAqIC4uLmlucHV0U2hhcGVdYFxuICAgKi9cbiAgYmF0Y2hTaXplPzogbnVtYmVyO1xuICAvKipcbiAgICogVGhlIGRhdGEtdHlwZSBmb3IgdGhpcyBsYXllci4gRGVmYXVsdHMgdG8gJ2Zsb2F0MzInLlxuICAgKiBUaGlzIGFyZ3VtZW50IGlzIG9ubHkgYXBwbGljYWJsZSB0byBpbnB1dCBsYXllcnMgKHRoZSBmaXJzdCBsYXllciBvZiBhXG4gICAqIG1vZGVsKS5cbiAgICovXG4gIGR0eXBlPzogRGF0YVR5cGU7XG4gIC8qKiBOYW1lIGZvciB0aGlzIGxheWVyLiAqL1xuICBuYW1lPzogc3RyaW5nO1xuICAvKipcbiAgICogV2hldGhlciB0aGUgd2VpZ2h0cyBvZiB0aGlzIGxheWVyIGFyZSB1cGRhdGFibGUgYnkgYGZpdGAuXG4gICAqIERlZmF1bHRzIHRvIHRydWUuXG4gICAqL1xuICB0cmFpbmFibGU/OiBib29sZWFuO1xuICAvKipcbiAgICogSW5pdGlhbCB3ZWlnaHQgdmFsdWVzIG9mIHRoZSBsYXllci5cbiAgICovXG4gIHdlaWdodHM/OiBUZW5zb3JbXTtcbiAgLyoqIExlZ2FjeSBzdXBwb3J0LiBEbyBub3QgdXNlIGZvciBuZXcgY29kZS4gKi9cbiAgaW5wdXREVHlwZT86IERhdGFUeXBlO1xufVxuXG4vLyBJZiBuZWNlc3NhcnksIGFkZCBgb3V0cHV0YCBhcmd1bWVudHMgdG8gdGhlIENhbGxIb29rIGZ1bmN0aW9uLlxuLy8gVGhpcyBpcyBjdXJyZW50bHkgdXNlZCBmb3IgdGVzdGluZyBvbmx5LCBidXQgbWF5IGJlIHVzZWQgZm9yIGRlYnVnZ2VyLXJlbGF0ZWRcbi8vIHB1cnBvc2VzIGluIHRoZSBmdXR1cmUuXG5leHBvcnQgdHlwZSBDYWxsSG9vayA9IChpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpID0+IHZvaWQ7XG5cbmxldCBfbmV4dExheWVySUQgPSAwO1xuXG4vKipcbiAqIEEgbGF5ZXIgaXMgYSBncm91cGluZyBvZiBvcGVyYXRpb25zIGFuZCB3ZWlnaHRzIHRoYXQgY2FuIGJlIGNvbXBvc2VkIHRvXG4gKiBjcmVhdGUgYSBgdGYuTGF5ZXJzTW9kZWxgLlxuICpcbiAqIExheWVycyBhcmUgY29uc3RydWN0ZWQgYnkgdXNpbmcgdGhlIGZ1bmN0aW9ucyB1bmRlciB0aGVcbiAqIFt0Zi5sYXllcnNdKCNMYXllcnMtQmFzaWMpIG5hbWVzcGFjZS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTGF5ZXJzJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnLCBuYW1lc3BhY2U6ICdsYXllcnMnfVxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTGF5ZXIgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZSB7XG4gIC8qKiBOYW1lIGZvciB0aGlzIGxheWVyLiBNdXN0IGJlIHVuaXF1ZSB3aXRoaW4gYSBtb2RlbC4gKi9cbiAgbmFtZTogc3RyaW5nO1xuICAvKipcbiAgICogTGlzdCBvZiBJbnB1dFNwZWMgY2xhc3MgaW5zdGFuY2VzLlxuICAgKlxuICAgKiBFYWNoIGVudHJ5IGRlc2NyaWJlcyBvbmUgcmVxdWlyZWQgaW5wdXQ6XG4gICAqIC0gbmRpbVxuICAgKiAtIGR0eXBlXG4gICAqIEEgbGF5ZXIgd2l0aCBgbmAgaW5wdXQgdGVuc29ycyBtdXN0IGhhdmUgYW4gYGlucHV0U3BlY2Agb2YgbGVuZ3RoIGBuYC5cbiAgICovXG4gIGlucHV0U3BlYzogSW5wdXRTcGVjW107XG4gIHN1cHBvcnRzTWFza2luZzogYm9vbGVhbjtcbiAgLyoqIFdoZXRoZXIgdGhlIGxheWVyIHdlaWdodHMgd2lsbCBiZSB1cGRhdGVkIGR1cmluZyB0cmFpbmluZy4gKi9cbiAgcHJvdGVjdGVkIHRyYWluYWJsZV86IGJvb2xlYW47XG4gIGJhdGNoSW5wdXRTaGFwZTogU2hhcGU7XG4gIGR0eXBlOiBEYXRhVHlwZTtcbiAgaW5pdGlhbFdlaWdodHM6IFRlbnNvcltdO1xuXG4gIGluYm91bmROb2RlczogTm9kZVtdO1xuICBvdXRib3VuZE5vZGVzOiBOb2RlW107XG5cbiAgYWN0aXZpdHlSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG5cbiAgcHJvdGVjdGVkIF90cmFpbmFibGVXZWlnaHRzOiBMYXllclZhcmlhYmxlW107XG4gIHByaXZhdGUgX25vblRyYWluYWJsZVdlaWdodHM6IExheWVyVmFyaWFibGVbXTtcbiAgcHJpdmF0ZSBfbG9zc2VzOiBSZWd1bGFyaXplckZuW107XG4gIC8vIFRPRE8oY2Fpcyk6IF91cGRhdGVzIGlzIGN1cnJlbnRseSB1bnVzZWQuXG4gIHByaXZhdGUgX3VwZGF0ZXM6IFRlbnNvcltdO1xuICBwcml2YXRlIF9idWlsdDogYm9vbGVhbjtcbiAgcHJpdmF0ZSBfY2FsbEhvb2s6IENhbGxIb29rID0gbnVsbDtcblxuICBwcml2YXRlIF9hZGRlZFdlaWdodE5hbWVzOiBzdHJpbmdbXSA9IFtdO1xuXG4gIHJlYWRvbmx5IGlkOiBudW1iZXI7XG5cbiAgLy8gUG9ydGluZyBOb3RlczogUHlLZXJhcyBkb2VzIG5vdCBoYXZlIHRoaXMgcHJvcGVydHkgaW4gdGhpcyBiYXNlIExheWVyXG4gIC8vICAgY2xhc3MuIEluc3RlYWQgbGV0cyBMYXllciBzdWJjbGFzcyBzZXQgaXQgZHluYW1pY2FsbHkgYW5kIGNoZWNrcyB0aGVcbiAgLy8gICB2YWx1ZSB3aXRoIGBoYXNhdHRyYC4gSW4gdGZqcy1sYXllcnMsIHdlIGxldCB0aGlzIGJlIGEgbWVtYmVyIG9mIHRoaXNcbiAgLy8gICBiYXNlIGNsYXNzLlxuICBwcm90ZWN0ZWQgX3N0YXRlZnVsID0gZmFsc2U7XG5cbiAgcHJvdGVjdGVkIF9yZWZDb3VudDogbnVtYmVyfG51bGw7XG5cbiAgLy8gQSBmbGFnIGZvciB3aGV0aGVyIGZhc3QgKGkuZS4sIGFsbC16ZXJvKSB3ZWlnaHQgaW5pdGlhbGl6YXRpb24gaXMgdG9cbiAgLy8gYmUgdXNlZCBkdXJpbmcgYGJ1aWxkKClgIGNhbGwuIFRoaXMgc3BlZWRzIHVwIHdlaWdodCBpbml0aWFsaXphdGlvblxuICAvLyBieSBzYXZpbmcgdW5uZWNlc3NhcnkgY2FsbHMgdG8gZXhwZW5zaXZlIGluaXRpYWxpemVycyBpbiBjYXNlcyB3aGVyZVxuICAvLyB0aGUgaW5pdGlhbGl6ZWQgdmFsdWVzIHdpbGwgYmUgb3ZlcndyaXR0ZW4gYnkgbG9hZGVkIHdlaWdodCB2YWx1ZXNcbiAgLy8gZHVyaW5nIG1vZGVsIGxvYWRpbmcuXG4gIHByaXZhdGUgZmFzdFdlaWdodEluaXREdXJpbmdCdWlsZDogYm9vbGVhbjtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBMYXllckFyZ3MgPSB7fSkge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5pZCA9IF9uZXh0TGF5ZXJJRCsrO1xuXG4gICAgdGhpcy5hY3Rpdml0eVJlZ3VsYXJpemVyID0gbnVsbDtcblxuICAgIHRoaXMuaW5wdXRTcGVjID0gbnVsbDtcbiAgICB0aGlzLnN1cHBvcnRzTWFza2luZyA9IGZhbHNlO1xuXG4gICAgLy8gVGhlc2UgcHJvcGVydGllcyB3aWxsIGJlIHNldCB1cG9uIGNhbGwgb2YgdGhpcy5idWlsZCgpXG4gICAgdGhpcy5fdHJhaW5hYmxlV2VpZ2h0cyA9IFtdO1xuICAgIHRoaXMuX25vblRyYWluYWJsZVdlaWdodHMgPSBbXTtcbiAgICB0aGlzLl9sb3NzZXMgPSBbXTtcbiAgICB0aGlzLl91cGRhdGVzID0gW107XG4gICAgdGhpcy5fYnVpbHQgPSBmYWxzZTtcblxuICAgIC8qXG4gICAgICBUaGVzZSBsaXN0cyB3aWxsIGJlIGZpbGxlZCB2aWEgc3VjY2Vzc2l2ZSBjYWxsc1xuICAgICAgdG8gdGhpcy5hZGRJbmJvdW5kTm9kZSgpLlxuICAgICAqL1xuICAgIHRoaXMuaW5ib3VuZE5vZGVzID0gW107XG4gICAgdGhpcy5vdXRib3VuZE5vZGVzID0gW107XG5cbiAgICBsZXQgbmFtZSA9IGFyZ3MubmFtZTtcbiAgICBpZiAoIW5hbWUpIHtcbiAgICAgIGNvbnN0IHByZWZpeCA9IHRoaXMuZ2V0Q2xhc3NOYW1lKCk7XG4gICAgICBuYW1lID0gZ2VuZXJpY191dGlscy50b1NuYWtlQ2FzZShwcmVmaXgpICsgJ18nICsgZ2V0VWlkKHByZWZpeCk7XG4gICAgfVxuICAgIHRoaXMubmFtZSA9IG5hbWU7XG5cbiAgICB0aGlzLnRyYWluYWJsZV8gPSBhcmdzLnRyYWluYWJsZSA9PSBudWxsID8gdHJ1ZSA6IGFyZ3MudHJhaW5hYmxlO1xuXG4gICAgaWYgKGFyZ3MuaW5wdXRTaGFwZSAhPSBudWxsIHx8IGFyZ3MuYmF0Y2hJbnB1dFNoYXBlICE9IG51bGwpIHtcbiAgICAgIC8qXG4gICAgICAgIEluIHRoaXMgY2FzZSB3ZSB3aWxsIGxhdGVyIGNyZWF0ZSBhbiBpbnB1dCBsYXllclxuICAgICAgICB0byBpbnNlcnQgYmVmb3JlIHRoZSBjdXJyZW50IGxheWVyXG4gICAgICAgKi9cbiAgICAgIGxldCBiYXRjaElucHV0U2hhcGU6IFNoYXBlO1xuICAgICAgaWYgKGFyZ3MuYmF0Y2hJbnB1dFNoYXBlICE9IG51bGwpIHtcbiAgICAgICAgYmF0Y2hJbnB1dFNoYXBlID0gYXJncy5iYXRjaElucHV0U2hhcGU7XG4gICAgICB9IGVsc2UgaWYgKGFyZ3MuaW5wdXRTaGFwZSAhPSBudWxsKSB7XG4gICAgICAgIGxldCBiYXRjaFNpemU6IG51bWJlciA9IG51bGw7XG4gICAgICAgIGlmIChhcmdzLmJhdGNoU2l6ZSAhPSBudWxsKSB7XG4gICAgICAgICAgYmF0Y2hTaXplID0gYXJncy5iYXRjaFNpemU7XG4gICAgICAgIH1cbiAgICAgICAgYmF0Y2hJbnB1dFNoYXBlID0gW2JhdGNoU2l6ZV0uY29uY2F0KGFyZ3MuaW5wdXRTaGFwZSk7XG4gICAgICB9XG4gICAgICB0aGlzLmJhdGNoSW5wdXRTaGFwZSA9IGJhdGNoSW5wdXRTaGFwZTtcblxuICAgICAgLy8gU2V0IGR0eXBlLlxuICAgICAgbGV0IGR0eXBlID0gYXJncy5kdHlwZTtcbiAgICAgIGlmIChkdHlwZSA9PSBudWxsKSB7XG4gICAgICAgIGR0eXBlID0gYXJncy5pbnB1dERUeXBlO1xuICAgICAgfVxuICAgICAgaWYgKGR0eXBlID09IG51bGwpIHtcbiAgICAgICAgZHR5cGUgPSAnZmxvYXQzMic7XG4gICAgICB9XG4gICAgICB0aGlzLmR0eXBlID0gZHR5cGU7XG4gICAgfVxuXG4gICAgaWYgKGFyZ3Mud2VpZ2h0cyAhPSBudWxsKSB7XG4gICAgICB0aGlzLmluaXRpYWxXZWlnaHRzID0gYXJncy53ZWlnaHRzO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmluaXRpYWxXZWlnaHRzID0gbnVsbDtcbiAgICB9XG5cbiAgICAvLyBUaGUgdmFsdWUgb2YgYF9yZWZDb3VudGAgaXMgaW5pdGlhbGl6ZWQgdG8gbnVsbC4gV2hlbiB0aGUgbGF5ZXIgaXMgdXNlZFxuICAgIC8vIGluIGEgc3ltYm9saWMgd2F5IGZvciB0aGUgZmlyc3QgdGltZSwgaXQgd2lsbCBiZSBzZXQgdG8gMS5cbiAgICB0aGlzLl9yZWZDb3VudCA9IG51bGw7XG5cbiAgICB0aGlzLmZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQgPSBmYWxzZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb252ZXJ0cyBhIGxheWVyIGFuZCBpdHMgaW5kZXggdG8gYSB1bmlxdWUgKGltbXV0YWJsZSB0eXBlKSBuYW1lLlxuICAgKiBUaGlzIGZ1bmN0aW9uIGlzIHVzZWQgaW50ZXJuYWxseSB3aXRoIGB0aGlzLmNvbnRhaW5lck5vZGVzYC5cbiAgICogQHBhcmFtIGxheWVyIFRoZSBsYXllci5cbiAgICogQHBhcmFtIG5vZGVJbmRleCBUaGUgbGF5ZXIncyBwb3NpdGlvbiAoZS5nLiB2aWEgZW51bWVyYXRlKSBpbiBhIGxpc3Qgb2ZcbiAgICogICBub2Rlcy5cbiAgICpcbiAgICogQHJldHVybnMgVGhlIHVuaXF1ZSBuYW1lLlxuICAgKi9cbiAgcHJvdGVjdGVkIHN0YXRpYyBub2RlS2V5KGxheWVyOiBMYXllciwgbm9kZUluZGV4OiBudW1iZXIpIHtcbiAgICByZXR1cm4gbGF5ZXIubmFtZSArICdfaWItJyArIG5vZGVJbmRleC50b1N0cmluZygpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhpcy5pbmJvdW5kTm9kZSBhdCBpbmRleCBub2RlSW5kZXguXG4gICAqXG4gICAqIFBvcnRpbmcgbm90ZTogVGhpcyBpcyBhIHJlcGxhY2VtZW50IGZvciBfZ2V0X25vZGVfYXR0cmlidXRlX2F0X2luZGV4KClcbiAgICogQHBhcmFtIG5vZGVJbmRleFxuICAgKiBAcGFyYW0gYXR0ck5hbWUgVGhlIG5hbWUgb2YgdGhlIGF0dHJpYnV0ZSByZWxhdGVkIHRvIHJlcXVlc3QgZm9yIHRoaXMgbm9kZS5cbiAgICovXG4gIHByaXZhdGUgZ2V0Tm9kZUF0SW5kZXgobm9kZUluZGV4OiBudW1iZXIsIGF0dHJOYW1lOiBzdHJpbmcpOiBOb2RlIHtcbiAgICBpZiAodGhpcy5pbmJvdW5kTm9kZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgUnVudGltZUVycm9yKFxuICAgICAgICAgICdUaGUgbGF5ZXIgaGFzIG5ldmVyIGJlZW4gY2FsbGVkICcgK1xuICAgICAgICAgIGBhbmQgdGh1cyBoYXMgbm8gZGVmaW5lZCAke2F0dHJOYW1lfS5gKTtcbiAgICB9XG4gICAgaWYgKHRoaXMuaW5ib3VuZE5vZGVzLmxlbmd0aCA8PSBub2RlSW5kZXgpIHtcbiAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgIGBBc2tlZCB0byBnZXQgJHthdHRyTmFtZX0gYXQgbm9kZSAke25vZGVJbmRleH0sIGAgK1xuICAgICAgICAgIGBidXQgdGhlIGxheWVyIGhhcyBvbmx5ICR7dGhpcy5pbmJvdW5kTm9kZXMubGVuZ3RofSBpbmJvdW5kIG5vZGVzLmApO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5pbmJvdW5kTm9kZXNbbm9kZUluZGV4XTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXRyaWV2ZXMgdGhlIGlucHV0IHRlbnNvcihzKSBvZiBhIGxheWVyIGF0IGEgZ2l2ZW4gbm9kZS5cbiAgICpcbiAgICogQHBhcmFtIG5vZGVJbmRleCBJbnRlZ2VyLCBpbmRleCBvZiB0aGUgbm9kZSBmcm9tIHdoaWNoIHRvIHJldHJpZXZlIHRoZVxuICAgKiAgIGF0dHJpYnV0ZS4gRS5nLiBgbm9kZUluZGV4PTBgIHdpbGwgY29ycmVzcG9uZCB0byB0aGUgZmlyc3QgdGltZSB0aGUgbGF5ZXJcbiAgICogICB3YXMgY2FsbGVkLlxuICAgKlxuICAgKiBAcmV0dXJuIEEgdGVuc29yIChvciBsaXN0IG9mIHRlbnNvcnMgaWYgdGhlIGxheWVyIGhhcyBtdWx0aXBsZSBpbnB1dHMpLlxuICAgKi9cbiAgZ2V0SW5wdXRBdChub2RlSW5kZXg6IG51bWJlcik6IFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW10ge1xuICAgIHJldHVybiBnZW5lcmljX3V0aWxzLnNpbmdsZXRvbk9yQXJyYXkoXG4gICAgICAgIHRoaXMuZ2V0Tm9kZUF0SW5kZXgobm9kZUluZGV4LCAnaW5wdXQnKS5pbnB1dFRlbnNvcnMpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHJpZXZlcyB0aGUgb3V0cHV0IHRlbnNvcihzKSBvZiBhIGxheWVyIGF0IGEgZ2l2ZW4gbm9kZS5cbiAgICpcbiAgICogQHBhcmFtIG5vZGVJbmRleCBJbnRlZ2VyLCBpbmRleCBvZiB0aGUgbm9kZSBmcm9tIHdoaWNoIHRvIHJldHJpZXZlIHRoZVxuICAgKiAgIGF0dHJpYnV0ZS4gRS5nLiBgbm9kZUluZGV4PTBgIHdpbGwgY29ycmVzcG9uZCB0byB0aGUgZmlyc3QgdGltZSB0aGUgbGF5ZXJcbiAgICogICB3YXMgY2FsbGVkLlxuICAgKlxuICAgKiBAcmV0dXJuIEEgdGVuc29yIChvciBsaXN0IG9mIHRlbnNvcnMgaWYgdGhlIGxheWVyIGhhcyBtdWx0aXBsZSBvdXRwdXRzKS5cbiAgICovXG4gIGdldE91dHB1dEF0KG5vZGVJbmRleDogbnVtYmVyKTogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSB7XG4gICAgcmV0dXJuIGdlbmVyaWNfdXRpbHMuc2luZ2xldG9uT3JBcnJheShcbiAgICAgICAgdGhpcy5nZXROb2RlQXRJbmRleChub2RlSW5kZXgsICdvdXRwdXQnKS5vdXRwdXRUZW5zb3JzKTtcbiAgfVxuXG4gIC8vIFByb3BlcnRpZXNcblxuICAvKipcbiAgICogUmV0cmlldmVzIHRoZSBpbnB1dCB0ZW5zb3Iocykgb2YgYSBsYXllci5cbiAgICpcbiAgICogT25seSBhcHBsaWNhYmxlIGlmIHRoZSBsYXllciBoYXMgZXhhY3RseSBvbmUgaW5ib3VuZCBub2RlLFxuICAgKiBpLmUuIGlmIGl0IGlzIGNvbm5lY3RlZCB0byBvbmUgaW5jb21pbmcgbGF5ZXIuXG4gICAqXG4gICAqIEByZXR1cm4gSW5wdXQgdGVuc29yIG9yIGxpc3Qgb2YgaW5wdXQgdGVuc29ycy5cbiAgICpcbiAgICogQGV4Y2VwdGlvbiBBdHRyaWJ1dGVFcnJvciBpZiB0aGUgbGF5ZXIgaXMgY29ubmVjdGVkIHRvIG1vcmUgdGhhbiBvbmVcbiAgICogICBpbmNvbWluZyBsYXllcnMuXG4gICAqL1xuICBnZXQgaW5wdXQoKTogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSB7XG4gICAgaWYgKHRoaXMuaW5ib3VuZE5vZGVzLmxlbmd0aCA+IDEpIHtcbiAgICAgIHRocm93IG5ldyBBdHRyaWJ1dGVFcnJvcihcbiAgICAgICAgICBgTGF5ZXIgJHt0aGlzLm5hbWV9YCArXG4gICAgICAgICAgJyBoYXMgbXVsdGlwbGUgaW5ib3VuZCBub2RlcywgJyArXG4gICAgICAgICAgJ2hlbmNlIHRoZSBub3Rpb24gb2YgXCJsYXllciBpbnB1dFwiICcgK1xuICAgICAgICAgICdpcyBpbGwtZGVmaW5lZC4gJyArXG4gICAgICAgICAgJ1VzZSBgZ2V0SW5wdXRBdChub2RlSW5kZXgpYCBpbnN0ZWFkLicpO1xuICAgIH0gZWxzZSBpZiAodGhpcy5pbmJvdW5kTm9kZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgQXR0cmlidXRlRXJyb3IoXG4gICAgICAgICAgYExheWVyICR7dGhpcy5uYW1lfWAgK1xuICAgICAgICAgICcgaXMgbm90IGNvbm5lY3RlZCwgbm8gaW5wdXQgdG8gcmV0dXJuLicpO1xuICAgIH1cbiAgICByZXR1cm4gZ2VuZXJpY191dGlscy5zaW5nbGV0b25PckFycmF5KFxuICAgICAgICB0aGlzLmdldE5vZGVBdEluZGV4KDAsICdpbnB1dCcpLmlucHV0VGVuc29ycyk7XG4gIH1cblxuICAvKipcbiAgICogUmV0cmlldmVzIHRoZSBvdXRwdXQgdGVuc29yKHMpIG9mIGEgbGF5ZXIuXG4gICAqXG4gICAqIE9ubHkgYXBwbGljYWJsZSBpZiB0aGUgbGF5ZXIgaGFzIGV4YWN0bHkgb25lIGluYm91bmQgbm9kZSxcbiAgICogaS5lLiBpZiBpdCBpcyBjb25uZWN0ZWQgdG8gb25lIGluY29taW5nIGxheWVyLlxuICAgKlxuICAgKiBAcmV0dXJuIE91dHB1dCB0ZW5zb3Igb3IgbGlzdCBvZiBvdXRwdXQgdGVuc29ycy5cbiAgICpcbiAgICogQGV4Y2VwdGlvbiBBdHRyaWJ1dGVFcnJvciBpZiB0aGUgbGF5ZXIgaXMgY29ubmVjdGVkIHRvIG1vcmUgdGhhbiBvbmVcbiAgICogICBpbmNvbWluZyBsYXllcnMuXG4gICAqL1xuICBnZXQgb3V0cHV0KCk6IFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW10ge1xuICAgIGlmICh0aGlzLmluYm91bmROb2Rlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIHRocm93IG5ldyBBdHRyaWJ1dGVFcnJvcihcbiAgICAgICAgICBgTGF5ZXIgJHt0aGlzLm5hbWV9YCArXG4gICAgICAgICAgJyBoYXMgbm8gaW5ib3VuZCBub2Rlcy4nKTtcbiAgICB9XG4gICAgaWYgKHRoaXMuaW5ib3VuZE5vZGVzLmxlbmd0aCA+IDEpIHtcbiAgICAgIHRocm93IG5ldyBBdHRyaWJ1dGVFcnJvcihcbiAgICAgICAgICBgTGF5ZXIgJHt0aGlzLm5hbWV9YCArXG4gICAgICAgICAgJyBoYXMgbXVsdGlwbGUgaW5ib3VuZCBub2RlcywgJyArXG4gICAgICAgICAgJ2hlbmNlIHRoZSBub3Rpb24gb2YgXCJsYXllciBvdXRwdXRcIiAnICtcbiAgICAgICAgICAnaXMgaWxsLWRlZmluZWQuICcgK1xuICAgICAgICAgICdVc2UgYGdldE91dHB1dEF0KG5vZGVJbmRleClgIGluc3RlYWQuJyk7XG4gICAgfVxuICAgIHJldHVybiBnZW5lcmljX3V0aWxzLnNpbmdsZXRvbk9yQXJyYXkoXG4gICAgICAgIHRoaXMuZ2V0Tm9kZUF0SW5kZXgoMCwgJ291dHB1dCcpLm91dHB1dFRlbnNvcnMpO1xuICB9XG5cbiAgZ2V0IGxvc3NlcygpOiBSZWd1bGFyaXplckZuW10ge1xuICAgIHJldHVybiB0aGlzLl9sb3NzZXM7XG4gIH1cblxuICAvKipcbiAgICogUmV0cmlldmVzIHRoZSBMYXllcidzIGN1cnJlbnQgbG9zcyB2YWx1ZXMuXG4gICAqXG4gICAqIFVzZWQgZm9yIHJlZ3VsYXJpemVycyBkdXJpbmcgdHJhaW5pbmcuXG4gICAqL1xuICBjYWxjdWxhdGVMb3NzZXMoKTogU2NhbGFyW10ge1xuICAgIC8vIFBvcnRpbmcgTm9kZTogVGhpcyBpcyBhbiBhdWdtZW50YXRpb24gdG8gTGF5ZXIubG9zcyBpbiBQeUtlcmFzLlxuICAgIC8vICAgSW4gUHlLZXJhcywgTGF5ZXIubG9zcyByZXR1cm5zIHN5bWJvbGljIHRlbnNvcnMuIEhlcmUgYSBjb25jcmV0ZVxuICAgIC8vICAgVGVuc29yIChzcGVjaWZpY2FsbHkgU2NhbGFyKSB2YWx1ZXMgYXJlIHJldHVybmVkLiBUaGlzIGlzIGR1ZSB0byB0aGVcbiAgICAvLyAgIGltcGVyYXRpdmUgYmFja2VuZC5cbiAgICByZXR1cm4gdGhpcy5sb3NzZXMubWFwKGxvc3NGbiA9PiBsb3NzRm4oKSk7XG4gIH1cblxuICBnZXQgdXBkYXRlcygpOiBUZW5zb3JbXSB7XG4gICAgcmV0dXJuIHRoaXMuX3VwZGF0ZXM7XG4gIH1cblxuICBnZXQgYnVpbHQoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMuX2J1aWx0O1xuICB9XG5cbiAgc2V0IGJ1aWx0KGJ1aWx0OiBib29sZWFuKSB7XG4gICAgdGhpcy5fYnVpbHQgPSBidWlsdDtcbiAgfVxuXG4gIGdldCB0cmFpbmFibGUoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMudHJhaW5hYmxlXztcbiAgfVxuXG4gIHNldCB0cmFpbmFibGUodHJhaW5hYmxlOiBib29sZWFuKSB7XG4gICAgdGhpcy5fdHJhaW5hYmxlV2VpZ2h0cy5mb3JFYWNoKHcgPT4gdy50cmFpbmFibGUgPSB0cmFpbmFibGUpO1xuICAgIHRoaXMudHJhaW5hYmxlXyA9IHRyYWluYWJsZTtcbiAgfVxuXG4gIGdldCB0cmFpbmFibGVXZWlnaHRzKCk6IExheWVyVmFyaWFibGVbXSB7XG4gICAgaWYgKHRoaXMudHJhaW5hYmxlXykge1xuICAgICAgcmV0dXJuIHRoaXMuX3RyYWluYWJsZVdlaWdodHMuZmlsdGVyKHcgPT4gdy50cmFpbmFibGUpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gW107XG4gICAgfVxuICB9XG5cbiAgc2V0IHRyYWluYWJsZVdlaWdodHMod2VpZ2h0czogTGF5ZXJWYXJpYWJsZVtdKSB7XG4gICAgdGhpcy5fdHJhaW5hYmxlV2VpZ2h0cyA9IHdlaWdodHM7XG4gIH1cblxuICBnZXQgbm9uVHJhaW5hYmxlV2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIGlmICh0aGlzLnRyYWluYWJsZSkge1xuICAgICAgcmV0dXJuIHRoaXMuX3RyYWluYWJsZVdlaWdodHMuZmlsdGVyKHcgPT4gIXcudHJhaW5hYmxlKVxuICAgICAgICAgIC5jb25jYXQodGhpcy5fbm9uVHJhaW5hYmxlV2VpZ2h0cyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiB0aGlzLl90cmFpbmFibGVXZWlnaHRzLmNvbmNhdCh0aGlzLl9ub25UcmFpbmFibGVXZWlnaHRzKTtcbiAgICB9XG4gIH1cblxuICBzZXQgbm9uVHJhaW5hYmxlV2VpZ2h0cyh3ZWlnaHRzOiBMYXllclZhcmlhYmxlW10pIHtcbiAgICB0aGlzLl9ub25UcmFpbmFibGVXZWlnaHRzID0gd2VpZ2h0cztcbiAgfVxuXG4gIC8qKlxuICAgKiBUaGUgY29uY2F0ZW5hdGlvbiBvZiB0aGUgbGlzdHMgdHJhaW5hYmxlV2VpZ2h0cyBhbmQgbm9uVHJhaW5hYmxlV2VpZ2h0c1xuICAgKiAoaW4gdGhpcyBvcmRlcikuXG4gICAqL1xuICBnZXQgd2VpZ2h0cygpOiBMYXllclZhcmlhYmxlW10ge1xuICAgIHJldHVybiB0aGlzLnRyYWluYWJsZVdlaWdodHMuY29uY2F0KHRoaXMubm9uVHJhaW5hYmxlV2VpZ2h0cyk7XG4gIH1cblxuICBnZXQgc3RhdGVmdWwoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMuX3N0YXRlZnVsO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlc2V0IHRoZSBzdGF0ZXMgb2YgdGhlIGxheWVyLlxuICAgKlxuICAgKiBUaGlzIG1ldGhvZCBvZiB0aGUgYmFzZSBMYXllciBjbGFzcyBpcyBlc3NlbnRpYWxseSBhIG5vLW9wLlxuICAgKiBTdWJjbGFzc2VzIHRoYXQgYXJlIHN0YXRlZnVsIChlLmcuLCBzdGF0ZWZ1bCBSTk5zKSBzaG91bGQgb3ZlcnJpZGUgdGhpc1xuICAgKiBtZXRob2QuXG4gICAqL1xuICByZXNldFN0YXRlcygpOiB2b2lkIHtcbiAgICBpZiAoIXRoaXMuc3RhdGVmdWwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQ2Fubm90IGNhbGwgdGhlIHJlc2V0U3RhdGVzKCkgbWV0aG9kIG9mIGEgbm9uLXN0YXRlZnVsIExheWVyICcgK1xuICAgICAgICAgICdvYmplY3QuJyk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIENoZWNrcyBjb21wYXRpYmlsaXR5IGJldHdlZW4gdGhlIGxheWVyIGFuZCBwcm92aWRlZCBpbnB1dHMuXG4gICAqXG4gICAqIFRoaXMgY2hlY2tzIHRoYXQgdGhlIHRlbnNvcihzKSBgaW5wdXRgXG4gICAqIHZlcmlmeSB0aGUgaW5wdXQgYXNzdW1wdGlvbnMgb2YgdGhlIGxheWVyXG4gICAqIChpZiBhbnkpLiBJZiBub3QsIGV4Y2VwdGlvbnMgYXJlIHJhaXNlZC5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0cyBJbnB1dCB0ZW5zb3Igb3IgbGlzdCBvZiBpbnB1dCB0ZW5zb3JzLlxuICAgKlxuICAgKiBAZXhjZXB0aW9uIFZhbHVlRXJyb3IgaW4gY2FzZSBvZiBtaXNtYXRjaCBiZXR3ZWVuXG4gICAqICAgdGhlIHByb3ZpZGVkIGlucHV0cyBhbmQgdGhlIGV4cGVjdGF0aW9ucyBvZiB0aGUgbGF5ZXIuXG4gICAqL1xuICBwcm90ZWN0ZWQgYXNzZXJ0SW5wdXRDb21wYXRpYmlsaXR5KGlucHV0czogVGVuc29yfFRlbnNvcltdfFN5bWJvbGljVGVuc29yfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFN5bWJvbGljVGVuc29yW10pOiB2b2lkIHtcbiAgICBjb25zdCBpbnB1dHNMaXN0ID0gZ2VuZXJpY191dGlscy50b0xpc3QoaW5wdXRzKTtcbiAgICBpZiAodGhpcy5pbnB1dFNwZWMgPT0gbnVsbCB8fCB0aGlzLmlucHV0U3BlYy5sZW5ndGggPT09IDApIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgaW5wdXRTcGVjID0gZ2VuZXJpY191dGlscy50b0xpc3QodGhpcy5pbnB1dFNwZWMpO1xuICAgIGlmIChpbnB1dHNMaXN0Lmxlbmd0aCAhPT0gaW5wdXRTcGVjLmxlbmd0aCkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYExheWVyICR7dGhpcy5uYW1lfSBleHBlY3RzICR7aW5wdXRTcGVjLmxlbmd0aH0gaW5wdXRzLCBgICtcbiAgICAgICAgICBgYnV0IGl0IHJlY2VpdmVkICR7aW5wdXRzTGlzdC5sZW5ndGh9IGlucHV0IHRlbnNvcnMuIGAgK1xuICAgICAgICAgIGBJbnB1dCByZWNlaXZlZDogJHtpbnB1dHN9YCk7XG4gICAgfVxuICAgIGZvciAobGV0IGlucHV0SW5kZXggPSAwOyBpbnB1dEluZGV4IDwgaW5wdXRzTGlzdC5sZW5ndGg7IGlucHV0SW5kZXgrKykge1xuICAgICAgY29uc3QgeCA9IGlucHV0c0xpc3RbaW5wdXRJbmRleF07XG4gICAgICBjb25zdCBzcGVjOiBJbnB1dFNwZWMgPSBpbnB1dFNwZWNbaW5wdXRJbmRleF07XG4gICAgICBpZiAoc3BlYyA9PSBudWxsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICAvLyBDaGVjayBuZGltLlxuICAgICAgY29uc3QgbmRpbSA9IHgucmFuaztcbiAgICAgIGlmIChzcGVjLm5kaW0gIT0gbnVsbCkge1xuICAgICAgICBpZiAobmRpbSAhPT0gc3BlYy5uZGltKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBJbnB1dCAke2lucHV0SW5kZXh9IGlzIGluY29tcGF0aWJsZSB3aXRoIGxheWVyICR7dGhpcy5uYW1lfTogYCArXG4gICAgICAgICAgICAgIGBleHBlY3RlZCBuZGltPSR7c3BlYy5uZGltfSwgZm91bmQgbmRpbT0ke25kaW19YCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChzcGVjLm1heE5EaW0gIT0gbnVsbCkge1xuICAgICAgICBpZiAobmRpbSA+IHNwZWMubWF4TkRpbSkge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICBgSW5wdXQgJHtpbnB1dEluZGV4fSBpcyBpbmNvbXBhdGlibGUgd2l0aCBsYXllciAke3RoaXMubmFtZX1gICtcbiAgICAgICAgICAgICAgYDogZXhwZWN0ZWQgbWF4X25kaW09JHtzcGVjLm1heE5EaW19LCBmb3VuZCBuZGltPSR7bmRpbX1gKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHNwZWMubWluTkRpbSAhPSBudWxsKSB7XG4gICAgICAgIGlmIChuZGltIDwgc3BlYy5taW5ORGltKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBJbnB1dCAke2lucHV0SW5kZXh9IGlzIGluY29tcGF0aWJsZSB3aXRoIGxheWVyICR7dGhpcy5uYW1lfWAgK1xuICAgICAgICAgICAgICBgOiBleHBlY3RlZCBtaW5fbmRpbT0ke3NwZWMubWluTkRpbX0sIGZvdW5kIG5kaW09JHtuZGltfS5gKTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICAvLyBDaGVjayBkdHlwZS5cbiAgICAgIGlmIChzcGVjLmR0eXBlICE9IG51bGwpIHtcbiAgICAgICAgaWYgKHguZHR5cGUgIT09IHNwZWMuZHR5cGUpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgYElucHV0ICR7aW5wdXRJbmRleH0gaXMgaW5jb21wYXRpYmxlIHdpdGggbGF5ZXIgJHt0aGlzLm5hbWV9IGAgK1xuICAgICAgICAgICAgICBgOiBleHBlY3RlZCBkdHlwZT0ke3NwZWMuZHR5cGV9LCBmb3VuZCBkdHlwZT0ke3guZHR5cGV9LmApO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIC8vIENoZWNrIHNwZWNpZmljIHNoYXBlIGF4ZXMuXG4gICAgICBpZiAoc3BlYy5heGVzKSB7XG4gICAgICAgIGNvbnN0IHhTaGFwZSA9IHguc2hhcGU7XG4gICAgICAgIGZvciAoY29uc3Qga2V5IGluIHNwZWMuYXhlcykge1xuICAgICAgICAgIGNvbnN0IGF4aXMgPSBOdW1iZXIoa2V5KTtcbiAgICAgICAgICBjb25zdCB2YWx1ZSA9IHNwZWMuYXhlc1trZXldO1xuICAgICAgICAgIC8vIFBlcmZvcm0gUHl0aG9uLXN0eWxlIHNsaWNpbmcgaW4gY2FzZSBheGlzIDwgMDtcbiAgICAgICAgICAvLyBUT0RPKGNhaXMpOiBVc2UgaHR0cHM6Ly9naXRodWIuY29tL2Fsdml2aS90eXBlc2NyaXB0LXVuZGVyc2NvcmUgdG9cbiAgICAgICAgICAvLyBlbnN1cmUgdHlwZSBzYWZldHkgdGhyb3VnaCBVbmRlcnNjb3JlIGNhbGxzLlxuICAgICAgICAgIGNvbnN0IHhTaGFwZUF0QXhpcyA9XG4gICAgICAgICAgICAgIGF4aXMgPj0gMCA/IHhTaGFwZVtheGlzXSA6IHhTaGFwZVt4U2hhcGUubGVuZ3RoICsgYXhpc107XG4gICAgICAgICAgaWYgKHZhbHVlICE9IG51bGwgJiYgW3ZhbHVlLCBudWxsXS5pbmRleE9mKHhTaGFwZUF0QXhpcykgPT09IC0xKSB7XG4gICAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgICBgSW5wdXQgJHtpbnB1dEluZGV4fSBpcyBpbmNvbXBhdGlibGUgd2l0aCBsYXllciBgICtcbiAgICAgICAgICAgICAgICBgJHt0aGlzLm5hbWV9OiBleHBlY3RlZCBheGlzICR7YXhpc30gb2YgaW5wdXQgc2hhcGUgdG8gYCArXG4gICAgICAgICAgICAgICAgYGhhdmUgdmFsdWUgJHt2YWx1ZX0gYnV0IGdvdCBzaGFwZSAke3hTaGFwZX0uYCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIC8vIENoZWNrIHNoYXBlLlxuICAgICAgaWYgKHNwZWMuc2hhcGUgIT0gbnVsbCkge1xuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNwZWMuc2hhcGUubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgICBjb25zdCBzcGVjRGltID0gc3BlYy5zaGFwZVtpXTtcbiAgICAgICAgICBjb25zdCBkaW0gPSB4LnNoYXBlW2ldO1xuICAgICAgICAgIGlmIChzcGVjRGltICE9IG51bGwgJiYgZGltICE9IG51bGwpIHtcbiAgICAgICAgICAgIGlmIChzcGVjRGltICE9PSBkaW0pIHtcbiAgICAgICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgICAgICBgSW5wdXQgJHtpbnB1dEluZGV4fSBpcyBpbmNvbXBhdGlibGUgd2l0aCBsYXllciBgICtcbiAgICAgICAgICAgICAgICAgIGAke3RoaXMubmFtZX06IGV4cGVjdGVkIHNoYXBlPSR7c3BlYy5zaGFwZX0sIGAgK1xuICAgICAgICAgICAgICAgICAgYGZvdW5kIHNoYXBlPSR7eC5zaGFwZX0uYCk7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFRoaXMgaXMgd2hlcmUgdGhlIGxheWVyJ3MgbG9naWMgbGl2ZXMuXG4gICAqXG4gICAqIEBwYXJhbSBpbnB1dHMgSW5wdXQgdGVuc29yLCBvciBsaXN0L3R1cGxlIG9mIGlucHV0IHRlbnNvcnMuXG4gICAqIEBwYXJhbSBrd2FyZ3MgQWRkaXRpb25hbCBrZXl3b3JkIGFyZ3VtZW50cy5cbiAgICpcbiAgICogQHJldHVybiBBIHRlbnNvciBvciBsaXN0L3R1cGxlIG9mIHRlbnNvcnMuXG4gICAqL1xuICBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgcmV0dXJuIGlucHV0cztcbiAgfVxuXG4gIHByb3RlY3RlZCBpbnZva2VDYWxsSG9vayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwga3dhcmdzOiBLd2FyZ3MpIHtcbiAgICBpZiAodGhpcy5fY2FsbEhvb2sgIT0gbnVsbCkge1xuICAgICAgdGhpcy5fY2FsbEhvb2soaW5wdXRzLCBrd2FyZ3MpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBTZXQgY2FsbCBob29rLlxuICAgKiBUaGlzIGlzIGN1cnJlbnRseSB1c2VkIGZvciB0ZXN0aW5nIG9ubHkuXG4gICAqIEBwYXJhbSBjYWxsSG9va1xuICAgKi9cbiAgc2V0Q2FsbEhvb2soY2FsbEhvb2s6IENhbGxIb29rKSB7XG4gICAgdGhpcy5fY2FsbEhvb2sgPSBjYWxsSG9vaztcbiAgfVxuXG4gIC8qKlxuICAgKiBDbGVhciBjYWxsIGhvb2suXG4gICAqIFRoaXMgaXMgY3VycmVudGx5IHVzZWQgZm9yIHRlc3Rpbmcgb25seS5cbiAgICovXG4gIGNsZWFyQ2FsbEhvb2soKSB7XG4gICAgdGhpcy5fY2FsbEhvb2sgPSBudWxsO1xuICB9XG5cbiAgLyoqXG4gICAqIEJ1aWxkcyBvciBleGVjdXRlcyBhIGBMYXllcmAncyBsb2dpYy5cbiAgICpcbiAgICogV2hlbiBjYWxsZWQgd2l0aCBgdGYuVGVuc29yYChzKSwgZXhlY3V0ZSB0aGUgYExheWVyYCdzIGNvbXB1dGF0aW9uIGFuZFxuICAgKiByZXR1cm4gVGVuc29yKHMpLiBGb3IgZXhhbXBsZTpcbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgZGVuc2VMYXllciA9IHRmLmxheWVycy5kZW5zZSh7XG4gICAqICAgdW5pdHM6IDEsXG4gICAqICAga2VybmVsSW5pdGlhbGl6ZXI6ICd6ZXJvcycsXG4gICAqICAgdXNlQmlhczogZmFsc2VcbiAgICogfSk7XG4gICAqXG4gICAqIC8vIEludm9rZSB0aGUgbGF5ZXIncyBhcHBseSgpIG1ldGhvZCB3aXRoIGEgYHRmLlRlbnNvcmAgKHdpdGggY29uY3JldGVcbiAgICogLy8gbnVtZXJpYyB2YWx1ZXMpLlxuICAgKiBjb25zdCBpbnB1dCA9IHRmLm9uZXMoWzIsIDJdKTtcbiAgICogY29uc3Qgb3V0cHV0ID0gZGVuc2VMYXllci5hcHBseShpbnB1dCk7XG4gICAqXG4gICAqIC8vIFRoZSBvdXRwdXQncyB2YWx1ZSBpcyBleHBlY3RlZCB0byBiZSBbWzBdLCBbMF1dLCBkdWUgdG8gdGhlIGZhY3QgdGhhdFxuICAgKiAvLyB0aGUgZGVuc2UgbGF5ZXIgaGFzIGEga2VybmVsIGluaXRpYWxpemVkIHRvIGFsbC16ZXJvcyBhbmQgZG9lcyBub3QgaGF2ZVxuICAgKiAvLyBhIGJpYXMuXG4gICAqIG91dHB1dC5wcmludCgpO1xuICAgKiBgYGBcbiAgICpcbiAgICogV2hlbiBjYWxsZWQgd2l0aCBgdGYuU3ltYm9saWNUZW5zb3JgKHMpLCB0aGlzIHdpbGwgcHJlcGFyZSB0aGUgbGF5ZXIgZm9yXG4gICAqIGZ1dHVyZSBleGVjdXRpb24uICBUaGlzIGVudGFpbHMgaW50ZXJuYWwgYm9vay1rZWVwaW5nIG9uIHNoYXBlcyBvZlxuICAgKiBleHBlY3RlZCBUZW5zb3JzLCB3aXJpbmcgbGF5ZXJzIHRvZ2V0aGVyLCBhbmQgaW5pdGlhbGl6aW5nIHdlaWdodHMuXG4gICAqXG4gICAqIENhbGxpbmcgYGFwcGx5YCB3aXRoIGB0Zi5TeW1ib2xpY1RlbnNvcmBzIGFyZSB0eXBpY2FsbHkgdXNlZCBkdXJpbmcgdGhlXG4gICAqIGJ1aWxkaW5nIG9mIG5vbi1gdGYuU2VxdWVudGlhbGAgbW9kZWxzLiBGb3IgZXhhbXBsZTpcbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgZmxhdHRlbkxheWVyID0gdGYubGF5ZXJzLmZsYXR0ZW4oKTtcbiAgICogY29uc3QgZGVuc2VMYXllciA9IHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDF9KTtcbiAgICpcbiAgICogLy8gVXNlIHRmLmxheWVycy5pbnB1dCgpIHRvIG9idGFpbiBhIFN5bWJvbGljVGVuc29yIGFzIGlucHV0IHRvIGFwcGx5KCkuXG4gICAqIGNvbnN0IGlucHV0ID0gdGYuaW5wdXQoe3NoYXBlOiBbMiwgMl19KTtcbiAgICogY29uc3Qgb3V0cHV0MSA9IGZsYXR0ZW5MYXllci5hcHBseShpbnB1dCk7XG4gICAqXG4gICAqIC8vIG91dHB1dDEuc2hhcGUgaXMgW251bGwsIDRdLiBUaGUgZmlyc3QgZGltZW5zaW9uIGlzIHRoZSB1bmRldGVybWluZWRcbiAgICogLy8gYmF0Y2ggc2l6ZS4gVGhlIHNlY29uZCBkaW1lbnNpb24gY29tZXMgZnJvbSBmbGF0dGVuaW5nIHRoZSBbMiwgMl1cbiAgICogLy8gc2hhcGUuXG4gICAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dHB1dDEuc2hhcGUpKTtcbiAgICpcbiAgICogLy8gVGhlIG91dHB1dCBTeW1ib2xpY1RlbnNvciBvZiB0aGUgZmxhdHRlbiBsYXllciBjYW4gYmUgdXNlZCB0byBjYWxsXG4gICAqIC8vIHRoZSBhcHBseSgpIG9mIHRoZSBkZW5zZSBsYXllcjpcbiAgICogY29uc3Qgb3V0cHV0MiA9IGRlbnNlTGF5ZXIuYXBwbHkob3V0cHV0MSk7XG4gICAqXG4gICAqIC8vIG91dHB1dDIuc2hhcGUgaXMgW251bGwsIDFdLiBUaGUgZmlyc3QgZGltZW5zaW9uIGlzIHRoZSB1bmRldGVybWluZWRcbiAgICogLy8gYmF0Y2ggc2l6ZS4gVGhlIHNlY29uZCBkaW1lbnNpb24gbWF0Y2hlcyB0aGUgbnVtYmVyIG9mIHVuaXRzIG9mIHRoZVxuICAgKiAvLyBkZW5zZSBsYXllci5cbiAgICogY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkob3V0cHV0Mi5zaGFwZSkpO1xuICAgKlxuICAgKiAvLyBUaGUgaW5wdXQgYW5kIG91dHB1dCBjYW4gYmUgdXNlZCB0byBjb25zdHJ1Y3QgYSBtb2RlbCB0aGF0IGNvbnNpc3RzXG4gICAqIC8vIG9mIHRoZSBmbGF0dGVuIGFuZCBkZW5zZSBsYXllcnMuXG4gICAqIGNvbnN0IG1vZGVsID0gdGYubW9kZWwoe2lucHV0czogaW5wdXQsIG91dHB1dHM6IG91dHB1dDJ9KTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBpbnB1dHMgYSBgdGYuVGVuc29yYCBvciBgdGYuU3ltYm9saWNUZW5zb3JgIG9yIGFuIEFycmF5IG9mIHRoZW0uXG4gICAqIEBwYXJhbSBrd2FyZ3MgQWRkaXRpb25hbCBrZXl3b3JkIGFyZ3VtZW50cyB0byBiZSBwYXNzZWQgdG8gYGNhbGwoKWAuXG4gICAqXG4gICAqIEByZXR1cm4gT3V0cHV0IG9mIHRoZSBsYXllcidzIGBjYWxsYCBtZXRob2QuXG4gICAqXG4gICAqIEBleGNlcHRpb24gVmFsdWVFcnJvciBlcnJvciBpbiBjYXNlIHRoZSBsYXllciBpcyBtaXNzaW5nIHNoYXBlIGluZm9ybWF0aW9uXG4gICAqICAgZm9yIGl0cyBgYnVpbGRgIGNhbGwuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCAnc3ViaGVhZGluZyc6ICdDbGFzc2VzJ31cbiAgICovXG4gIC8vIFBvcnRpbmcgTm90ZTogVGhpcyBpcyBhIHJlcGxhY2VtZW50IGZvciBfX2NhbGxfXygpIGluIFB5dGhvbi5cbiAgYXBwbHkoXG4gICAgICBpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXXxTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdLFxuICAgICAga3dhcmdzPzogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdfFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW10ge1xuICAgIGt3YXJncyA9IGt3YXJncyB8fCB7fTtcblxuICAgIHRoaXMuYXNzZXJ0Tm90RGlzcG9zZWQoKTtcblxuICAgIC8vIEVuc3VyZSBpbnB1dHMgYXJlIGFsbCB0aGUgc2FtZSB0eXBlLlxuICAgIGNvbnN0IGlucHV0c0xpc3QgPSBnZW5lcmljX3V0aWxzLnRvTGlzdChpbnB1dHMpO1xuXG4gICAgY29uc3QgYWxsQXJlU3ltYm9saWMgPSBjaGVja0FsbFN5bWJvbGljKGlucHV0cyk7XG4gICAgY29uc3Qgbm9uZUFyZVN5bWJvbGljID0gY2hlY2tOb25lU3ltYm9saWMoaW5wdXRzKTtcblxuICAgIGlmIChhbGxBcmVTeW1ib2xpYyA9PT0gbm9uZUFyZVN5bWJvbGljKSB7XG4gICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAnQXJndW1lbnRzIHRvIGFwcGx5KCkgbXVzdCBiZSBhbGwgJyArXG4gICAgICAgICAgJ1N5bWJvbGljVGVuc29ycyBvciBhbGwgVGVuc29ycycpO1xuICAgIH1cblxuICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogbmFtZVNjb3BlKCkgbWF5IG5vdCBiZSBuZWNlc3NhcnkuXG4gICAgcmV0dXJuIG5hbWVTY29wZSh0aGlzLm5hbWUsICgpID0+IHtcbiAgICAgIC8vIEhhbmRsZSBsYXlpbmcgYnVpbGRpbmcgKHdlaWdodCBjcmVhdGluZywgaW5wdXQgc3BlYyBsb2NraW5nKS5cbiAgICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgICAvKlxuICAgICAgICAgIFRocm93IGV4Y2VwdGlvbnMgaW4gY2FzZSB0aGUgaW5wdXQgaXMgbm90IGNvbXBhdGlibGVcbiAgICAgICAgICB3aXRoIHRoZSBpbnB1dFNwZWMgc3BlY2lmaWVkIGluIHRoZSBsYXllciBjb25zdHJ1Y3Rvci5cbiAgICAgICAgICovXG4gICAgICAgIHRoaXMuYXNzZXJ0SW5wdXRDb21wYXRpYmlsaXR5KGlucHV0cyk7XG5cbiAgICAgICAgLy8gQ29sbGVjdCBpbnB1dCBzaGFwZXMgdG8gYnVpbGQgbGF5ZXIuXG4gICAgICAgIGNvbnN0IGlucHV0U2hhcGVzOiBTaGFwZVtdID0gW107XG4gICAgICAgIGZvciAoY29uc3QgeEVsZW0gb2YgZ2VuZXJpY191dGlscy50b0xpc3QoaW5wdXRzKSkge1xuICAgICAgICAgIGlucHV0U2hhcGVzLnB1c2goeEVsZW0uc2hhcGUpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMuYnVpbGQoZ2VuZXJpY191dGlscy5zaW5nbGV0b25PckFycmF5KGlucHV0U2hhcGVzKSk7XG4gICAgICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuXG4gICAgICAgIC8vIExvYWQgd2VpZ2h0cyB0aGF0IHdlcmUgc3BlY2lmaWVkIGF0IGxheWVyIGluc3RhbnRpYXRpb24uXG4gICAgICAgIGlmICh0aGlzLmluaXRpYWxXZWlnaHRzKSB7XG4gICAgICAgICAgdGhpcy5zZXRXZWlnaHRzKHRoaXMuaW5pdGlhbFdlaWdodHMpO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKHRoaXMuX3JlZkNvdW50ID09PSBudWxsICYmIG5vbmVBcmVTeW1ib2xpYykge1xuICAgICAgICAgIC8vIFRoZSBmaXJzdCB1c2Ugb2YgdGhpcyBsYXllciBpcyBhIG5vbi1zeW1ib2xpYyBjYWxsLCBzZXQgcmVmIGNvdW50XG4gICAgICAgICAgLy8gdG8gMSBzbyB0aGUgTGF5ZXIgY2FuIGJlIHByb3Blcmx5IGRpc3Bvc2VkIGlmIGl0cyBkaXNwb3NlKCkgbWV0aG9kXG4gICAgICAgICAgLy8gaXMgY2FsbGVkLlxuICAgICAgICAgIHRoaXMuX3JlZkNvdW50ID0gMTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICAvKlxuICAgICAgICBUaHJvdyBleGNlcHRpb25zIGluIGNhc2UgdGhlIGlucHV0IGlzIG5vdCBjb21wYXRpYmxlXG4gICAgICAgIHdpdGggdGhlIGlucHV0U3BlYyBzZXQgYXQgYnVpbGQgdGltZS5cbiAgICAgICovXG4gICAgICB0aGlzLmFzc2VydElucHV0Q29tcGF0aWJpbGl0eShpbnB1dHMpO1xuXG4gICAgICAvLyBIYW5kbGUgbWFzayBwcm9wYWdhdGlvbi5cbiAgICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5KTogTWFzayBwcm9wYWdhdGlvbiBub3QgY3VycmVudGx5IGltcGxlbWVudGVkLlxuXG4gICAgICAvLyBBY3R1YWxseSBjYWxsIHRoZSBsYXllciwgY29sbGVjdGluZyBvdXRwdXQocyksIG1hc2socyksIGFuZCBzaGFwZShzKS5cbiAgICAgIGlmIChub25lQXJlU3ltYm9saWMpIHtcbiAgICAgICAgbGV0IG91dHB1dCA9IHRoaXMuY2FsbChpbnB1dHMsIGt3YXJncyk7XG5cbiAgICAgICAgLy8gQXBwbHkgbWFza3MgdG8gdGhlIG91dHB1dCB0ZW5zb3JzIGlmIHRoZSBsYXllciBzdXBwb3J0cyBpdC5cbiAgICAgICAgaWYgKHRoaXMuc3VwcG9ydHNNYXNraW5nKSB7XG4gICAgICAgICAgLy8gVE9ETyhtYXR0c291bGFuaWxsZSk6IHBhc3MgdGhlIGlucHV0IHRlbnNvcnMnIG1hc2tzIHRvIGNvbXB1dGVNYXNrXG4gICAgICAgICAgdGhpcy5zZXRNYXNrTWV0YWRhdGEoaW5wdXRzLCBvdXRwdXQpO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gSWYgdGhlIGxheWVyIHJldHVybnMgdGVuc29ycyBmcm9tIGl0cyBpbnB1dHMsIHVubW9kaWZpZWQsXG4gICAgICAgIC8vIHdlIGNvcHkgdGhlbSB0byBhdm9pZCBsb3NzIG9mIHRlbnNvciBtZXRhZGF0YS5cbiAgICAgICAgY29uc3Qgb3V0cHV0TGlzdDogVGVuc29yW10gPSBnZW5lcmljX3V0aWxzLnRvTGlzdChvdXRwdXQpO1xuICAgICAgICBjb25zdCBvdXRwdXRMaXN0Q29weTogVGVuc29yW10gPSBbXTtcbiAgICAgICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBUaGlzIGNvcHlpbmcgbWF5IG5vdCBiZSBuZWNlc3NhcnkgZ2l2ZW4gb3VyIGVhZ2VyXG4gICAgICAgIC8vIGJhY2tlbmQuXG4gICAgICAgIGZvciAobGV0IHggb2Ygb3V0cHV0TGlzdCkge1xuICAgICAgICAgIGlmIChpbnB1dHNMaXN0LmluZGV4T2YoeCkgIT09IC0xKSB7XG4gICAgICAgICAgICB4ID0geC5jbG9uZSgpO1xuICAgICAgICAgIH1cbiAgICAgICAgICBvdXRwdXRMaXN0Q29weS5wdXNoKHgpO1xuICAgICAgICB9XG4gICAgICAgIG91dHB1dCA9IGdlbmVyaWNfdXRpbHMuc2luZ2xldG9uT3JBcnJheShvdXRwdXRMaXN0Q29weSk7XG5cbiAgICAgICAgaWYgKHRoaXMuYWN0aXZpdHlSZWd1bGFyaXplciAhPSBudWxsKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgICAgICdMYXllciBpbnZvY2F0aW9uIGluIHRoZSBwcmVzZW5jZSBvZiBhY3Rpdml0eSAnICtcbiAgICAgICAgICAgICAgJ3JlZ3VsYXJpemVyKHMpIGlzIG5vdCBzdXBwb3J0ZWQgeWV0LicpO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBDYWxsIGFkZEluYm91bmROb2RlKCk/XG4gICAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb25zdCBpbnB1dFNoYXBlID0gY29sbGVjdElucHV0U2hhcGUoaW5wdXRzKTtcbiAgICAgICAgY29uc3Qgb3V0cHV0U2hhcGUgPSB0aGlzLmNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlKTtcbiAgICAgICAgbGV0IG91dHB1dDogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXTtcbiAgICAgICAgY29uc3Qgb3V0cHV0RFR5cGUgPSBndWVzc091dHB1dERUeXBlKGlucHV0cyk7XG4gICAgICAgIHRoaXMud2Fybk9uSW5jb21wYXRpYmxlSW5wdXRTaGFwZShcbiAgICAgICAgICAgIEFycmF5LmlzQXJyYXkoaW5wdXRzKSA/IGlucHV0U2hhcGVbMF0gYXMgU2hhcGUgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW5wdXRTaGFwZSBhcyBTaGFwZSk7XG5cbiAgICAgICAgaWYgKG91dHB1dFNoYXBlICE9IG51bGwgJiYgb3V0cHV0U2hhcGUubGVuZ3RoID4gMCAmJlxuICAgICAgICAgICAgQXJyYXkuaXNBcnJheShvdXRwdXRTaGFwZVswXSkpIHtcbiAgICAgICAgICAvLyBXZSBoYXZlIG11bHRpcGxlIG91dHB1dCBzaGFwZXMuIENyZWF0ZSBtdWx0aXBsZSBvdXRwdXQgdGVuc29ycy5cbiAgICAgICAgICBvdXRwdXQgPSAob3V0cHV0U2hhcGUgYXMgU2hhcGVbXSlcbiAgICAgICAgICAgICAgICAgICAgICAgLm1hcChcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIChzaGFwZSwgaW5kZXgpID0+IG5ldyBTeW1ib2xpY1RlbnNvcihcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBvdXRwdXREVHlwZSwgc2hhcGUsIHRoaXMsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZ2VuZXJpY191dGlscy50b0xpc3QoaW5wdXRzKSwga3dhcmdzLCB0aGlzLm5hbWUsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW5kZXgpKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBvdXRwdXQgPSBuZXcgU3ltYm9saWNUZW5zb3IoXG4gICAgICAgICAgICAgIG91dHB1dERUeXBlLCBvdXRwdXRTaGFwZSBhcyBTaGFwZSwgdGhpcyxcbiAgICAgICAgICAgICAgZ2VuZXJpY191dGlscy50b0xpc3QoaW5wdXRzKSwga3dhcmdzLCB0aGlzLm5hbWUpO1xuICAgICAgICB9XG5cbiAgICAgICAgLypcbiAgICAgICAgICBBZGQgYW4gaW5ib3VuZCBub2RlIHRvIHRoZSBsYXllciwgc28gdGhhdCBpdCBrZWVwcyB0cmFja1xuICAgICAgICAgIG9mIHRoZSBjYWxsIGFuZCBvZiBhbGwgbmV3IHZhcmlhYmxlcyBjcmVhdGVkIGR1cmluZyB0aGUgY2FsbC5cbiAgICAgICAgICBUaGlzIGFsc28gdXBkYXRlcyB0aGUgbGF5ZXIgaGlzdG9yeSBvZiB0aGUgb3V0cHV0IHRlbnNvcihzKS5cbiAgICAgICAgICBJZiB0aGUgaW5wdXQgdGVuc29yKHMpIGhhZCBubyBwcmV2aW91cyBoaXN0b3J5LFxuICAgICAgICAgIHRoaXMgZG9lcyBub3RoaW5nLlxuICAgICAgICAqL1xuICAgICAgICB0aGlzLmFkZEluYm91bmROb2RlKGlucHV0cywgb3V0cHV0LCBudWxsLCBudWxsLFxuICAgICAgICAgICAgaW5wdXRTaGFwZSwgb3V0cHV0U2hhcGUsIGt3YXJncyk7XG4gICAgICAgIHRoaXMuX3JlZkNvdW50Kys7XG5cbiAgICAgICAgaWYgKHRoaXMuYWN0aXZpdHlSZWd1bGFyaXplciAhPSBudWxsKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgICAgICdMYXllciBpbnZvY2F0aW9uIGluIHRoZSBwcmVzZW5jZSBvZiBhY3Rpdml0eSAnICtcbiAgICAgICAgICAgICAgJ3JlZ3VsYXJpemVyKHMpIGlzIG5vdCBzdXBwb3J0ZWQgeWV0LicpO1xuICAgICAgICB9XG5cbiAgICAgICAgcmV0dXJuIG91dHB1dDtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDaGVjayBjb21wYXRpYmlsaXR5IGJldHdlZW4gaW5wdXQgc2hhcGUgYW5kIHRoaXMgbGF5ZXIncyBiYXRjaElucHV0U2hhcGUuXG4gICAqXG4gICAqIFByaW50IHdhcm5pbmcgaWYgYW55IGluY29tcGF0aWJpbGl0eSBpcyBmb3VuZC5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0U2hhcGUgSW5wdXQgc2hhcGUgdG8gYmUgY2hlY2tlZC5cbiAgICovXG4gIHByb3RlY3RlZCB3YXJuT25JbmNvbXBhdGlibGVJbnB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlKSB7XG4gICAgaWYgKHRoaXMuYmF0Y2hJbnB1dFNoYXBlID09IG51bGwpIHtcbiAgICAgIHJldHVybjtcbiAgICB9IGVsc2UgaWYgKGlucHV0U2hhcGUubGVuZ3RoICE9PSB0aGlzLmJhdGNoSW5wdXRTaGFwZS5sZW5ndGgpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgVGhlIHJhbmsgb2YgdGhlIGlucHV0IHRlbnNvciBwcm92aWRlZCAoc2hhcGU6IGAgK1xuICAgICAgICAgIGAke0pTT04uc3RyaW5naWZ5KGlucHV0U2hhcGUpfSkgZG9lcyBub3QgbWF0Y2ggdGhhdCBvZiB0aGUgYCArXG4gICAgICAgICAgYGJhdGNoSW5wdXRTaGFwZSAoJHtKU09OLnN0cmluZ2lmeSh0aGlzLmJhdGNoSW5wdXRTaGFwZSl9KSBgICtcbiAgICAgICAgICBgb2YgdGhlIGxheWVyICR7dGhpcy5uYW1lfWApO1xuICAgIH0gZWxzZSB7XG4gICAgICBsZXQgZGltTWlzbWF0Y2ggPSBmYWxzZTtcbiAgICAgIHRoaXMuYmF0Y2hJbnB1dFNoYXBlLmZvckVhY2goKGRpbWVuc2lvbiwgaSkgPT4ge1xuICAgICAgICBpZiAoZGltZW5zaW9uICE9IG51bGwgJiYgaW5wdXRTaGFwZVtpXSAhPSBudWxsICYmXG4gICAgICAgICAgICBpbnB1dFNoYXBlW2ldICE9PSBkaW1lbnNpb24pIHtcbiAgICAgICAgICBkaW1NaXNtYXRjaCA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgICAgaWYgKGRpbU1pc21hdGNoKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBUaGUgc2hhcGUgb2YgdGhlIGlucHV0IHRlbnNvciBgICtcbiAgICAgICAgICAgIGAoJHtKU09OLnN0cmluZ2lmeShpbnB1dFNoYXBlKX0pIGRvZXMgbm90IGAgK1xuICAgICAgICAgICAgYG1hdGNoIHRoZSBleHBlY3RhdGlvbiBvZiBsYXllciAke3RoaXMubmFtZX06IGAgK1xuICAgICAgICAgICAgYCR7SlNPTi5zdHJpbmdpZnkodGhpcy5iYXRjaElucHV0U2hhcGUpfWApO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBSZXRyaWV2ZXMgdGhlIG91dHB1dCBzaGFwZShzKSBvZiBhIGxheWVyLlxuICAgKlxuICAgKiBPbmx5IGFwcGxpY2FibGUgaWYgdGhlIGxheWVyIGhhcyBvbmx5IG9uZSBpbmJvdW5kIG5vZGUsIG9yIGlmIGFsbCBpbmJvdW5kXG4gICAqIG5vZGVzIGhhdmUgdGhlIHNhbWUgb3V0cHV0IHNoYXBlLlxuICAgKlxuICAgKiBAcmV0dXJucyBPdXRwdXQgc2hhcGUgb3Igc2hhcGVzLlxuICAgKiBAdGhyb3dzIEF0dHJpYnV0ZUVycm9yOiBpZiB0aGUgbGF5ZXIgaXMgY29ubmVjdGVkIHRvIG1vcmUgdGhhbiBvbmUgaW5jb21pbmdcbiAgICogICBub2Rlcy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsICdzdWJoZWFkaW5nJzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgZ2V0IG91dHB1dFNoYXBlKCk6IFNoYXBlfFNoYXBlW10ge1xuICAgIGlmICh0aGlzLmluYm91bmROb2RlcyA9PSBudWxsIHx8IHRoaXMuaW5ib3VuZE5vZGVzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhyb3cgbmV3IEF0dHJpYnV0ZUVycm9yKFxuICAgICAgICAgIGBUaGUgbGF5ZXIgJHt0aGlzLm5hbWV9IGhhcyBuZXZlciBiZWVuIGNhbGxlZCBhbmQgdGh1cyBoYXMgbm8gYCArXG4gICAgICAgICAgYGRlZmluZWQgb3V0cHV0IHNoYXBlLmApO1xuICAgIH1cbiAgICBjb25zdCBhbGxPdXRwdXRTaGFwZXM6IHN0cmluZ1tdID0gW107XG4gICAgZm9yIChjb25zdCBub2RlIG9mIHRoaXMuaW5ib3VuZE5vZGVzKSB7XG4gICAgICBjb25zdCBzaGFwZVN0cmluZyA9IEpTT04uc3RyaW5naWZ5KG5vZGUub3V0cHV0U2hhcGVzKTtcbiAgICAgIGlmIChhbGxPdXRwdXRTaGFwZXMuaW5kZXhPZihzaGFwZVN0cmluZykgPT09IC0xKSB7XG4gICAgICAgIGFsbE91dHB1dFNoYXBlcy5wdXNoKHNoYXBlU3RyaW5nKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGFsbE91dHB1dFNoYXBlcy5sZW5ndGggPT09IDEpIHtcbiAgICAgIGNvbnN0IG91dHB1dFNoYXBlcyA9IHRoaXMuaW5ib3VuZE5vZGVzWzBdLm91dHB1dFNoYXBlcztcbiAgICAgIGlmIChBcnJheS5pc0FycmF5KG91dHB1dFNoYXBlcykgJiYgQXJyYXkuaXNBcnJheShvdXRwdXRTaGFwZXNbMF0pICYmXG4gICAgICAgICAgb3V0cHV0U2hhcGVzLmxlbmd0aCA9PT0gMSkge1xuICAgICAgICByZXR1cm4gKG91dHB1dFNoYXBlcyBhcyBTaGFwZVtdKVswXTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBvdXRwdXRTaGFwZXM7XG4gICAgICB9XG5cbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IEF0dHJpYnV0ZUVycm9yKFxuICAgICAgICAgIGBUaGUgbGF5ZXIgJHt0aGlzLm5hbWV9IGhhcyBtdWx0aXBsZSBpbmJvdW5kIG5vZGVzIHdpdGggZGlmZmVyZW50IGAgK1xuICAgICAgICAgIGBvdXRwdXQgc2hhcGVzLiBIZW5jZSB0aGUgbm90aW9uIG9mIFwib3V0cHV0IHNoYXBlXCIgaXMgaWxsLWRlZmluZWQgYCArXG4gICAgICAgICAgYGZvciB0aGUgbGF5ZXIuYCk7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBJbXBsZW1lbnQgZ2V0T3V0cHV0U2hhcGVBdCgpLlxuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBDb3VudHMgdGhlIHRvdGFsIG51bWJlciBvZiBudW1iZXJzIChlLmcuLCBmbG9hdDMyLCBpbnQzMikgaW4gdGhlXG4gICAqIHdlaWdodHMuXG4gICAqXG4gICAqIEByZXR1cm5zIEFuIGludGVnZXIgY291bnQuXG4gICAqIEB0aHJvd3MgUnVudGltZUVycm9yOiBJZiB0aGUgbGF5ZXIgaXMgbm90IGJ1aWx0IHlldCAoaW4gd2hpY2ggY2FzZSBpdHNcbiAgICogICB3ZWlnaHRzIGFyZSBub3QgZGVmaW5lZCB5ZXQuKVxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgJ3N1YmhlYWRpbmcnOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBjb3VudFBhcmFtcygpOiBudW1iZXIge1xuICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgdGhyb3cgbmV3IFJ1bnRpbWVFcnJvcihcbiAgICAgICAgICBgWW91IHRyaWVkIHRvIGNhbGwgY291bnRQYXJhbXMoKSBvbiAke3RoaXMubmFtZX0sIGAgK1xuICAgICAgICAgIGBidXQgdGhlIGxheWVyIGlzIG5vdCBidWlsdCB5ZXQuIEJ1aWxkIGl0IGZpcnN0IGJ5IGNhbGxpbmcgYCArXG4gICAgICAgICAgYGJ1aWxkKGJhdGNoSW5wdXRTaGFwZSkuYCk7XG4gICAgfVxuICAgIHJldHVybiB2YXJpYWJsZV91dGlscy5jb3VudFBhcmFtc0luV2VpZ2h0cyh0aGlzLndlaWdodHMpO1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZXMgdGhlIGxheWVyIHdlaWdodHMuXG4gICAqXG4gICAqIE11c3QgYmUgaW1wbGVtZW50ZWQgb24gYWxsIGxheWVycyB0aGF0IGhhdmUgd2VpZ2h0cy5cbiAgICpcbiAgICogQ2FsbGVkIHdoZW4gYXBwbHkoKSBpcyBjYWxsZWQgdG8gY29uc3RydWN0IHRoZSB3ZWlnaHRzLlxuICAgKlxuICAgKiBAcGFyYW0gaW5wdXRTaGFwZSBBIGBTaGFwZWAgb3IgYXJyYXkgb2YgYFNoYXBlYCAodW51c2VkKS5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsICdzdWJoZWFkaW5nJzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSkge1xuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIGN1cnJlbnQgdmFsdWVzIG9mIHRoZSB3ZWlnaHRzIG9mIHRoZSBsYXllci5cbiAgICpcbiAgICogQHBhcmFtIHRyYWluYWJsZU9ubHkgV2hldGhlciB0byBnZXQgdGhlIHZhbHVlcyBvZiBvbmx5IHRyYWluYWJsZSB3ZWlnaHRzLlxuICAgKiBAcmV0dXJucyBXZWlnaHQgdmFsdWVzIGFzIGFuIGBBcnJheWAgb2YgYHRmLlRlbnNvcmBzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgJ3N1YmhlYWRpbmcnOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBnZXRXZWlnaHRzKHRyYWluYWJsZU9ubHkgPSBmYWxzZSk6IFRlbnNvcltdIHtcbiAgICByZXR1cm4gYmF0Y2hHZXRWYWx1ZSh0cmFpbmFibGVPbmx5ID8gdGhpcy50cmFpbmFibGVXZWlnaHRzIDogdGhpcy53ZWlnaHRzKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXRzIHRoZSB3ZWlnaHRzIG9mIHRoZSBsYXllciwgZnJvbSBUZW5zb3JzLlxuICAgKlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBhIGxpc3Qgb2YgVGVuc29ycy4gVGhlIG51bWJlciBvZiBhcnJheXMgYW5kIHRoZWlyIHNoYXBlXG4gICAqICAgbXVzdCBtYXRjaCBudW1iZXIgb2YgdGhlIGRpbWVuc2lvbnMgb2YgdGhlIHdlaWdodHMgb2YgdGhlIGxheWVyIChpLmUuXG4gICAqICAgaXQgc2hvdWxkIG1hdGNoIHRoZSBvdXRwdXQgb2YgYGdldFdlaWdodHNgKS5cbiAgICpcbiAgICogQGV4Y2VwdGlvbiBWYWx1ZUVycm9yIElmIHRoZSBwcm92aWRlZCB3ZWlnaHRzIGxpc3QgZG9lcyBub3QgbWF0Y2ggdGhlXG4gICAqICAgbGF5ZXIncyBzcGVjaWZpY2F0aW9ucy5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsICdzdWJoZWFkaW5nJzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgc2V0V2VpZ2h0cyh3ZWlnaHRzOiBUZW5zb3JbXSk6IHZvaWQge1xuICAgIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgcGFyYW1zID0gdGhpcy53ZWlnaHRzO1xuICAgICAgaWYgKHBhcmFtcy5sZW5ndGggIT09IHdlaWdodHMubGVuZ3RoKSB7XG4gICAgICAgIC8vIFRPRE8oY2Fpcyk6IFJlc3RvcmUgdGhlIGZvbGxvd2luZyBhbmQgdXNlIGBwcm92aWRlZFdlaWdodHNgLCBpbnN0ZWFkXG4gICAgICAgIC8vIG9mIGB3ZWlnaHRzYCBpbiB0aGUgZXJyb3IgbWVzc2FnZSwgb25jZSB0aGUgZGVlcGxlYXJuLmpzIGJ1ZyBpc1xuICAgICAgICAvLyBmaXhlZDogaHR0cHM6Ly9naXRodWIuY29tL1BBSVItY29kZS9kZWVwbGVhcm5qcy9pc3N1ZXMvNDk4IGNvbnN0XG4gICAgICAgIC8vIHByb3ZpZGVkV2VpZ2h0cyA9IEpTT04uc3RyaW5naWZ5KHdlaWdodHMpLnNsaWNlKDAsIDUwKTtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICBgWW91IGNhbGxlZCBzZXRXZWlnaHRzKHdlaWdodHMpIG9uIGxheWVyIFwiJHt0aGlzLm5hbWV9XCIgYCArXG4gICAgICAgICAgICBgd2l0aCBhIHdlaWdodCBsaXN0IG9mIGxlbmd0aCAke3dlaWdodHMubGVuZ3RofSwgYCArXG4gICAgICAgICAgICBgYnV0IHRoZSBsYXllciB3YXMgZXhwZWN0aW5nICR7cGFyYW1zLmxlbmd0aH0gd2VpZ2h0cy4gYCArXG4gICAgICAgICAgICBgUHJvdmlkZWQgd2VpZ2h0czogJHt3ZWlnaHRzfS4uLmApO1xuICAgICAgfVxuICAgICAgaWYgKHBhcmFtcy5sZW5ndGggPT09IDApIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgY29uc3Qgd2VpZ2h0VmFsdWVUdXBsZXM6IEFycmF5PFtMYXllclZhcmlhYmxlLCBUZW5zb3JdPiA9IFtdO1xuICAgICAgY29uc3QgcGFyYW1WYWx1ZXMgPSBiYXRjaEdldFZhbHVlKHBhcmFtcyk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHBhcmFtVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGNvbnN0IHB2ID0gcGFyYW1WYWx1ZXNbaV07XG4gICAgICAgIGNvbnN0IHAgPSBwYXJhbXNbaV07XG4gICAgICAgIGNvbnN0IHcgPSB3ZWlnaHRzW2ldO1xuICAgICAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwocHYuc2hhcGUsIHcuc2hhcGUpKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgICAgIGBMYXllciB3ZWlnaHQgc2hhcGUgJHtwdi5zaGFwZX0gYCArXG4gICAgICAgICAgICAgIGBub3QgY29tcGF0aWJsZSB3aXRoIHByb3ZpZGVkIHdlaWdodCBzaGFwZSAke3cuc2hhcGV9YCk7XG4gICAgICAgIH1cbiAgICAgICAgd2VpZ2h0VmFsdWVUdXBsZXMucHVzaChbcCwgd10pO1xuICAgICAgfVxuICAgICAgYmF0Y2hTZXRWYWx1ZSh3ZWlnaHRWYWx1ZVR1cGxlcyk7XG4gICAgfSk7XG4gIH1cblxuICAvKipcbiAgICogQWRkcyBhIHdlaWdodCB2YXJpYWJsZSB0byB0aGUgbGF5ZXIuXG4gICAqXG4gICAqIEBwYXJhbSBuYW1lIE5hbWUgb2YgdGhlIG5ldyB3ZWlnaHQgdmFyaWFibGUuXG4gICAqIEBwYXJhbSBzaGFwZSBUaGUgc2hhcGUgb2YgdGhlIHdlaWdodC5cbiAgICogQHBhcmFtIGR0eXBlIFRoZSBkdHlwZSBvZiB0aGUgd2VpZ2h0LlxuICAgKiBAcGFyYW0gaW5pdGlhbGl6ZXIgQW4gaW5pdGlhbGl6ZXIgaW5zdGFuY2UuXG4gICAqIEBwYXJhbSByZWd1bGFyaXplciBBIHJlZ3VsYXJpemVyIGluc3RhbmNlLlxuICAgKiBAcGFyYW0gdHJhaW5hYmxlIFdoZXRoZXIgdGhlIHdlaWdodCBzaG91bGQgYmUgdHJhaW5lZCB2aWEgYmFja3Byb3Agb3Igbm90XG4gICAqICAgKGFzc3VtaW5nIHRoYXQgdGhlIGxheWVyIGl0c2VsZiBpcyBhbHNvIHRyYWluYWJsZSkuXG4gICAqIEBwYXJhbSBjb25zdHJhaW50IEFuIG9wdGlvbmFsIHRyYWluYWJsZS5cbiAgICogQHJldHVybiBUaGUgY3JlYXRlZCB3ZWlnaHQgdmFyaWFibGUuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCAnc3ViaGVhZGluZyc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHByb3RlY3RlZCBhZGRXZWlnaHQoXG4gICAgICBuYW1lOiBzdHJpbmcsIHNoYXBlOiBTaGFwZSwgZHR5cGU/OiBEYXRhVHlwZSwgaW5pdGlhbGl6ZXI/OiBJbml0aWFsaXplcixcbiAgICAgIHJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXIsIHRyYWluYWJsZT86IGJvb2xlYW4sIGNvbnN0cmFpbnQ/OiBDb25zdHJhaW50LFxuICAgICAgZ2V0SW5pdGlhbGl6ZXJGdW5jPzogRnVuY3Rpb24pOiBMYXllclZhcmlhYmxlIHtcbiAgICAvLyBSZWplY3QgZHVwbGljYXRlIHdlaWdodCBuYW1lcy5cbiAgICBpZiAodGhpcy5fYWRkZWRXZWlnaHROYW1lcy5pbmRleE9mKG5hbWUpICE9PSAtMSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYER1cGxpY2F0ZSB3ZWlnaHQgbmFtZSAke25hbWV9IGZvciBsYXllciAke3RoaXMubmFtZX1gKTtcbiAgICB9XG4gICAgdGhpcy5fYWRkZWRXZWlnaHROYW1lcy5wdXNoKG5hbWUpO1xuXG4gICAgaWYgKGR0eXBlID09IG51bGwpIHtcbiAgICAgIGR0eXBlID0gJ2Zsb2F0MzInO1xuICAgIH1cblxuICAgIGlmICh0aGlzLmZhc3RXZWlnaHRJbml0RHVyaW5nQnVpbGQpIHtcbiAgICAgIGluaXRpYWxpemVyID0gZ2V0SW5pdGlhbGl6ZXJGdW5jICE9IG51bGwgPyBnZXRJbml0aWFsaXplckZ1bmMoKSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZ2V0SW5pdGlhbGl6ZXIoJ3plcm9zJyk7XG4gICAgfVxuICAgIGNvbnN0IGluaXRWYWx1ZSA9IGluaXRpYWxpemVyLmFwcGx5KHNoYXBlLCBkdHlwZSk7XG4gICAgY29uc3Qgd2VpZ2h0ID1cbiAgICAgICAgbmV3IExheWVyVmFyaWFibGUoaW5pdFZhbHVlLCBkdHlwZSwgbmFtZSwgdHJhaW5hYmxlLCBjb25zdHJhaW50KTtcbiAgICBpbml0VmFsdWUuZGlzcG9zZSgpO1xuICAgIC8vIFJlcXVlc3QgYmFja2VuZCBub3QgdG8gZGlzcG9zZSB0aGUgd2VpZ2h0cyBvZiB0aGUgbW9kZWwgb24gc2NvcGUoKSBleGl0LlxuICAgIGlmIChyZWd1bGFyaXplciAhPSBudWxsKSB7XG4gICAgICB0aGlzLmFkZExvc3MoKCkgPT4gcmVndWxhcml6ZXIuYXBwbHkod2VpZ2h0LnJlYWQoKSkpO1xuICAgIH1cbiAgICBpZiAodHJhaW5hYmxlID09IG51bGwpIHtcbiAgICAgIHRyYWluYWJsZSA9IHRydWU7XG4gICAgfVxuICAgIGlmICh0cmFpbmFibGUpIHtcbiAgICAgIHRoaXMuX3RyYWluYWJsZVdlaWdodHMucHVzaCh3ZWlnaHQpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLl9ub25UcmFpbmFibGVXZWlnaHRzLnB1c2god2VpZ2h0KTtcbiAgICB9XG4gICAgcmV0dXJuIHdlaWdodDtcbiAgfVxuXG4gIC8qKlxuICAgKiBTZXQgdGhlIGZhc3Qtd2VpZ2h0LWluaXRpYWxpemF0aW9uIGZsYWcuXG4gICAqXG4gICAqIEluIGNhc2VzIHdoZXJlIHRoZSBpbml0aWFsaXplZCB3ZWlnaHQgdmFsdWVzIHdpbGwgYmUgaW1tZWRpYXRlbHlcbiAgICogb3ZlcndyaXR0ZW4gYnkgbG9hZGVkIHdlaWdodCB2YWx1ZXMgZHVyaW5nIG1vZGVsIGxvYWRpbmcsIHNldHRpbmdcbiAgICogdGhlIGZsYWcgdG8gYHRydWVgIHNhdmVzIHVubmVjZXNzYXJ5IGNhbGxzIHRvIHBvdGVudGlhbGx5IGV4cGVuc2l2ZVxuICAgKiBpbml0aWFsaXplcnMgYW5kIHNwZWVkcyB1cCB0aGUgbG9hZGluZyBwcm9jZXNzLlxuICAgKlxuICAgKiBAcGFyYW0gdmFsdWUgVGFyZ2V0IHZhbHVlIG9mIHRoZSBmbGFnLlxuICAgKi9cbiAgc2V0RmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCh2YWx1ZTogYm9vbGVhbikge1xuICAgIHRoaXMuZmFzdFdlaWdodEluaXREdXJpbmdCdWlsZCA9IHZhbHVlO1xuICB9XG5cbiAgLyoqXG4gICAqIEFkZCBsb3NzZXMgdG8gdGhlIGxheWVyLlxuICAgKlxuICAgKiBUaGUgbG9zcyBtYXkgcG90ZW50aWFsbHkgYmUgY29uZGl0aW9uYWwgb24gc29tZSBpbnB1dHMgdGVuc29ycyxcbiAgICogZm9yIGluc3RhbmNlIGFjdGl2aXR5IGxvc3NlcyBhcmUgY29uZGl0aW9uYWwgb24gdGhlIGxheWVyJ3MgaW5wdXRzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgJ3N1YmhlYWRpbmcnOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhZGRMb3NzKGxvc3NlczogUmVndWxhcml6ZXJGbnxSZWd1bGFyaXplckZuW10pOiB2b2lkIHtcbiAgICBpZiAobG9zc2VzID09IG51bGwgfHwgQXJyYXkuaXNBcnJheShsb3NzZXMpICYmIGxvc3Nlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgLy8gVXBkYXRlIHRoaXMubG9zc2VzXG4gICAgbG9zc2VzID0gZ2VuZXJpY191dGlscy50b0xpc3QobG9zc2VzKTtcbiAgICBpZiAodGhpcy5fbG9zc2VzICE9PSB1bmRlZmluZWQgJiYgdGhpcy5fbG9zc2VzICE9PSBudWxsKSB7XG4gICAgICB0aGlzLmxvc3Nlcy5wdXNoKC4uLmxvc3Nlcyk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBvdXRwdXQgc2hhcGUgb2YgdGhlIGxheWVyLlxuICAgKlxuICAgKiBBc3N1bWVzIHRoYXQgdGhlIGxheWVyIHdpbGwgYmUgYnVpbHQgdG8gbWF0Y2ggdGhhdCBpbnB1dCBzaGFwZSBwcm92aWRlZC5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0U2hhcGUgQSBzaGFwZSAodHVwbGUgb2YgaW50ZWdlcnMpIG9yIGEgbGlzdCBvZiBzaGFwZSB0dXBsZXNcbiAgICogICAob25lIHBlciBvdXRwdXQgdGVuc29yIG9mIHRoZSBsYXllcikuIFNoYXBlIHR1cGxlcyBjYW4gaW5jbHVkZSBudWxsIGZvclxuICAgKiAgIGZyZWUgZGltZW5zaW9ucywgaW5zdGVhZCBvZiBhbiBpbnRlZ2VyLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgJ3N1YmhlYWRpbmcnOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIHJldHVybiBpbnB1dFNoYXBlO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGFuIG91dHB1dCBtYXNrIHRlbnNvci5cbiAgICpcbiAgICogQHBhcmFtIGlucHV0cyBUZW5zb3Igb3IgbGlzdCBvZiB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gbWFzayBUZW5zb3Igb3IgbGlzdCBvZiB0ZW5zb3JzLlxuICAgKlxuICAgKiBAcmV0dXJuIG51bGwgb3IgYSB0ZW5zb3IgKG9yIGxpc3Qgb2YgdGVuc29ycywgb25lIHBlciBvdXRwdXQgdGVuc29yIG9mIHRoZVxuICAgKiBsYXllcikuXG4gICAqL1xuICBjb21wdXRlTWFzayhpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgbWFzaz86IFRlbnNvcnxUZW5zb3JbXSk6IFRlbnNvclxuICAgICAgfFRlbnNvcltdIHtcbiAgICBpZiAoIXRoaXMuc3VwcG9ydHNNYXNraW5nKSB7XG4gICAgICBpZiAobWFzayAhPSBudWxsKSB7XG4gICAgICAgIGlmIChBcnJheS5pc0FycmF5KG1hc2spKSB7XG4gICAgICAgICAgbWFzay5mb3JFYWNoKG1hc2tFbGVtZW50ID0+IHtcbiAgICAgICAgICAgIGlmIChtYXNrRWxlbWVudCAhPSBudWxsKSB7XG4gICAgICAgICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoXG4gICAgICAgICAgICAgICAgICBgTGF5ZXIgJHt0aGlzLm5hbWV9IGRvZXMgbm90IHN1cHBvcnQgbWFza2luZywgYCArXG4gICAgICAgICAgICAgICAgICAnYnV0IHdhcyBwYXNzZWQgYW4gaW5wdXRNYXNrLicpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoXG4gICAgICAgICAgICAgIGBMYXllciAke3RoaXMubmFtZX0gZG9lcyBub3Qgc3VwcG9ydCBtYXNraW5nLCBgICtcbiAgICAgICAgICAgICAgJ2J1dCB3YXMgcGFzc2VkIGFuIGlucHV0TWFzay4nKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgLy8gbWFza2luZyBub3QgZXhwbGljaXRseSBzdXBwb3J0ZWQ6IHJldHVybiBudWxsIGFzIG1hc2tcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICAvLyBpZiBtYXNraW5nIGlzIGV4cGxpY3RseSBzdXBwb3J0ZWQsIGJ5IGRlZmF1bHRcbiAgICAvLyBjYXJyeSBvdmVyIHRoZSBpbnB1dCBtYXNrXG4gICAgcmV0dXJuIG1hc2s7XG4gIH1cblxuICBwcml2YXRlIHNldE1hc2tNZXRhZGF0YShpbnB1dHM6IFRlbnNvcnxUZW5zb3JbXSwgb3V0cHV0czogVGVuc29yfFRlbnNvcltdLFxuICAgICAgICAgICAgICAgICAgICAgICAgICBwcmV2aW91c01hc2s/OiBUZW5zb3J8VGVuc29yW10pOiB2b2lkIHtcbiAgICBpZiAoIXRoaXMuc3VwcG9ydHNNYXNraW5nKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3Qgb3V0cHV0TWFza3MgPSB0aGlzLmNvbXB1dGVNYXNrKGlucHV0cywgcHJldmlvdXNNYXNrKTtcbiAgICBpZiAob3V0cHV0cyBpbnN0YW5jZW9mIEFycmF5ICYmIG91dHB1dE1hc2tzIGluc3RhbmNlb2YgQXJyYXkpIHtcbiAgICAgIGlmIChvdXRwdXRzLmxlbmd0aCAhPT0gb3V0cHV0TWFza3MubGVuZ3RoKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgJHt0aGlzLm5hbWV9IG91dHB1dHMgJHtvdXRwdXRzLmxlbmd0aH0gdGVuc29ycyBgXG4gICAgICAgICAgKyBgYnV0ICR7b3V0cHV0TWFza3MubGVuZ3RofSBtYXNrcyBmb3IgdGhvc2UgdGVuc29yc2ApO1xuICAgICAgfVxuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBvdXRwdXRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIG91dHB1dHNbaV0ua2VyYXNNYXNrID0gb3V0cHV0TWFza3NbaV07XG4gICAgICB9XG4gICAgfSBlbHNlIGlmIChvdXRwdXRNYXNrcyBpbnN0YW5jZW9mIEFycmF5KSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYHt0aGlzLm5hbWV9IG91dHB1dHMgYSBzaW5nbGUgdGVuc29yIGBcbiAgICAgICAgKyBgYnV0ICR7b3V0cHV0TWFza3MubGVuZ3RofSBtYXNrc2ApO1xuICAgIH0gZWxzZSBpZiAob3V0cHV0cyBpbnN0YW5jZW9mIEFycmF5KSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYHt0aGlzLm5hbWV9IG91dHB1dHMgJHtvdXRwdXRzLmxlbmd0aH0gdGVuc29ycyBgXG4gICAgICAgICsgYGJ1dCBvbmx5IG9uZSBtYXNrYCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIG91dHB1dHMua2VyYXNNYXNrID0gb3V0cHV0TWFza3M7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIEludGVybmFsIG1ldGhvZCB0byBjcmVhdGUgYW4gaW5ib3VuZCBub2RlIGZvciB0aGUgbGF5ZXIuXG4gICAqXG4gICAqIEBwYXJhbSBpbnB1dFRlbnNvcnMgTGlzdCBvZiBpbnB1dCB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gb3V0cHV0VGVuc29ycyBMaXN0IG9mIG91dHB1dCB0ZW5zb3JzLlxuICAgKiBAcGFyYW0gaW5wdXRNYXNrcyBMaXN0IG9mIGlucHV0IG1hc2tzIChhIG1hc2sgY2FuIGJlIGEgdGVuc29yLCBvciBudWxsKS5cbiAgICogQHBhcmFtIG91dHB1dE1hc2tzIExpc3Qgb2Ygb3V0cHV0IG1hc2tzIChhIG1hc2sgY2FuIGJlIGEgdGVuc29yLCBvciBudWxsKS5cbiAgICogQHBhcmFtIGlucHV0U2hhcGVzIExpc3Qgb2YgaW5wdXQgc2hhcGUgdHVwbGVzLlxuICAgKiBAcGFyYW0gb3V0cHV0U2hhcGVzIExpc3Qgb2Ygb3V0cHV0IHNoYXBlIHR1cGxlcy5cbiAgICogQHBhcmFtIGt3YXJncyBEaWN0aW9uYXJ5IG9mIGtleXdvcmQgYXJndW1lbnRzIHRoYXQgd2VyZSBwYXNzZWQgdG8gdGhlXG4gICAqICAgYGNhbGxgIG1ldGhvZCBvZiB0aGUgbGF5ZXIgYXQgdGhlIGNhbGwgdGhhdCBjcmVhdGVkIHRoZSBub2RlLlxuICAgKi9cbiAgcHJpdmF0ZSBhZGRJbmJvdW5kTm9kZShcbiAgICAgIGlucHV0VGVuc29yczogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXSxcbiAgICAgIG91dHB1dFRlbnNvcnM6IFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW10sXG4gICAgICBpbnB1dE1hc2tzOiBUZW5zb3J8VGVuc29yW10sIG91dHB1dE1hc2tzOiBUZW5zb3J8VGVuc29yW10sXG4gICAgICBpbnB1dFNoYXBlczogU2hhcGV8U2hhcGVbXSwgb3V0cHV0U2hhcGVzOiBTaGFwZXxTaGFwZVtdLFxuICAgICAga3dhcmdzOiB7fSA9IG51bGwpOiB2b2lkIHtcbiAgICBjb25zdCBpbnB1dFRlbnNvckxpc3Q6IFN5bWJvbGljVGVuc29yW10gPVxuICAgICAgICBnZW5lcmljX3V0aWxzLnRvTGlzdChpbnB1dFRlbnNvcnMpO1xuICAgIG91dHB1dFRlbnNvcnMgPSBnZW5lcmljX3V0aWxzLnRvTGlzdChvdXRwdXRUZW5zb3JzKTtcbiAgICBpbnB1dE1hc2tzID0gZ2VuZXJpY191dGlscy50b0xpc3QoaW5wdXRNYXNrcyk7XG4gICAgb3V0cHV0TWFza3MgPSBnZW5lcmljX3V0aWxzLnRvTGlzdChvdXRwdXRNYXNrcyk7XG4gICAgaW5wdXRTaGFwZXMgPSB0eXBlc191dGlscy5ub3JtYWxpemVTaGFwZUxpc3QoaW5wdXRTaGFwZXMpO1xuICAgIG91dHB1dFNoYXBlcyA9IHR5cGVzX3V0aWxzLm5vcm1hbGl6ZVNoYXBlTGlzdChvdXRwdXRTaGFwZXMpO1xuXG4gICAgLy8gQ29sbGVjdCBpbnB1dCB0ZW5zb3IocykgY29vcmRpbmF0ZXMuXG4gICAgY29uc3QgaW5ib3VuZExheWVyczogTGF5ZXJbXSA9IFtdO1xuICAgIGNvbnN0IG5vZGVJbmRpY2VzOiBudW1iZXJbXSA9IFtdO1xuICAgIGNvbnN0IHRlbnNvckluZGljZXM6IG51bWJlcltdID0gW107XG4gICAgZm9yIChjb25zdCB4IG9mIGlucHV0VGVuc29yTGlzdCkge1xuICAgICAgLypcbiAgICAgICAqIFRPRE8obWljaGFlbHRlcnJ5KTogS2VyYXMgYWRkcyB0aGlzIHZhbHVlIHRvIHRlbnNvcnM7IGl0J3Mgbm90XG4gICAgICAgKiBjbGVhciB3aGV0aGVyIHdlJ2xsIHVzZSB0aGlzIG9yIG5vdC5cbiAgICAgICAqL1xuICAgICAgaW5ib3VuZExheWVycy5wdXNoKHguc291cmNlTGF5ZXIpO1xuICAgICAgbm9kZUluZGljZXMucHVzaCh4Lm5vZGVJbmRleCk7XG4gICAgICB0ZW5zb3JJbmRpY2VzLnB1c2goeC50ZW5zb3JJbmRleCk7XG4gICAgfVxuXG4gICAgLy8gQ3JlYXRlIG5vZGUsIGFkZCBpdCB0byBpbmJvdW5kIG5vZGVzLlxuICAgIC8vIChUaGlzIGNhbGwgaGFzIHNpZGUgZWZmZWN0cy4pXG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXVudXNlZC1leHByZXNzaW9uXG4gICAgbmV3IE5vZGUoXG4gICAgICAgIHtcbiAgICAgICAgICBvdXRib3VuZExheWVyOiB0aGlzLFxuICAgICAgICAgIGluYm91bmRMYXllcnMsXG4gICAgICAgICAgbm9kZUluZGljZXMsXG4gICAgICAgICAgdGVuc29ySW5kaWNlcyxcbiAgICAgICAgICBpbnB1dFRlbnNvcnM6IGlucHV0VGVuc29yTGlzdCxcbiAgICAgICAgICBvdXRwdXRUZW5zb3JzLFxuICAgICAgICAgIGlucHV0TWFza3MsXG4gICAgICAgICAgb3V0cHV0TWFza3MsXG4gICAgICAgICAgaW5wdXRTaGFwZXMsXG4gICAgICAgICAgb3V0cHV0U2hhcGVzXG4gICAgICAgIH0sXG4gICAgICAgIGt3YXJncyk7XG5cbiAgICAvLyBVcGRhdGUgdGVuc29yIGhpc3RvcnlcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHB1dFRlbnNvcnMubGVuZ3RoOyBpKyspIHtcbiAgICAgIC8vIFRPRE8obWljaGFlbHRlcnJ5OiBfdXNlc19sZWFybmluZ19waGFzZSBub3QgdHJhY2tlZC5cbiAgICAgIG91dHB1dFRlbnNvcnNbaV0uc291cmNlTGF5ZXIgPSB0aGlzO1xuICAgICAgb3V0cHV0VGVuc29yc1tpXS5ub2RlSW5kZXggPSB0aGlzLmluYm91bmROb2Rlcy5sZW5ndGggLSAxO1xuICAgICAgb3V0cHV0VGVuc29yc1tpXS50ZW5zb3JJbmRleCA9IGk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgdGhlIGNvbmZpZyBvZiB0aGUgbGF5ZXIuXG4gICAqXG4gICAqIEEgbGF5ZXIgY29uZmlnIGlzIGEgVFMgZGljdGlvbmFyeSAoc2VyaWFsaXphYmxlKVxuICAgKiBjb250YWluaW5nIHRoZSBjb25maWd1cmF0aW9uIG9mIGEgbGF5ZXIuXG4gICAqIFRoZSBzYW1lIGxheWVyIGNhbiBiZSByZWluc3RhbnRpYXRlZCBsYXRlclxuICAgKiAod2l0aG91dCBpdHMgdHJhaW5lZCB3ZWlnaHRzKSBmcm9tIHRoaXMgY29uZmlndXJhdGlvbi5cbiAgICpcbiAgICogVGhlIGNvbmZpZyBvZiBhIGxheWVyIGRvZXMgbm90IGluY2x1ZGUgY29ubmVjdGl2aXR5XG4gICAqIGluZm9ybWF0aW9uLCBub3IgdGhlIGxheWVyIGNsYXNzIG5hbWUuICBUaGVzZSBhcmUgaGFuZGxlZFxuICAgKiBieSAnQ29udGFpbmVyJyAob25lIGxheWVyIG9mIGFic3RyYWN0aW9uIGFib3ZlKS5cbiAgICpcbiAgICogUG9ydGluZyBOb3RlOiBUaGUgVFMgZGljdGlvbmFyeSBmb2xsb3dzIFRTIG5hbWluZyBzdGFuZGFyZHMgZm9yXG4gICAqIGtleXMsIGFuZCB1c2VzIHRmanMtbGF5ZXJzIHR5cGUtc2FmZSBFbnVtcy4gIFNlcmlhbGl6YXRpb24gbWV0aG9kc1xuICAgKiBzaG91bGQgdXNlIGEgaGVscGVyIGZ1bmN0aW9uIHRvIGNvbnZlcnQgdG8gdGhlIHB5dGhvbmljIHN0b3JhZ2VcbiAgICogc3RhbmRhcmQuIChzZWUgc2VyaWFsaXphdGlvbl91dGlscy5jb252ZXJ0VHNUb1B5dGhvbmljKVxuICAgKlxuICAgKiBAcmV0dXJucyBUUyBkaWN0aW9uYXJ5IG9mIGNvbmZpZ3VyYXRpb24uXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCAnc3ViaGVhZGluZyc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZzpcbiAgICAgICAgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge25hbWU6IHRoaXMubmFtZSwgdHJhaW5hYmxlOiB0aGlzLnRyYWluYWJsZX07XG4gICAgaWYgKHRoaXMuYmF0Y2hJbnB1dFNoYXBlICE9IG51bGwpIHtcbiAgICAgIGNvbmZpZ1snYmF0Y2hJbnB1dFNoYXBlJ10gPSB0aGlzLmJhdGNoSW5wdXRTaGFwZTtcbiAgICB9XG4gICAgaWYgKHRoaXMuZHR5cGUgIT0gbnVsbCkge1xuICAgICAgY29uZmlnWydkdHlwZSddID0gdGhpcy5kdHlwZTtcbiAgICB9XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxuXG4gIC8qKlxuICAgKiBEaXNwb3NlIHRoZSB3ZWlnaHQgdmFyaWFibGVzIHRoYXQgdGhpcyBMYXllciBpbnN0YW5jZSBob2xkcy5cbiAgICpcbiAgICogQHJldHVybnMge251bWJlcn0gTnVtYmVyIG9mIGRpc3Bvc2VkIHZhcmlhYmxlcy5cbiAgICovXG4gIHByb3RlY3RlZCBkaXNwb3NlV2VpZ2h0cygpOiBudW1iZXIge1xuICAgIHRoaXMud2VpZ2h0cy5mb3JFYWNoKHdlaWdodCA9PiB3ZWlnaHQuZGlzcG9zZSgpKTtcbiAgICByZXR1cm4gdGhpcy53ZWlnaHRzLmxlbmd0aDtcbiAgfVxuXG4gIHByb3RlY3RlZCBhc3NlcnROb3REaXNwb3NlZCgpIHtcbiAgICBpZiAodGhpcy5fcmVmQ291bnQgPT09IDApIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgTGF5ZXIgJyR7dGhpcy5uYW1lfScgaXMgYWxyZWFkeSBkaXNwb3NlZC5gKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogQXR0ZW1wdCB0byBkaXNwb3NlIGxheWVyJ3Mgd2VpZ2h0cy5cbiAgICpcbiAgICogVGhpcyBtZXRob2QgZGVjcmVhc2VzIHRoZSByZWZlcmVuY2UgY291bnQgb2YgdGhlIExheWVyIG9iamVjdCBieSAxLlxuICAgKlxuICAgKiBBIExheWVyIGlzIHJlZmVyZW5jZS1jb3VudGVkLiBJdHMgcmVmZXJlbmNlIGNvdW50IGlzIGluY3JlbWVudGVkIGJ5IDFcbiAgICogdGhlIGZpcnN0IGl0ZW0gaXRzIGBhcHBseSgpYCBtZXRob2QgaXMgY2FsbGVkIGFuZCB3aGVuIGl0IGJlY29tZXMgYSBwYXJ0XG4gICAqIG9mIGEgbmV3IGBOb2RlYCAodGhyb3VnaCBjYWxsaW5nIHRoZSBgYXBwbHkoKWAgbWV0aG9kIG9uIGFcbiAgICogYHRmLlN5bWJvbGljVGVuc29yYCkuXG4gICAqXG4gICAqIElmIHRoZSByZWZlcmVuY2UgY291bnQgb2YgYSBMYXllciBiZWNvbWVzIDAsIGFsbCB0aGUgd2VpZ2h0cyB3aWxsIGJlXG4gICAqIGRpc3Bvc2VkIGFuZCB0aGUgdW5kZXJseWluZyBtZW1vcnkgKGUuZy4sIHRoZSB0ZXh0dXJlcyBhbGxvY2F0ZWQgaW4gV2ViR0wpXG4gICAqIHdpbGwgYmUgZnJlZWQuXG4gICAqXG4gICAqIE5vdGU6IElmIHRoZSByZWZlcmVuY2UgY291bnQgaXMgZ3JlYXRlciB0aGFuIDAgYWZ0ZXIgdGhlIGRlY3JlbWVudCwgdGhlXG4gICAqIHdlaWdodHMgb2YgdGhlIExheWVyIHdpbGwgKm5vdCogYmUgZGlzcG9zZWQuXG4gICAqXG4gICAqIEFmdGVyIGEgTGF5ZXIgaXMgZGlzcG9zZWQsIGl0IGNhbm5vdCBiZSB1c2VkIGluIGNhbGxzIHN1Y2ggYXMgYGFwcGx5KClgLFxuICAgKiBgZ2V0V2VpZ2h0cygpYCBvciBgc2V0V2VpZ2h0cygpYCBhbnltb3JlLlxuICAgKlxuICAgKiBAcmV0dXJucyBBIERpc3Bvc2VSZXN1bHQgT2JqZWN0IHdpdGggdGhlIGZvbGxvd2luZyBmaWVsZHM6XG4gICAqICAgLSByZWZDb3VudEFmdGVyRGlzcG9zZTogVGhlIHJlZmVyZW5jZSBjb3VudCBvZiB0aGUgQ29udGFpbmVyIGFmdGVyIHRoaXNcbiAgICogICAgIGBkaXNwb3NlKClgIGNhbGwuXG4gICAqICAgLSBudW1EaXNwb3NlZFZhcmlhYmxlczogTnVtYmVyIG9mIGB0Zi5WYXJpYWJsZWBzIChpLmUuLCB3ZWlnaHRzKSBkaXNwb3NlZFxuICAgKiAgICAgZHVyaW5nIHRoaXMgYGRpc3Bvc2UoKWAgY2FsbC5cbiAgICogQHRocm93cyB7RXJyb3J9IElmIHRoZSBsYXllciBpcyBub3QgYnVpbHQgeWV0LCBvciBpZiB0aGUgbGF5ZXIgaGFzIGFscmVhZHlcbiAgICogICBiZWVuIGRpc3Bvc2VkLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgJ3N1YmhlYWRpbmcnOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBkaXNwb3NlKCk6IERpc3Bvc2VSZXN1bHQge1xuICAgIGlmICghdGhpcy5idWlsdCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBDYW5ub3QgZGlzcG9zZSBMYXllciAke3RoaXMubmFtZX0gYmVjYXVzZSBpdCBoYXMgbm90IGJlZW4gYCArXG4gICAgICAgICAgYGJ1aWx0IHlldC5gKTtcbiAgICB9XG5cbiAgICBpZiAodGhpcy5fcmVmQ291bnQgPT09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgQ2Fubm90IGRpc3Bvc2UgTGF5ZXIgJHt0aGlzLm5hbWV9IGJlY2F1c2UgaXQgaGFzIG5vdCBiZWVuIHVzZWQgYCArXG4gICAgICAgICAgYHlldC5gKTtcbiAgICB9XG5cbiAgICB0aGlzLmFzc2VydE5vdERpc3Bvc2VkKCk7XG5cbiAgICBsZXQgbnVtRGlzcG9zZWRWYXJpYWJsZXMgPSAwO1xuICAgIGlmICgtLXRoaXMuX3JlZkNvdW50ID09PSAwKSB7XG4gICAgICBudW1EaXNwb3NlZFZhcmlhYmxlcyA9IHRoaXMuZGlzcG9zZVdlaWdodHMoKTtcbiAgICB9XG5cbiAgICByZXR1cm4ge3JlZkNvdW50QWZ0ZXJEaXNwb3NlOiB0aGlzLl9yZWZDb3VudCwgbnVtRGlzcG9zZWRWYXJpYWJsZXN9O1xuICB9XG59XG5cbi8qKlxuICogQ29sbGVjdHMgdGhlIGlucHV0IHNoYXBlKHMpIG9mIGEgbGlzdCBvZiBgdGYuVGVuc29yYHMgb3JcbiAqIGB0Zi5TeW1ib2xpY1RlbnNvcmBzLlxuICpcbiAqIFRPRE8obWljaGFlbHRlcnJ5KTogVXBkYXRlIFB5S2VyYXMgZG9jcyAoYmFja3BvcnQpLlxuICpcbiAqIEBwYXJhbSBpbnB1dFRlbnNvcnMgTGlzdCBvZiBpbnB1dCB0ZW5zb3JzIChvciBzaW5nbGUgaW5wdXQgdGVuc29yKS5cbiAqXG4gKiBAcmV0dXJuIExpc3Qgb2Ygc2hhcGUgdHVwbGVzIChvciBzaW5nbGUgdHVwbGUpLCBvbmUgdHVwbGUgcGVyIGlucHV0LlxuICovXG5mdW5jdGlvbiBjb2xsZWN0SW5wdXRTaGFwZShpbnB1dFRlbnNvcnM6IFN5bWJvbGljVGVuc29yfFN5bWJvbGljVGVuc29yW118VGVuc29yfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgVGVuc29yW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgaW5wdXRUZW5zb3JzID1cbiAgICAgIGdlbmVyaWNfdXRpbHMudG9MaXN0KGlucHV0VGVuc29ycykgYXMgU3ltYm9saWNUZW5zb3JbXSB8IFRlbnNvcltdO1xuICBjb25zdCBzaGFwZXM6IFNoYXBlW10gPSBbXTtcbiAgZm9yIChjb25zdCB4IG9mIGlucHV0VGVuc29ycykge1xuICAgIHNoYXBlcy5wdXNoKHguc2hhcGUpO1xuICB9XG4gIHJldHVybiBnZW5lcmljX3V0aWxzLnNpbmdsZXRvbk9yQXJyYXkoc2hhcGVzKTtcbn1cblxuLyoqXG4gKiBHdWVzc2VzIG91dHB1dCBkdHlwZSBiYXNlZCBvbiBpbnB1dHMuXG4gKlxuICogQXQgcHJlc2VudCwganVzdCByZXR1cm5zICdmbG9hdDMyJyBmb3IgYW55IGlucHV0LlxuICpcbiAqIEBwYXJhbSBpbnB1dFRlbnNvcnMgTGlzdCBvZiBpbnB1dCB0ZW5zb3JzIChvciBzaW5nbGUgaW5wdXQgdGVuc29yKS5cbiAqXG4gKiBAcmV0dXJuIFRoZSBndWVzc2VkIERUeXBlLiBBdCBwcmVzZW50LCBhbHdheXMgcmV0dXJucyAnZmxvYXQzMicuXG4gKi9cbmZ1bmN0aW9uIGd1ZXNzT3V0cHV0RFR5cGUoaW5wdXRUZW5zb3JzOiBTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdfFRlbnNvcnxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgVGVuc29yW10pOiBEYXRhVHlwZSB7XG4gIHJldHVybiAnZmxvYXQzMic7XG59XG5cbi8qKlxuICogUmV0dXJucyB0aGUgbGlzdCBvZiBpbnB1dCB0ZW5zb3JzIG5lY2Vzc2FyeSB0byBjb21wdXRlIGB0ZW5zb3JgLlxuICpcbiAqIE91dHB1dCB3aWxsIGFsd2F5cyBiZSBhIGxpc3Qgb2YgdGVuc29ycyAocG90ZW50aWFsbHkgd2l0aCAxIGVsZW1lbnQpLlxuICpcbiAqIEBwYXJhbSB0ZW5zb3IgVGhlIHRlbnNvciB0byBzdGFydCBmcm9tLlxuICogQHBhcmFtIGxheWVyIE9yaWdpbiBsYXllciBvZiB0aGUgdGVuc29yLlxuICogQHBhcmFtIG5vZGVJbmRleCBPcmlnaW4gbm9kZSBpbmRleCBvZiB0aGUgdGVuc29yLlxuICpcbiAqIEByZXR1cm4gQXJyYXkgb2YgaW5wdXQgdGVuc29ycy5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldFNvdXJjZUlucHV0cyhcbiAgICB0ZW5zb3I6IFN5bWJvbGljVGVuc29yLCBsYXllcj86IExheWVyLFxuICAgIG5vZGVJbmRleD86IG51bWJlcik6IFN5bWJvbGljVGVuc29yW10ge1xuICBpZiAobGF5ZXIgPT0gbnVsbCB8fCAobm9kZUluZGV4ICE9IG51bGwgJiYgbm9kZUluZGV4ID4gMCkpIHtcbiAgICBsYXllciA9IHRlbnNvci5zb3VyY2VMYXllcjtcbiAgICBub2RlSW5kZXggPSB0ZW5zb3Iubm9kZUluZGV4O1xuICB9XG4gIGlmIChsYXllci5pbmJvdW5kTm9kZXMubGVuZ3RoID09PSAwKSB7XG4gICAgcmV0dXJuIFt0ZW5zb3JdO1xuICB9IGVsc2Uge1xuICAgIGNvbnN0IG5vZGUgPSBsYXllci5pbmJvdW5kTm9kZXNbbm9kZUluZGV4XTtcbiAgICBpZiAobm9kZS5pbmJvdW5kTGF5ZXJzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgcmV0dXJuIG5vZGUuaW5wdXRUZW5zb3JzO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBzb3VyY2VUZW5zb3JzOiBTeW1ib2xpY1RlbnNvcltdID0gW107XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5vZGUuaW5ib3VuZExheWVycy5sZW5ndGg7IGkrKykge1xuICAgICAgICBjb25zdCB4ID0gbm9kZS5pbnB1dFRlbnNvcnNbaV07XG4gICAgICAgIGNvbnN0IGxheWVyID0gbm9kZS5pbmJvdW5kTGF5ZXJzW2ldO1xuICAgICAgICBjb25zdCBub2RlSW5kZXggPSBub2RlLm5vZGVJbmRpY2VzW2ldO1xuICAgICAgICBjb25zdCBwcmV2aW91c1NvdXJjZXMgPSBnZXRTb3VyY2VJbnB1dHMoeCwgbGF5ZXIsIG5vZGVJbmRleCk7XG4gICAgICAgIC8vIEF2b2lkIGlucHV0IHJlZHVuZGFuY3kuXG4gICAgICAgIGZvciAoY29uc3QgeCBvZiBwcmV2aW91c1NvdXJjZXMpIHtcbiAgICAgICAgICBpZiAoc291cmNlVGVuc29ycy5pbmRleE9mKHgpID09PSAtMSkge1xuICAgICAgICAgICAgc291cmNlVGVuc29ycy5wdXNoKHgpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgcmV0dXJuIHNvdXJjZVRlbnNvcnM7XG4gICAgfVxuICB9XG59XG5cbnR5cGUgTWF5YmVTeW1ib2xpYyA9IFN5bWJvbGljVGVuc29yIHwgVGVuc29yO1xuXG5mdW5jdGlvbiBjaGVja0FsbFN5bWJvbGljKHRlbnNvcnM6IE1heWJlU3ltYm9saWMgfCBNYXliZVN5bWJvbGljW11cbiAgICAgICAgICAgICAgICAgICAgICAgICApOiB0ZW5zb3JzIGlzIFN5bWJvbGljVGVuc29yIHwgU3ltYm9saWNUZW5zb3JbXSB7XG4gIGxldCBhbGxBcmVTeW1ib2xpYyA9IHRydWU7XG4gIGZvciAoY29uc3QgdGVuc29yIG9mIGdlbmVyaWNfdXRpbHMudG9MaXN0KHRlbnNvcnMpKSB7XG4gICAgaWYgKCEodGVuc29yIGluc3RhbmNlb2YgU3ltYm9saWNUZW5zb3IpKSB7XG4gICAgICBhbGxBcmVTeW1ib2xpYyA9IGZhbHNlO1xuICAgICAgYnJlYWs7XG4gICAgfVxuICB9XG4gIHJldHVybiBhbGxBcmVTeW1ib2xpYztcbn1cblxuZnVuY3Rpb24gY2hlY2tOb25lU3ltYm9saWModGVuc29yczogTWF5YmVTeW1ib2xpYyB8IE1heWJlU3ltYm9saWNbXVxuICAgICAgICAgICAgICAgICAgICAgICAgICApOiB0ZW5zb3JzIGlzIFRlbnNvciB8IFRlbnNvcltdIHtcbiAgbGV0IG5vbmVBcmVTeW1ib2xpYyA9IHRydWU7XG4gIGZvciAoY29uc3QgdGVuc29yIG9mIGdlbmVyaWNfdXRpbHMudG9MaXN0KHRlbnNvcnMpKSB7XG4gICAgaWYgKHRlbnNvciBpbnN0YW5jZW9mIFN5bWJvbGljVGVuc29yKSB7XG4gICAgICBub25lQXJlU3ltYm9saWMgPSBmYWxzZTtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgfVxuICByZXR1cm4gbm9uZUFyZVN5bWJvbGljO1xufVxuIl19