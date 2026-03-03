importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
ort.env.wasm.simd = true;

// CAREFUL: Leave at least 1 core for MediaPipe's WebGL context.
// If your device has 8 cores, this uses 7 for ONNX, 1 for MediaPipe/UI.
const maxThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.numThreads = Math.max(1, maxThreads - 1); 

let session;
const IMAGE_SIZE = 256;

async function init() {
    try {
        session = await ort.InferenceSession.create(
            './ear_unet_256_int8.onnx',
            {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            }
        );
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
}

init();

onmessage = async (e) => {
    if (!session) return;

    if (e.data.type === 'predict') {
        try {
            const inputTensor = new ort.Tensor(
                'float32',   
                e.data.tensorData,
                [1, 3, IMAGE_SIZE, IMAGE_SIZE]
            );

            const results = await session.run({ input: inputTensor });

            postMessage({
                type: 'result',
                outputData: results.out.data
            });

        } catch (err) {
            postMessage({ type: 'error', message: err.message });
        }
    }
};
