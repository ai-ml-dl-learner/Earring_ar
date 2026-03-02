// Import ONNX Runtime Web directly from the CDN
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');

// CRITICAL FOR DEPLOYMENT: Tell the worker exactly where to find the WebAssembly files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

let session;
const IMAGE_SIZE = 256; 

// Load the 256px model you just exported
ort.InferenceSession.create('./ear_fpn_256.onnx', { 
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
})
.then(s => {
    session = s;
    postMessage({ type: 'ready' }); 
})
.catch(err => {
    postMessage({ type: 'error', message: err.message });
});

onmessage = async (e) => {
    if (!session) return;

    if (e.data.type === 'predict') {
        try {
            const float32Data = e.data.tensorData; 
            const tensor = new ort.Tensor('float32', float32Data, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
            const results = await session.run({ input: tensor });
            
            // Send the 2-channel array back to the main UI
            postMessage({ 
                type: 'result', 
                outputData: results.out.data 
            });
        } catch (err) {
            postMessage({ type: 'error', message: err.message });
        }
    }
};