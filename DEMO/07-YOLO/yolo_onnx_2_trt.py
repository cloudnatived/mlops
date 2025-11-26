import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    # Use new API for TensorRT 8.x+
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Create builder config
    config = builder.create_builder_config()
    # Set memory pool limit instead of max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Get input tensor
    input_tensor = network.get_input(0)
    print(f"Input shape: {input_tensor.shape}")
    
    # Check if model has dynamic dimensions
    is_dynamic = any(dim == -1 for dim in input_tensor.shape)
    
    if is_dynamic:
        # Define optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # For YOLO models, typical input size is 640x640
        # Adjust these values based on your model requirements
        batch_size = 1
        input_height = 640
        input_width = 640
        
        # Set min, opt, and max shapes
        min_shape = (batch_size, 3, 320, 320)      # minimum input shape
        opt_shape = (batch_size, 3, 640, 640)      # optimal input shape  
        max_shape = (batch_size, 3, 1280, 1280)    # maximum input shape
        
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    
    # Build engine using new API
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print('ERROR: Failed to build engine.')
        return None
    
    # Save engine
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f'Successfully built TensorRT engine: {engine_file_path}')
    return serialized_engine

if __name__ == '__main__':
    build_engine('/Data/MODEL/YOLO/yolov13x.onnx', '/Data/MODEL/YOLO/yolov13x.trt')
