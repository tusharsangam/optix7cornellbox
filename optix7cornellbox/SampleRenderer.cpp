#include "SampleRenderer.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>


  extern "C" char embedded_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    HitGroupSBT data;
  };

  enum InstanceType
  {
      COUNT = 8
  };
  
  enum RAY{ RAY_TYPE_COUNT = 2};
  
  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  

  SampleRenderer::SampleRenderer(const OptixAabb& base_model, std::vector<Instance>& instances)
  {
      initOptix();

      std::cout << "#RT: creating optix context ..." << std::endl;
      createContext();

      std::cout << "#RT: setting up module ..." << std::endl;
      createModule();

      std::cout << "#RT: creating raygen programs ..." << std::endl;
      createRaygenPrograms();
      std::cout << "#RT: creating miss programs ..." << std::endl;
      createMissPrograms();
      std::cout << "#RT: creating hitgroup programs ..." << std::endl;
      createHitgroupPrograms();

      

      //launchParams.traversable = buildAccel(aabb);
      
      //copydatabetweenfloatarrays(, &(model.transformation[0]) ,12);
     

      launchParams.traversable = buildISAccel(instances, buildAccel(base_model) );

      std::cout << "#RT: setting up optix pipeline ..." << std::endl;
      createPipeline();

      std::cout << "#RT: building SBT ..." << std::endl;
      buildSBT(base_model, instances);


      launchParamsBuffer.alloc(sizeof(launchParams));
      std::cout << "#RT: context, module, pipeline, etc, all set up ..." << std::endl;



      std::cout << GDT_TERMINAL_GREEN;
      std::cout << "#RT: Optix 7 Sample fully set up" << std::endl;
      std::cout << GDT_TERMINAL_DEFAULT;
  }
  
  OptixTraversableHandle SampleRenderer::buildAccel( const OptixAabb& model)
  {     
      // upload the model to the device: the builder
      
      aabbbuffer.alloc_and_upload(model);
      CUdeviceptr d_aabbbuffer = aabbbuffer.d_pointer();

      OptixTraversableHandle asHandle{ 0 };
      
      
      OptixBuildInput aabbInput = {};
      
      aabbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
     
      aabbInput.customPrimitiveArray.aabbBuffers = &d_aabbbuffer;
      aabbInput.customPrimitiveArray.numPrimitives = 1;
      //aabbInput.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
      uint32_t aabbInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
      aabbInput.customPrimitiveArray.flags = aabbInputFlags;
      aabbInput.customPrimitiveArray.numSbtRecords = 1;
     
      // ==================================================================
    // GAS setup
    // ==================================================================

      OptixAccelBuildOptions accelOptions = {};
      accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      //accelOptions.motionOptions.numKeys = 1;
      accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

      OptixAccelBufferSizes blasBufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage
      (optixContext,
          &accelOptions,
          &aabbInput,
          1,  // num_build_inputs
          &blasBufferSizes
      ));

      // ==================================================================
      // prepare compaction
      // ==================================================================
      CUdeviceptr compactedSizeBuffer;
      size_t      compactedSizeOffset = ((blasBufferSizes.outputSizeInBytes + OPTIX_AABB_BUFFER_BYTE_ALIGNMENT - 1) / OPTIX_AABB_BUFFER_BYTE_ALIGNMENT) * OPTIX_AABB_BUFFER_BYTE_ALIGNMENT; //roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
      CUDA_CHECK(Malloc(
          reinterpret_cast<void**>(&compactedSizeBuffer),
          compactedSizeOffset + 8
      ));
      //CUDABuffer compactedSizeBuffer;
     // compactedSizeBuffer.alloc(sizeof(uint64_t));

      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = (CUdeviceptr)((char*)compactedSizeBuffer + compactedSizeOffset);

      // ==================================================================
      // execute build (main stage)
      // ==================================================================

      CUDABuffer tempBuffer;
      tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

      CUDABuffer outputBuffer;
      outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

      OPTIX_CHECK(optixAccelBuild(optixContext,
          /* stream */0,
          &accelOptions,
          &aabbInput,
          1,
          tempBuffer.d_pointer(),
          tempBuffer.sizeInBytes,

          compactedSizeBuffer,
          blasBufferSizes.outputSizeInBytes,

          &asHandle,

          &emitDesc, 1
      ));
      CUDA_SYNC_CHECK();

      // ==================================================================
      // perform compaction
      // ==================================================================
      uint64_t compactedSize;
      CUDA_CHECK(Memcpy(&compactedSize, (void*)emitDesc.result, sizeof(size_t), cudaMemcpyDeviceToHost));
      //compactedSizeBuffer.download(&compactedSize, 1);

      asBuffer.alloc(compactedSize);
      OPTIX_CHECK(optixAccelCompact(optixContext,
          /*stream:*/0,
          asHandle,
          asBuffer.d_pointer(),
          asBuffer.sizeInBytes,
          &asHandle));
      CUDA_SYNC_CHECK();

      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
      tempBuffer.free();
      CUDA_CHECK(Free((void *)compactedSizeBuffer));
        
      

      return asHandle;
      //return OptixTraversableHandle();
  }

  

  OptixTraversableHandle SampleRenderer::buildISAccel(std::vector<Instance>& instances, OptixTraversableHandle gas_handle)
  {
      // upload the model to the device: the builder
      //std::vector<float> transformation(std::begin(model.transformation), std::end(model.transformation));
      //transformationBuffer.alloc_and_upload(transformation);
      //CUdeviceptr d_transformationbuffer = transformationBuffer.d_pointer();
      OptixTraversableHandle IasHandle{ 0 };

     

      OptixInstance optix_instances[InstanceType::COUNT];
      const size_t instance_size_in_bytes = sizeof(OptixInstance) * InstanceType::COUNT;
      memset(optix_instances, 0, instance_size_in_bytes);
      for (size_t i = 0; i < InstanceType::COUNT; i++)
      { 
          optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
          optix_instances[i].instanceId = i;
          optix_instances[i].sbtOffset = i*RAY::RAY_TYPE_COUNT;// i* sizeof(OptixInstance);
          optix_instances[i].visibilityMask = 1;
          optix_instances[i].traversableHandle = gas_handle;
          memcpy(optix_instances[i].transform, instances[i].transformation, sizeof(float) * 12);

      }
     
      //printf("%f, %f, %f, %f", model.transformation[0], model.transformation[1], model.transformation[2], model.transformation[3]);
      //printf("%f, %f, %f, %f", optix_instances[0].transform[0], optix_instances[0].transform[1], optix_instances[0].transform[2], optix_instances[0].transform[3]);
      CUdeviceptr  d_instances;
      CUDA_CHECK(Malloc(reinterpret_cast<void**>(&d_instances), instance_size_in_bytes));
      CUDA_CHECK(Memcpy(
          reinterpret_cast<void*>(d_instances),
          optix_instances,
          instance_size_in_bytes,
          cudaMemcpyHostToDevice
      ));

      OptixBuildInput instance_input = {};
      instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      instance_input.instanceArray.instances = d_instances;
      instance_input.instanceArray.numInstances = InstanceType::COUNT;
 
      

      OptixAccelBuildOptions accel_options = {};
      accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

      OptixAccelBufferSizes ias_buffer_sizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(
          optixContext,
          &accel_options,
          &instance_input,
          1, // num build inputs
          &ias_buffer_sizes
      ));

      CUdeviceptr d_temp_buffer, d_ias_output_buffer = 0;
      CUDA_CHECK(Malloc(
          reinterpret_cast<void**>(&d_temp_buffer),
          ias_buffer_sizes.tempSizeInBytes
      ));

      CUDA_CHECK(Malloc(
          reinterpret_cast<void**>(&d_ias_output_buffer),
          ias_buffer_sizes.outputSizeInBytes
      ));

      OPTIX_CHECK(optixAccelBuild(
          optixContext,
          0,                  // CUDA stream
          &accel_options,
          &instance_input,
          1,                  // num build inputs
          d_temp_buffer,
          ias_buffer_sizes.tempSizeInBytes,
          d_ias_output_buffer,
          ias_buffer_sizes.outputSizeInBytes,
          &IasHandle,
          nullptr,            // emitted property list
          0                   // num emitted properties
      ));

      CUDA_CHECK(Free(reinterpret_cast<void*>(d_temp_buffer)));
      CUDA_CHECK(Free(reinterpret_cast<void*>(d_instances)));
      
      return IasHandle;
    
  }

  
  
  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
  {
    std::cout << "#RT: initializing optix..." << std::endl;
      
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#RT: no CUDA capable devices found!");
    std::cout << "#RT: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#RT: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#RT: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }



  /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
  void SampleRenderer::createModule()
  {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;//OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 6;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
    pipelineLinkOptions.maxTraceDepth          = 5;
      
    const std::string ptxCode = embedded_ptx_code;
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
    if (sizeof_log > 1) PRINT(log);
  }
    


  /*! does all setup for the raygen program(s) we are going to use */
  void SampleRenderer::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;           
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    
  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(2);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;           
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
    // NULL miss program for occlusion rays
    OptixProgramGroupOptions pgOptions2 = {};
    OptixProgramGroupDesc pgDesc2 = {};
    pgDesc2.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc2.miss.module = nullptr;
    pgDesc2.miss.entryFunctionName = nullptr;

    // OptixProgramGroup occlusion rays;
    char log2[2048];
    sizeof_log = sizeof(log2);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc2,
        1,
        &pgOptions2,
        log2, &sizeof_log,
        &missPGs[1]
    ));
    if (sizeof_log > 1) PRINT(log2);
    
  }
    
  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(2);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;           
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    //pgDesc.hitgroup.moduleAH            = module;           
    //pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    pgDesc.hitgroup.moduleIS = module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__aabb";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
    OptixProgramGroupDesc pgDesc2 = {};
    pgDesc2.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc2.hitgroup.moduleCH = module;
    pgDesc2.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    pgDesc2.hitgroup.moduleIS = module;
    pgDesc2.hitgroup.entryFunctionNameIS = "__intersection__occlusion_aabb";
    char log2[2048];
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc2,
        1,
        &pgOptions,
        log2, &sizeof_log,
        &hitgroupPGs[1]
    ));
  }
    

  /*! assembles the full pipeline of all programs */
  void SampleRenderer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) PRINT(log);

    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(raygenPGs[0], &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(missPGs[0], &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(missPGs[1], &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(hitgroupPGs[0], &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(hitgroupPGs[1], &stack_sizes));
    unsigned int maxTraceDepth = pipelineLinkOptions.maxTraceDepth;
    unsigned int maxCCDepth = 0;
    unsigned int maxDCDepth = 0;
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        maxTraceDepth,
        maxCCDepth,
        maxDCDepth,
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &continuationStackSize
    ));
    // This is 2 since the largest depth is IAS->GAS
    unsigned int maxTraversalDepth = 2;

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        maxTraversalDepth
    ));
    

   
    //if (sizeof_log > 1) PRINT(log);
  }


  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT(const OptixAabb& model, std::vector<Instance>& instances)
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i< missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------

    // we don't actually have any objects in this example, but let's
    // create a dummy one so the SBT doesn't have any null pointers
    // (which the sanity checks in compilation would complain about)
    int numObjects = InstanceType::COUNT ;
    std::vector<HitgroupRecord> hitgroupRecords;
    hitgroupRecords.resize(RAY::RAY_TYPE_COUNT * InstanceType::COUNT);
    const size_t hitgroup_record_size = sizeof(HitgroupRecord);
    for (int i=0;i<numObjects;i++) {

        {
            const int sbt_idx = i * RAY::RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &hitgroupRecords[sbt_idx]));
            hitgroupRecords[sbt_idx].data.objectID = i;
            hitgroupRecords[sbt_idx].data.aabb = model;
            hitgroupRecords[sbt_idx].data.color = instances[i].materialcolor;
            hitgroupRecords[sbt_idx].data.islight = instances[i].islight;
        }
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            memset(&hitgroupRecords[sbt_idx], 0, hitgroup_record_size);
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[1], &hitgroupRecords[sbt_idx]));
        
        }


    }
    
    
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }


  
  /*! render one frame */
  void SampleRenderer::render()
  {
    // sanity check: make sure we launch only after first resize is
    // already done:
      if (launchParams.frame.size.x == 0) return;
       
    
    launchParamsBuffer.upload(&launchParams,1);
      
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));
    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 1;
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    denoiserParams.blendFactor = 1.f / (1.f);

    // -------------------------------------------------------
    OptixImage2D inputLayer;
    inputLayer.data = colorBuffer.d_pointer();
    /// Width of the image (in pixels)
    inputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = denoisedBuffer.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    outputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    if (denoiserOn) {
#if OPTIX_VERSION >= 70300
        OptixDenoiserGuideLayer denoiserGuideLayer = {};

        OptixDenoiserLayer denoiserLayer = {};
        denoiserLayer.input = inputLayer;
        denoiserLayer.output = outputLayer;

        //printf("Denoiser invoked\n");

        OPTIX_CHECK(optixDenoiserInvoke(denoiser,
            /*stream*/0,
            &denoiserParams,
            denoiserState.d_pointer(),
            denoiserState.sizeInBytes,
            &denoiserGuideLayer,
            &denoiserLayer, 1,
            /*inputOffsetX*/0,
            /*inputOffsetY*/0,
            denoiserScratch.d_pointer(),
            denoiserScratch.sizeInBytes));
#else
        OPTIX_CHECK(optixDenoiserInvoke(denoiser,
            /*stream*/0,
            &denoiserParams,
            denoiserState.d_pointer(),
            denoiserState.size(),
            &inputLayer, 1,
            /*inputOffsetX*/0,
            /*inputOffsetY*/0,
            &outputLayer,
            denoiserScratch.d_pointer(),
            denoiserScratch.size()));
#endif
    }
    else {
        cudaMemcpy((void*)outputLayer.data, (void*)inputLayer.data,
            outputLayer.width * outputLayer.height * sizeof(float4),
            cudaMemcpyDeviceToDevice);
    }

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
    //launchParams.frame_index += 1;
  }

  void build_uvw(vec3f& U, vec3f& V, vec3f& W, vec3f& up, float m_aspectRatio, float m_fovY) {
      //W = at - from; // Do not normalize W -- it implies focal length
      float wlen = length(W);
      U = normalize(cross(W, up));
      V = normalize(cross(U, W));

      float vlen = wlen * tanf(0.5f * m_fovY * M_PI / 180.0f);
      V *= vlen;
      float ulen = vlen * m_aspectRatio;
      U *= ulen;
  }
  /*! set camera to render with */
  void SampleRenderer::setCamera(const Camera &camera)
  {
    /*lastSetCamera = camera;
    launchParams.camera.position  = camera.from;
    launchParams.camera.direction = normalize(camera.from - camera.at);
    launchParams.camera.horizontal = normalize(cross(camera.up, launchParams.camera.direction));
    launchParams.camera.vertical = normalize(cross(launchParams.camera.direction, launchParams.camera.horizontal));
    launchParams.camera.fov = 20. * M_PI / 180.;
    */
    
    lastSetCamera = camera;
    launchParams.camera.position = camera.from;
    /*launchParams.camera.direction = normalize(camera.at - camera.from);
    const float cosFovy = 0.66f;
    const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
    launchParams.camera.horizontal
      = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                                           camera.up));
    launchParams.camera.vertical
      = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                  launchParams.camera.direction));
     */
    
    vec3f U, V, W = camera.at - camera.from;
    vec3f up = camera.up;
    build_uvw(U, V, W, up, (launchParams.frame.size.x / float(launchParams.frame.size.y)), camera.m_fovY);
    launchParams.camera.vertical = V;
    launchParams.camera.horizontal = U;
    launchParams.camera.direction = W;
  }
  
  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {
      if (denoiser) {
          OPTIX_CHECK(optixDenoiserDestroy(denoiser));
      };
      // ------------------------------------------------------------------
      // create the denoiser:
      OptixDenoiserOptions denoiserOptions = {};
    // if window minimized
#if OPTIX_VERSION >= 70300
      OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser));
#else
      denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
      // these only exist in 7.0, not 7.1
      denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

      OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
      OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));
#endif

      // .. then compute and allocate memory resources for the denoiser
      OptixDenoiserSizes denoiserReturnSizes;
      OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y,
          &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
      denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
      denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
          denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
      denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

      // ------------------------------------------------------------------
      // resize our cuda frame buffer
    denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));
    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x*newSize.y*sizeof(float4));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size  = newSize;
    launchParams.frame.colorBuffer = (float4*)colorBuffer.d_pointer();

    cudaMalloc((void**)&launchParams.dev_random, launchParams.frame.size.x * launchParams.frame.size.y * sizeof(curandState));

    OPTIX_CHECK(optixDenoiserSetup(denoiser, 0,
        newSize.x, newSize.y,
        denoiserState.d_pointer(),
        denoiserState.sizeInBytes,
        denoiserScratch.d_pointer(),
        denoiserScratch.sizeInBytes));

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);
  }

  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(vec4f h_pixels[])
  {
      colorBuffer.download(h_pixels,
                         launchParams.frame.size.x*launchParams.frame.size.y);
  }
  