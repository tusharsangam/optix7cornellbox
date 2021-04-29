#include <optix_device.h>
#include <device_functions.h>
#include "LaunchParams.h"
#include <curand_kernel.h>
#include <curand.h>


  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

    

  extern "C" __constant__ PRD;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_SHADOW, RAY_TYPE_COUNT };
  
  static __forceinline__ __device__  float rnd(curandState* seed) {
      return curand_uniform(seed);
  }
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  template<typename T>
  static __forceinline__ __device__
      void printvec(T& vectoprint)
  {
      printf("%f, %f, %f \n", vectoprint.x, vectoprint.y, vectoprint.z);
  }

  static __forceinline__ __device__
      void printmat(float* arr)
  {
      printf("%.2f, %.2f, %.2f %.2f\n", arr[0], arr[1], arr[2]);
  }

 


  

  static __forceinline__ __device__ int sign(float x) {
      return (x < 0.0) ? -1 : 1;
  }
  static __forceinline__ __device__
  gdt::vec3f& getDominantDirection(gdt::vec3f& point) {
      gdt::vec3f abspoint = abs(point);
      if (abspoint.x >= abspoint.y && abspoint.x >= abspoint.z)    return gdt::vec3f{ sign(point.x), 0, 0 };
      if (abspoint.y >= abspoint.x && abspoint.y >= abspoint.z)    return gdt::vec3f{ 0, sign(point.y), 0 };
      if (abspoint.z >= abspoint.x && abspoint.z >= abspoint.y)    return gdt::vec3f{ 0, 0, sign(point.z) };
  }
  
  static __forceinline__ __device__
      bool rayaabbintersection(OptixAabb& aabb, gdt::vec3f& rayorigin, gdt::vec3f& raydirection, float* tptr)
  {
      float tmin = optixGetRayTmin();
      float tmax = optixGetRayTmax();
      gdt::vec3f mins = { aabb.minX, aabb.minY, aabb.minZ };
      gdt::vec3f maxs = { aabb.maxX, aabb.maxY, aabb.maxZ };

      gdt::vec3f tMin = (mins - rayorigin)/ raydirection;
      gdt::vec3f tMax = (maxs - rayorigin) / raydirection;
      gdt::vec3f t1 = min(tMin, tMax);
      gdt::vec3f t2 = max(tMin, tMax);
      float tNear = gdt::max(gdt::max(gdt::max(t1.x, t1.y), t1.z), tmin);
      float tFar = gdt::min(gdt::min(gdt::min(t2.x, t2.y), t2.z), tmax);
      if (tNear < tFar) {
          *tptr = tNear;
          return true;
      }
      return false;
  }

  static __forceinline__ __device__
      bool isUnderShadow(vec3f& point, vec3f& lightposition, PRD& prd)
  {
      gdt::vec3f lightdirection = gdt::normalize(lightposition - point);
     
      prd.anyhit = false;
      uint32_t o0, o1; 
     
      packPointer(&prd, o0, o1);
      
      optixTrace(
          optixLaunchParams.traversable,
          point,
          lightdirection,
          0.01f,
          length(lightposition - point) - 0.01f,
          0.0f,                    // rayTime
          OptixVisibilityMask(1),
          OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
          RAY_TYPE_SHADOW,      // SBT offset
          RAY_TYPE_COUNT,          // SBT stride
          RAY_TYPE_SHADOW,      // missSBTIndex
          o0, o1);
      if (prd.anyhit) {
          prd.anyhit = false;
          return true;
      }
      return false;
  }
  
  /*static __forceinline__ __device__ void xorshift(unsigned int& value) {
      // Xorshift*32
      // Based on George Marsaglia's work: http://www.jstatsoft.org/v08/i14/paper
      value ^= value << 13;
      value ^= value >> 17;
      value ^= value << 5;
      
  }

  
  static __forceinline__ __device__ float fract(float x) {
      return x - floor(x);
  }

  static __forceinline__ __device__ float nextFloat(unsigned int& seed) {
      xorshift(seed);
      // FIXME: This should have been a seed mapped from MIN..MAX to 0..1 instead
      
      return fabsf(fract(float(seed) / 3141.592653));
  }


  static __forceinline__ __device__ float rnd(unsigned int& seed) {
      return nextFloat(seed);
  }*/

  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
 /* static __forceinline__ __device__ float random(float min, float high, unsigned int& seed) {
      return min + (high - min) * rnd(seed);
  }
  */

  static __forceinline__ __device__ float random(float min, float high, curandState* seed) {
      return min + (high - min) * rnd(seed);
  }

  static __forceinline__ __device__ vec3f sampleDirectionOnUnitHemisphereCosineSample(vec3f N, PRD& prd) {
      vec3f direction = vec3f(0);
      float r = sqrt(rnd(prd.seed));
      float phi = 2.0 * M_PI * rnd(prd.seed);


      direction.x = r * cos(phi);
      direction.y = r * sin(phi);
      direction.z = sqrt(1.0 - r * r);


      vec3f tempVector = vec3f(0);
      vec3f W = N;

      if (fabs(N.x) <= fabs(N.y) && fabs(N.x) <= fabs(N.z))
          tempVector.x = 1.0;
      else if (fabs(N.y) <= fabs(N.x) && fabs(N.y) <= fabs(N.z))
          tempVector.y = 1.0;
      else tempVector.z = 1.0;
      vec3f U = normalize(cross(tempVector, N));
      vec3f V = cross(N, U);

      direction = vec3f(dot(direction, U), dot(direction, V), dot(direction, W));

      return normalize(direction);

  }

  static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, vec3f& p)
  {
      // Uniformly sample disk.
      const float r = sqrtf(u1);
      const float phi = 2.0f * M_PI * u2;
      p.x = r * cosf(phi);
      p.y = r * sinf(phi);

      // Project up to hemisphere.
      p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
  }

  struct Onb
  {
      __forceinline__ __device__ Onb(const vec3f& normal)
      {
          m_normal = normal;

          if (fabs(m_normal.x) > fabs(m_normal.z))
          {
              m_binormal.x = -m_normal.y;
              m_binormal.y = m_normal.x;
              m_binormal.z = 0;
          }
          else
          {
              m_binormal.x = 0;
              m_binormal.y = -m_normal.z;
              m_binormal.z = m_normal.y;
          }

          m_binormal = normalize(m_binormal);
          m_tangent = cross(m_binormal, m_normal);
      }

      __forceinline__ __device__ void inverse_transform(vec3f& p) const
      {
          p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
         
      }

      vec3f m_tangent;
      vec3f m_binormal;
      vec3f m_normal;
  };

  extern "C" __global__ void __closesthit__radiance()
  {
   
    
    HitGroupSBT& mesh = *(HitGroupSBT*)optixGetSbtDataPointer();
   
    
    
    PRD& prd = *(PRD*)getPRD<PRD>();

    {
        
       //trace secondary ray
        //todo sample light point from light aabb
        gdt::vec3f& lightpoint = vec3f{
          random(optixLaunchParams.light.lightaabb.minX, optixLaunchParams.light.lightaabb.maxX, prd.seed),
          random(optixLaunchParams.light.lightaabb.minY, optixLaunchParams.light.lightaabb.maxY, prd.seed) ,
          random(optixLaunchParams.light.lightaabb.minZ, optixLaunchParams.light.lightaabb.maxZ, prd.seed) ,
        };
        
        vec3f hitpoint{
            int_as_float(optixGetAttribute_0()),
            int_as_float(optixGetAttribute_1()),
            int_as_float(optixGetAttribute_2())
        };

        vec3f normal{
            int_as_float(optixGetAttribute_3()),
            int_as_float(optixGetAttribute_4()),
            int_as_float(optixGetAttribute_5())
        };
        
        float low = 0.0;
        float high = 1.0;
        if (mesh.islight) {
            prd.islighthit = true;
            prd.Ldirect = mesh.color;
            
        }
        else {
            
            if (mesh.objectID == 2) {
                //if we hit ceil, its dark by default, TODO add a height hack
                prd.Ldirect = vec3f{0.0};
            }
            else if (isUnderShadow(hitpoint, lightpoint, prd)) {
                prd.Ldirect = vec3f{ 0 };
            }
            else {
                vec3f& lightnormal = optixLaunchParams.light.normal;
                gdt::vec3f& lightdirection = gdt::normalize(lightpoint - hitpoint);
                float costheta_1 = clamp(dot(normal, lightdirection), low, high);
                float costheta_2 = clamp(dot(lightnormal, -lightdirection), low, high);
                float lengt = length(lightpoint - hitpoint);
                float costheta = clamp(costheta_1 * costheta_2, low, high);
                prd.Ldirect = optixLaunchParams.light.lightarea * costheta * mesh.color / (M_PI * lengt * lengt);
                //prd.Ldirect = costheta_1 * mesh.color;
            }
        
        
        }
        
            
        
        
        
        const float z1 = rnd(prd.seed);
        const float z2 = rnd(prd.seed);
        vec3f w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
           
        Onb onb(normal);
        onb.inverse_transform(w_in);

        //w_in = sampleDirectionOnUnitHemisphereCosineSample(normal, prd);
           
        prd.sample_direction = w_in;
        prd.hitpoint = hitpoint;
        
        prd.materialcolor_hit = mesh.color;
        
        
        
       
    }
   
    
    

  }
  
  extern "C" __global__ void __closesthit__occlusion()
  { 
      PRD& occluded = *(PRD*)getPRD<PRD>();
      occluded.anyhit = true;
  }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    PRD &prd = *(PRD*)getPRD<PRD>();
    // set to constant white as background color
    prd.bg_hit = true;
    prd.Ldirect = vec3f{1.0};
  }

  


  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
   
    
    PRD pixelColorPRD;
    // normalized screen plane position, in [0,1]^2
    const gdt::vec2f screen(gdt::vec2f(ix+.5f,iy+.5f)/ gdt::vec2f(optixLaunchParams.frame.size));
    //unsigned int seed = tea<4> (int(screen.x) + int(screen.y) * int(optixLaunchParams.frame.size.x), optixLaunchParams.frame_index);
    //unsigned int seed = tea<4>(ix * optixLaunchParams.frame.size.x + ix, optixLaunchParams.frame_index);
    unsigned long long seed = ix + optixLaunchParams.frame.size.x * iy;
    
    curand_init(seed, 0, 0, &optixLaunchParams.dev_random[seed]);
   
    pixelColorPRD.seed = &optixLaunchParams.dev_random[seed];
    // generate ray direction
    vec3f result = {0.0};
    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);
    for (size_t i = 0; i < optixLaunchParams.number_of_samples; i++)
    {       
            
            //jitter direction
            vec2f jitter{ rnd(pixelColorPRD.seed) , rnd(pixelColorPRD.seed) };
            const vec2f jitterpixel = 2.0f * vec2f(
                (static_cast<float>(ix) + jitter.x) / static_cast<float>(optixLaunchParams.frame.size.x),
                (static_cast<float>(iy) + jitter.y) / static_cast<float>(optixLaunchParams.frame.size.y)
            ) - 1.0f;
            //vec2f& pixel = vec2f{ screen + vec2f{ (float(j) + ) / float(optixLaunchParams.number_of_samples), (float(i) + rnd(pixelColorPRD.seed)) / float(optixLaunchParams.number_of_samples) } };
            //gdt::vec3f& rayDir = gdt::normalize(camera.direction + (jitterpixel.x ) * camera.horizontal + (jitterpixel.y) * camera.vertical);
            vec3f ray_origin = optixLaunchParams.camera.position;
            vec3f ray_direction = normalize(jitterpixel.x * optixLaunchParams.camera.horizontal + jitterpixel.y * optixLaunchParams.camera.vertical + optixLaunchParams.camera.direction);
            vec3f factor{ 1.0 };

            pixelColorPRD.bg_hit = false;
            pixelColorPRD.islighthit = false;
            
            for (int j=0; j<optixLaunchParams.depth; j++)
            {
                //printf("%d \n", j);
                optixTrace(optixLaunchParams.traversable,
                    ray_origin,
                    ray_direction,
                    0.01f,    // tmin
                    1e16f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE, OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    SURFACE_RAY_TYPE,             // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride, RAY_TYPE_COUNT
                    SURFACE_RAY_TYPE,             // missSBTIndex 
                    u0, u1);

                if (pixelColorPRD.bg_hit) break;
                
                result += factor*pixelColorPRD.Ldirect;
                factor *=  pixelColorPRD.materialcolor_hit;

               
                if (pixelColorPRD.islighthit) break;
               
                
                //update ray origin & direction
                ray_origin = pixelColorPRD.hitpoint; //+ 1e-5f*pixelColorPRD.sample_direction; //+ 1e-5f* pixelColorPRD.sample_direction;
                ray_direction = pixelColorPRD.sample_direction;
                
               // break;
            }
           
        
    }
    
    result = result / optixLaunchParams.number_of_samples; 
    //gdt::vec3f rayDir = gdt::normalize(camera.direction + (pixel.x -0.5 ) * camera.horizontal + (pixel.y -0.5) * camera.vertical);
    
    
        // and write to frame buffer ...
    /*const int r = int(255.99f * result.x);
    const int g = int(255.99f * result.y);
    const int b = int(255.99f * result.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);*/
    //vec4f rgba(result, 1.f);
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = make_float4(result.x, result.y, result.z, 1.f);
  }

  extern "C" __global__ void __intersection__occlusion_aabb()
  {

      PRD& prd = *(PRD*)getPRD<PRD>();
      prd.anyhit = true;
      uint32_t u0, u1;
      packPointer(&prd, u0, u1);
     
  }
  
  #define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

  extern "C" __global__ void __intersection__aabb()
  {

      

      HitGroupSBT& mesh = *(HitGroupSBT*)optixGetSbtDataPointer();

      OptixAabb& opaabb = mesh.aabb;
      float t = 0.0;
     

      gdt::vec3f& origin = gdt::vec3f{ optixGetObjectRayOrigin() };
      gdt::vec3f& direction = gdt::vec3f{ optixGetObjectRayDirection() };


     bool intersects = rayaabbintersection(opaabb, origin, direction, &t);
      if (intersects) {
          vec3f& point = origin + t * direction;
          vec3f& normal = getDominantDirection(point);
          point = vec3f{ optixTransformPointFromObjectToWorldSpace(point)  };
          normal = normalize(vec3f{ optixTransformNormalFromObjectToWorldSpace(normal) });
         
          optixReportIntersection(t, 0, float3_as_ints(point), float3_as_ints(normal));

      }
      


     
  }

  

  

