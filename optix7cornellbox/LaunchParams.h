#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include <curand_kernel.h>
#include <curand.h>


  using namespace gdt;
  
  struct LaunchParams
  {
    struct {
      float4 *colorBuffer; //uint32_t
      gdt::vec2i     size;
    } frame;
    
    struct {
      gdt::vec3f position;
      gdt::vec3f direction;
      gdt::vec3f horizontal;
      gdt::vec3f vertical;
      float fov;
    } camera;

    struct {
        OptixAabb lightaabb{ -0.3, 1.0, -0.3, 0.3, 1.0, 0.3 };
        gdt::vec3f normal{0, -1, 0};
        float lightarea{8.0};
        gdt::vec3f midpoint{0., 0.99, 0.};
    } light;

    int number_of_samples = 16;
    int depth = 5;
    
    curandState* dev_random;
    OptixTraversableHandle traversable;
  };


  struct PRD {
      
      gdt::vec3f color;
      curandState* seed;
      vec3f hitpoint;
      vec3f sample_direction;
      vec3f materialcolor_hit;
      vec3f Ldirect;
      bool anyhit{false};
      bool bg_hit{ false };
      
      bool islighthit;
      
  };

  

  struct Instance {
      gdt::vec3f materialcolor;
      float transformation[12];
      bool islight = false;
  };

  struct HitGroupSBT {
      
      int objectID;
      OptixAabb aabb;
      gdt::vec3f color;
      bool islight = false;

  };

 
