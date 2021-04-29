// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>



  struct SampleWindow : public GLFCameraWindow
  {
      SampleWindow(const std::string& title,
                    const OptixAabb& model,
                  std::vector<Instance>& instances,
                 const Camera &camera,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale),
        sample(model, instances)
    {
    }

    virtual void render() override
    {
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }
    virtual void draw() override
    {
        sample.downloadPixels(pixels.data());
        if (fbTexture == 0)
            glGenTextures(1, &fbTexture);

        glBindTexture(GL_TEXTURE_2D, fbTexture);
        GLenum texFormat = GL_RGBA;
        GLenum texelType = GL_FLOAT;//GL_UNSIGNED_BYTE;
        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
            texelType, pixels.data());

        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glDisable(GL_DEPTH_TEST);

        glViewport(0, 0, fbSize.x, fbSize.y);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.f, 0.f);
            glVertex3f(0.f, 0.f, 0.f);

            glTexCoord2f(0.f, 1.f);
            glVertex3f(0.f, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 1.f);
            glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 0.f);
            glVertex3f((float)fbSize.x, 0.f, 0.f);
        }
        glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    virtual void key(int key, int mods)
    {
        if (key == 'D' || key == ' ' || key == 'd') {
            sample.denoiserOn = !sample.denoiserOn;
            std::cout << "denoising now " << (sample.denoiserOn ? "ON" : "OFF") << std::endl;
        }

        if (key == ',') {
            sample.launchParams.number_of_samples
                = std::max(1, sample.launchParams.number_of_samples - 1);
            std::cout << "num samples/pixel now "
                << sample.launchParams.number_of_samples << std::endl;
        }
        if (key == '.') {
            sample.launchParams.number_of_samples
                = std::max(1, sample.launchParams.number_of_samples + 1);
            std::cout << "num samples/pixel now "
                << sample.launchParams.number_of_samples << std::endl;
        }
        
        
    }

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<vec4f> pixels;
  };
  
  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      
     
      
      OptixAabb base_aabb{-0.5, -0.5, -0.5, 0.5, 0.5, 0.5};
      std::vector<Instance> instances;
      
      //red_wall - > left
      Instance ins;
      float transformation_red[12] = {0.00999, 0., 0., -1.,   0., 2.0, 0.0, 0.0,   0.0, 0.0, 2.0, 0.0  };
      memcpy(ins.transformation, transformation_red, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 166.0 / 255., 13. / 255., 13. / 255. };
      instances.push_back(ins);
      
      //green wall -> right
      ins = Instance{};
      float transformation_green[12] = { 0.00999, 0., 0., 1.,   0., 2.0, 0.0, 0.0,   0.0, 0.0, 2.0, 0.0 };
      memcpy(ins.transformation, transformation_green, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 31. / 255., 115. / 255., 38. / 255. };
      instances.push_back(ins);

      
      //ceil
      ins = Instance{};
      float transformation_ceil[12] = { 2., 0., 0., 0.,   0., 0.0099, 0.0, 1.0,   0.0, 0.0, 2.0, 0.0 };
      memcpy(ins.transformation, transformation_ceil, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 186. / 255., 186. / 255., 186. / 255. };
      instances.push_back(ins);

      
      //Floor
      ins = Instance{};
      float transformation_floor[12] = { 2.0, 0., 0., 0.,   0., 0.0099, 0.0, -1.0,   0.0, 0.0, 2.0, 0.0 };
      memcpy(ins.transformation, transformation_floor, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 186. / 255., 186. / 255., 186. / 255. };
      instances.push_back(ins);

      
      //BackWall
      ins = Instance{};
      float transformation_back[12] = { 2.0, 0., 0., 0.,   0., 2.0, 0.0, 0.0,   0.0, 0.0, 0.0099, -1.0 };
      memcpy(ins.transformation, transformation_back, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 186. / 255., 186. / 255., 186. / 255. };
      instances.push_back(ins);

      
      //small box
      ins = Instance{};
      float transformation_small[12] = { 0.43301, 0., 0.25, 0.4,   0., 0.66, 0.0, -0.67,   -0.25, 0.0, 0.43301, 0.0 };
      memcpy(ins.transformation, transformation_small, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 0.6 };
      instances.push_back(ins);

      //copy transforms
      //large box 
      ins = Instance{};
      float transformation_large[12] = { 0.43301, 0., 0.25, -0.200,   0., 1.5, 0.0, -0.25,   -0.25, 0.0, 0.43301, -0.200 };
      memcpy(ins.transformation, transformation_large, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 0.6 };
      instances.push_back(ins);

      //light 
      ins = Instance{};
      float transformation_light[12] = { 0.60, 0., 0., 0.,   0., 0.0099, 0.0, 1.0,   0.0, 0.0, 0.60, 0.0 };
      memcpy(ins.transformation, transformation_light, sizeof(float) * 12);
      ins.materialcolor = vec3f{ 1.0 };
      ins.islight = true;
      instances.push_back(ins);
      

      Camera camera = { /*from*/vec3f(0.f,0.f,5.f),
                        /* at */vec3f(0.f,0.f,0.f),
                        /* up */vec3f(0.f,1.f,0.f) };
      camera.m_fovY = 20;

      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = 10.f;

      SampleWindow *window = new SampleWindow("Optix 7 Cornell Box",
                                              base_aabb, instances,camera,worldScale);
      //window->enableInspectMode();

      
      std::cout << "Press 'd' to enable/disable denoising" << std::endl;
      std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
      std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
      window->run();
      
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
      exit(1);
    }
    return 0;
  }
  

