
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <strstream>
#include "DisplayCalculator.h"
#include "mat4x4.h"
#include "Mesh.h"
#include "CPU/mainCPU.h"
#include "Light.h"
#include "SceneData.h"

// includes CUDA
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "defines.h"

using namespace std;

namespace GPU
{

	DisplayCalculator displayCalculator;
	char  windowTitle[50];
	int lastMouseX;
	int lastMouseY;
	bool lmbDown = false;
	GLuint bufferID;
	GLuint textureID;
	float angleX = 0;
	float angleY = 0;
	float deltaTime = 0.033;

	void checkGLError()
	{
		GLenum err;
		while(( err = glGetError())){
			std::cout << err;
		}
	}
	void updateWorldMatrix()
	{
		angleX += 0.5f*deltaTime;
		angleY += 0.7f*deltaTime;
		displayCalculator.mesh.SetWorldMatrix(getRotationMatrix(angleX, angleY, 0));
	}

	void Display()
	{
		StopWatchInterface *timer = 0;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);

		glClearColor(1.0,0.0,1.0,1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		checkGLError();
		updateWorldMatrix();
		displayCalculator.mesh.UpdateMeshVertices();
		displayCalculator.GenerateDisplay();
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
				displayCalculator.mapWidth, displayCalculator.mapHeight,  GL_BGRA,GL_UNSIGNED_BYTE,NULL);
		checkGLError();
		glBegin(GL_QUADS);
			glTexCoord2f(0,1.0f);
			glVertex3f(0,0,0);

			glTexCoord2f(0,0);
			glVertex3f(0,1.0f,0);

			glTexCoord2f(1.0f,0);
			glVertex3f(1.0f,1.0f,0);

			glTexCoord2f(1.0f,1.0f);
			glVertex3f(1.0f,0,0);
		glEnd();
		glFlush();
		glutSwapBuffers();

		sdkStopTimer(&timer);
		deltaTime = sdkGetTimerValue(&timer)/1000.0;
		snprintf(windowTitle, 50, "Raycasting Triangles - %f FPS", 1.0/deltaTime);
		glutSetWindowTitle(windowTitle);
		sdkDeleteTimer(&timer);
		glutPostRedisplay();
	}
	void Reshape(int width, int height)
	{
		glViewport(0,0,width, height);

		glMatrixMode( GL_PROJECTION );

		glLoadIdentity();
		gluOrtho2D(0,1,0,1);
		cudaGLUnmapBufferObject(bufferID);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4,NULL,GL_DYNAMIC_COPY);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA8,width, height, 0, GL_BGRA,GL_UNSIGNED_BYTE,NULL);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		displayCalculator.mapWidth = width;
		displayCalculator.mapHeight = height;
		displayCalculator.SetCameraFieldOfView(5.0f*width/height, 5.0f);
		printf("Width = %d, height = %d\n", width, height);
		cudaGLMapBufferObject((void **)&displayCalculator.colorMap,bufferID);
		glutPostRedisplay();

	}
	void MouseMotion(int x, int y)
	{
		lastMouseX = x;
		lastMouseY = y;

	}

	void bindTexture(int width, int height)
	{
		//alocate gl buffer
		glGenBuffers(1, &bufferID);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4,NULL,GL_DYNAMIC_COPY);
		cudaGLRegisterBufferObject(bufferID);
		//allocate gl texture
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1,&textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA8,width, height, 0, GL_BGRA,GL_UNSIGNED_BYTE,NULL);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		displayCalculator.mapWidth = width;
		displayCalculator.mapHeight = height;
		cudaGLMapBufferObject((void **)&displayCalculator.colorMap,bufferID);
		getLastCudaError("Failed to bind texture");

	}

	void CreateDefaultMesh()
	{
		const int pointsLen = 4;
		const int trianglesLen = 12;
		float3 points[pointsLen]
					  {
					make_float3(sqrt(8.f/9.f), 0.0f, -1.f/3.f),
					make_float3(-sqrt(2.f/9.f), sqrt(2.f/3.f), -1.f/3.f),
					make_float3(-sqrt(2.f/9.f), -sqrt(2.f/3.f), -1.f/3.f),
					make_float3(0.0f, 0.0f, 1.0f)
					  };
		int triangles[trianglesLen]
					  {
					2,0,1,
					3,2,1,
					2,3,0,
					1,0,3
					  };

		displayCalculator.mesh.SetPoints(points,pointsLen);
		displayCalculator.mesh.SetTriangles(triangles,trianglesLen);
	}

	bool LoadMeshFromFile(char * filename)
	{
		ifstream f(filename);
		if(!f.is_open())
		{
			printf("failed to load from file %s\n", filename);
			return false;
		}
		vector<float3> vertices;
		vector<int> triangles;
		while(!f.eof())
		{
			char line[128];
			f.getline(line, 128);

			strstream s;
			s << line;

			char junk;

			if(line[0] == 'v')
			{
				float3 v;
					s >> junk >> v.x >> v.y >> v.z;
					vertices.push_back(v);
			}
			if(line[0] == 'f')
			{
				int a[3];
					s >> junk >> a[0] >> a[1] >> a[2];
				triangles.push_back(a[0]-1);
				triangles.push_back(a[1]-1);
				triangles.push_back(a[2]-1);
			}
		}

		displayCalculator.mesh.SetPoints(vertices.data(),vertices.size());
		displayCalculator.mesh.SetTriangles(triangles.data(),triangles.size());
		return true;
	}

	void CreateMesh(int argc, char **argv)
	{
		if(argc != 2)
		{
			CreateDefaultMesh();
		}
		else
		{
			if(!LoadMeshFromFile(argv[1]))
			{
				CreateDefaultMesh();
			}
		}
		printf("triangles=%d, vertices=%d\n", displayCalculator.mesh.trianglesLength, displayCalculator.mesh.pointsLength);
		displayCalculator.mesh.CopyToDevice();
		getLastCudaError("failed copy to device");

		displayCalculator.SetCameraPosition(make_float3(0.0f, 0.0f, -5.0f));
		displayCalculator.SetCameraFieldOfView(5.0f, 5.0f);
		displayCalculator.sceneData.lights.push_back(
				Light(make_float3(1.0f, 0.0f, 0.0f), make_float3(-2.0f, 0.0f, -2.0f))
		);

		displayCalculator.sceneData.lights.push_back(
				Light(make_float3(0.0f, 0.0f, 1.0f), make_float3(2.0f, 0.0f, -2.0f))
		);
		displayCalculator.sceneData.SendLightsToGPU();
	}

	void StartGL(int argc, char **argv)
	{
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
		glutInitWindowSize(600, 600);
		glutCreateWindow("Raycasting triangles");
		glewInit();
		glutDisplayFunc(Display);
		glutReshapeFunc(Reshape);
		glutPassiveMotionFunc(MouseMotion);
		bindTexture(600,600);
		CreateMesh(argc, argv);
		glutMainLoop();
	}

}


int main(int argc, char **argv)
{
	if(argc > 1 && 0 == strcmp(argv[1], "CPU"))
	{
		CPU::mainCPU(argc, argv);
	}
	else
	{
		GPU::StartGL(argc, argv);
	}
	return 0;
}
