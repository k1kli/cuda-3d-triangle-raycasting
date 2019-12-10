
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

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
#include "../DisplayCalculator.h"
#include "../mat4x4.h"
#include "../Mesh.h"
#include "mainCPU.h"

// includes CUDA
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "../defines.h"

using namespace std;

namespace CPU
{

	DisplayCalculator displayCalculator(true);
	char  windowTitle[50];
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
		glRasterPos2d(-1.0, -1.0);
		checkGLError();
		updateWorldMatrix();
		displayCalculator.mesh.UpdateMeshVertices();
		displayCalculator.GenerateDisplay();
		glDrawPixels(displayCalculator.mapWidth, displayCalculator.mapHeight,
				GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, displayCalculator.colorMap);
		checkGLError();
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
		displayCalculator.mapWidth = width;
		displayCalculator.mapHeight = height;
		if(displayCalculator.colorMap != nullptr)
		{
			delete[] displayCalculator.colorMap;
		}
		displayCalculator.colorMap = new int[width*height];
		displayCalculator.SetCameraFieldOfView(5.0f*width/height, 5.0f);
		printf("Width = %d, height = %d\n", width, height);
		glutPostRedisplay();

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
		if(argc != 3)
		{
			CreateDefaultMesh();
		}
		else
		{
			if(!LoadMeshFromFile(argv[2]))
			{
				CreateDefaultMesh();
			}
		}
		printf("triangles=%d, vertices=%d\n", displayCalculator.mesh.trianglesLength, displayCalculator.mesh.pointsLength);

		displayCalculator.SetCameraPosition(make_float3(0.0f, 0.0f, -5.0f));
		displayCalculator.SetCameraFieldOfView(5.0f, 5.0f);
		displayCalculator.sceneData.lights.push_back(
				Light(make_float3(1.0f, 0.0f, 0.0f), make_float3(-2.0f, 0.0f, -2.0f))
		);

		displayCalculator.sceneData.lights.push_back(
				Light(make_float3(0.0f, 0.0f, 1.0f), make_float3(2.0f, 0.0f, -2.0f))
		);
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
		CreateMesh(argc, argv);
		glutMainLoop();
	}


	int mainCPU(int argc, char **argv)
	{
		StartGL(argc, argv);
		return 0;
	}

}
