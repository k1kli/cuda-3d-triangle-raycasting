
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

// includes CUDA
#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "defines.h"

using namespace std;

DisplayCalculator displayCalculator;
char  windowTitle[50];
int lastMouseX;
int lastMouseY;
bool lmbDown = false;
GLuint bufferID;
GLuint textureID;
float angleY = 0;
float angleX = 0;

void checkGLError()
{
    GLenum err;
    while(( err = glGetError())){
        std::cout << err;
    }
}

float3 rotationXRows[] = {
		make_float3(1,0,0),
		make_float3(0,1,0),
		make_float3(0,0,1)
};
float3 rotationYRows[] = {
		make_float3(1,0,0),
		make_float3(0,1,0),
		make_float3(0,0,1)
};

void UpdateRotationMatrices()
{
	float sX = sin(angleX);
	float sY = sin(angleY);
	float cX = cos(angleX);
	float cY = cos(angleY);
	rotationXRows[1] = make_float3(0, cX, -sX);
	rotationXRows[2] = make_float3(0, sX,  cX);

	rotationYRows[0] = make_float3(cY, 0, sY);
	rotationYRows[2] = make_float3(-sY, 0, cY);
}
float3 multiplyPoint3x3(float3 point, float3 matrix3x3Rows[3])
{
	return make_float3(dot(point, matrix3x3Rows[0]),dot(point, matrix3x3Rows[1]),dot(point, matrix3x3Rows[2]));
}

void UpdateCameraPosition()
{
	angleY += 0.02f*PI*(lastMouseX-displayCalculator.mapWidth/2)/displayCalculator.mapWidth;
	angleX += 0.02f*PI*(lastMouseY-displayCalculator.mapHeight/2)/displayCalculator.mapHeight;
	UpdateRotationMatrices();
	float distance = 10;
	float3 camera = make_float3(0,0, -distance);
	float3 cameraPlusUp = camera+make_float3(0,1.f,-distance);
	camera = multiplyPoint3x3(camera, rotationXRows);
	camera = multiplyPoint3x3(camera, rotationYRows);
	cameraPlusUp = multiplyPoint3x3(cameraPlusUp, rotationXRows);
	cameraPlusUp = multiplyPoint3x3(cameraPlusUp, rotationYRows);
	displayCalculator.SetCameraPosition(camera);
	displayCalculator.SetCameraLookAt(displayCalculator.lookAt,cameraPlusUp);
}

void Display()
{
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	glClearColor(1.0,0.0,1.0,1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	checkGLError();
	UpdateCameraPosition();
	displayCalculator.GenerateDisplayPerspective();
//	int k = cudaGLUnmapBufferObject(bufferID);
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
    snprintf(windowTitle, 50, "Raycasting Triangles - %f FPS", 1000.0/sdkGetTimerValue(&timer));
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
	printf("Width = %d, height = %d\n", width, height);
	cudaGLMapBufferObject((void **)&displayCalculator.d_colorMap,bufferID);
	lastMouseX = width/2;
	lastMouseY = height/2;
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
	cudaGLMapBufferObject((void **)&displayCalculator.d_colorMap,bufferID);
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
	displayCalculator.mesh.RecalculateNormals();
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
	displayCalculator.mesh.RecalculateNormals();
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

	UpdateCameraPosition();
	displayCalculator.SetCameraLookAt(make_float3(0.0f, 0.0f, 0.0f),make_float3(0,1.f,0));
	displayCalculator.SetCameraFieldOfView(20.0f, 20.0f);
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


int main(int argc, char **argv)
{
	StartGL(argc, argv);
	return 0;
}
