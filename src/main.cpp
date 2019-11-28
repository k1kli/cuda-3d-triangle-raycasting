
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "DisplayCalculator.h"

// includes CUDA
#include <cuda_runtime.h>
#include <helper_timer.h>

using namespace std;

DisplayCalculator displayCalculator;
char  windowTitle[50];
int mouseXPos;
int mouseYPos;
bool lmbDown = false;
GLuint bufferID;
GLuint textureID;

void checkGLError()
{
    GLenum err;
    while(( err = glGetError())){
        std::cout << err;
    }
}

void Display()
{
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	glClearColor(1.0,1.0,1.0,1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2d(-1.0, -1.0);
	int width = glutGet(GLUT_WINDOW_WIDTH);
	int height = glutGet(GLUT_WINDOW_HEIGHT);
	checkGLError();
	cudaGLMapBufferObject((void **)&displayCalculator.d_colorMap,bufferID);
	cudaGLUnmapBufferObject(bufferID);
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
	glOrtho(0,1,0,1,-1,1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4,NULL,GL_DYNAMIC_COPY);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA8,width, height, 0, GL_BGRA,GL_UNSIGNED_BYTE,NULL);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	displayCalculator.mapWidth = width;
	displayCalculator.mapHeight = height;
	printf("Width = %d, height = %d\n", width, height);
	glutPostRedisplay();

}
void Mouse(int button, int state, int x, int y)
{
	if(button == GLUT_LEFT_BUTTON)
	{
		if(state==GLUT_DOWN)
			exit(0);
		lmbDown = (state==GLUT_DOWN);
	}
}
void MouseMotion(int x, int y)
{
	mouseXPos = x;
	mouseYPos = y;
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
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	displayCalculator.mapWidth = width;
	displayCalculator.mapHeight = height;

}

void CreateMesh()
{
	const int pointsLen = 8;
	const int trianglesLen = 15;
	float3 points[pointsLen]
	              {
				make_float3(0.0f, 0.0f, 0.0f),
				make_float3(0.0f, 0.0f, 1.0f),
				make_float3(0.0f, 1.0f, 0.0f),
				make_float3(0.0f, 1.0f, 1.0f),
				make_float3(1.0f, 0.0f, 0.0f),
				make_float3(1.0f, 0.0f, 1.0f),
				make_float3(1.0f, 1.0f, 0.0f),
				make_float3(1.0f, 1.0f, 1.0f)
	              };
	int triangles[trianglesLen]
	              {
				4,2,0,
				0,4,1,
				0,1,2,
				4,7,2,
				7,1,2
	              };
	displayCalculator.mesh.SetPoints(points,pointsLen);
	displayCalculator.mesh.SetTriangles(triangles,trianglesLen);
	displayCalculator.mesh.RecalculateNormals();
	displayCalculator.mesh.CopyToDevice();
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
	glutMouseFunc(Mouse);
	glutPassiveMotionFunc(MouseMotion);
	bindTexture(600,600);
	glutMainLoop();
}


int main(int argc, char **argv)
{
	StartGL(argc, argv);
	return 0;
}
