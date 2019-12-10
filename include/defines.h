/*
 * Defines.h
 *
 *  Created on: 18 lis 2019
 *      Author: karol
 */

#ifndef DEFINES_H_
#define DEFINES_H_


#ifndef MAX
#define MAX(a,b) (a)>(b) ? (a) : (b)
#endif
#ifndef MIN
#define MIN(a,b) (a)>(b) ? (b) : (a)
#endif
#define DIVROUNDUP(a, b) ((a)+(b) - 1) / (b)
#define PI 3.1415f


//use this to manipulate scene or modify code in main.cpp or CPU/mainCPU.cpp in functions setMesh and setScene

//between 0.0f and 100.0f
#define SMOOTHNESS 40.0f
//between 0.0f and 1.0f
#define DIFFUSE 0.3f
//between 0.0f and 1.0f
#define SPECULAR 0.7f
//for conservation of energy diffuse and specular should sum up to 1

//rgba format
#define OBJECT_COLOR 0xFFFFFFFF



//uncomment this for multiple lights of different colors in a circle


#define MULTIPLE_LIGHTS
#define LIGHT_COUNT 8


//uncommment this for one light of specific color


//#define SINGLE_LIGHT
//
////all between 0 and 1
//#define LIGHT_COLOR_X 1.0f
//#define LIGHT_COLOR_Y 1.0f
//#define LIGHT_COLOR_Z 1.0f
//
//#define LIGHT_POS_X 1.0f
//#define LIGHT_POS_Y 0.0f
//#define LIGHT_POS_Z -1.0f




#endif /* DEFINES_H_ */
