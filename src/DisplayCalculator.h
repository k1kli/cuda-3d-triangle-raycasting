/*
 * DisplayCalculator.h
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#ifndef DISPLAYCALCULATOR_H_
#define DISPLAYCALCULATOR_H_

#include "Mesh.h"

class DisplayCalculator {
	float fovWidth;
	float fovHeight;
public:
	float3 cameraPosition;
	float3 lookAt;
	float3 cameraUpDirection;
	int * d_colorMap;
	Mesh mesh;
	int mapWidth;
	int mapHeight;
	DisplayCalculator();
	virtual ~DisplayCalculator();
	void GenerateDisplay();
	void GenerateDisplayPerspective();
	void SetCameraPosition(float3 position);
	void SetCameraLookAt(float3 lookAt, float3 upDirection);
	void SetCameraFieldOfView(float width, float height);
};

#endif /* DISPLAYCALCULATOR_H_ */
