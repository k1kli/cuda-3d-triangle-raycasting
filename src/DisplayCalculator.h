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
	float4 cameraPosition;
	float4 lookAt;
	float4 cameraUpDirection;
	float fovWidth;
	float fovHeight;
public:
	int * d_colorMap;
	Mesh mesh;
	int mapWidth;
	int mapHeight;
	DisplayCalculator();
	virtual ~DisplayCalculator();
	void GenerateDisplay();
	void SetCameraPosition(float4 position);
	void SetCameraLookAt(float3 lookAt, float3 upDirection);
	void SetCameraFieldOfView(float width, float height);
};

#endif /* DISPLAYCALCULATOR_H_ */
