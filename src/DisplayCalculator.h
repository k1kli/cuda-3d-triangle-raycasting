/*
 * DisplayCalculator.h
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#ifndef DISPLAYCALCULATOR_H_
#define DISPLAYCALCULATOR_H_

#include "Mesh.h"
#include "SceneData.h"

class DisplayCalculator {
	float fovWidth;
	float fovHeight;
	bool onCPU;
	void GenerateDisplayCPU();
	unsigned int GetColorOfClosestHitpointCPU(float3 & rayStartingPoint);
	unsigned int CalculateLightCPU(float3 & hitPoint, float3 &  normalVector);
public:
	float3 cameraPosition;
	int * colorMap = nullptr;
	Mesh mesh;
	int mapWidth;
	int mapHeight;
	SceneData sceneData;
	DisplayCalculator(bool onCPU = false);
	virtual ~DisplayCalculator();
	void GenerateDisplay();
	void SetCameraPosition(float3 position);
	void SetCameraFieldOfView(float width, float height);
};

#endif /* DISPLAYCALCULATOR_H_ */
