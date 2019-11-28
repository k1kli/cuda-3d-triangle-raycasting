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
public:
	int * d_colorMap;
	Mesh mesh;
	int mapWidth;
	int mapHeight;
	DisplayCalculator();
	virtual ~DisplayCalculator();
	void GenerateDisplay();
	void SetCameraPosition(float4 position);
	void SetCameraLookAt(float4 lookAt);
};

#endif /* DISPLAYCALCULATOR_H_ */
