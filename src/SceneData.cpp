/*
 * SceneData.cpp
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#include "SceneData.h"
#include <vector>
#include "DisplayCalculatorKernels.h"

using namespace std;

SceneData::SceneData() {
	// TODO Auto-generated constructor stub

}

SceneData::~SceneData() {
	// TODO Auto-generated destructor stub
}

void SceneData::SendLightsToGPU() {
	vector<float3> lightPositions;
	vector<float3> lightColors;
	for(int i = 0; i < lights.size(); i++)
	{
		lightPositions.push_back(lights[i].position);
		lightColors.push_back(lights[i].color);
	}
	UpdateLightsGPU(lightColors.data(), lightPositions.data(), lights.size());
}
