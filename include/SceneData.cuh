/*
 * SceneData.h
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#ifndef SCENEDATA_H_
#define SCENEDATA_H_
#include <vector>
#include "Light.cuh"

class SceneData {
public:
	std::vector<Light> lights;
	SceneData();
	virtual ~SceneData();
	void SendLightsToGPU();
};

#endif /* SCENEDATA_H_ */
