/*
 * SceneData.h
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#ifndef SCENEDATA_H_
#define SCENEDATA_H_

class SceneData {
public:
	float3 * lightColor;
	float3 * lightPos;
	int lightsCount
	uint objectColor = 0xFFFFFFFF;
	SceneData();
	virtual ~SceneData();
	void SetLights(float3 * lightColors, float3 * lightPositions, int lightsCount)
};

#endif /* SCENEDATA_H_ */
