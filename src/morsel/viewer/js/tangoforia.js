TangoforiaUtils = {};

TangoforiaUtils.unpackVFTransform = function(vftf) {
	var a = vftf;
	var rtemp = new THREE.Matrix4();
	rtemp.set(-a[1], -a[0],  -a[2], a[3],
	 		   a[5], a[4],  a[6], -a[7],
	 		   a[9], a[8], a[10], -a[11],
	 		     0,    0,    0,   1);
	// rtemp.set( a[0], a[1],  a[2], a[3],
	//  		   a[4], a[5],  a[6], a[7],
	//  		   a[8], a[9], a[10], a[11],
	//  		     0,    0,    0,   1);
	//target.rotation.setFromRotationMatrix(rtemp);
	//var ptemp = new THREE.Vector3(a[3], -a[7], -a[11]);
	
	//console.log(ptemp);
	//target.position.copy(ptemp);
	//return {rotation: rtemp, position: ptemp};
	return rtemp;
};

TangoforiaUtils.computeOffset = function(reportedPoseMat, truePoseMat) {
	var offsetMat = new THREE.Matrix4();
	var inverseMat = new THREE.Matrix4();

	// we want M_offset * M_reported = M_true
	// so M_offset = M_true * M_reported.inv
	inverseMat.getInverse(reportedPoseMat);
	offsetMat.multiplyMatrices(truePoseMat, inverseMat);

	// console.log("source (reported): ");
	// console.log(matToStr(reportedPoseMat));
	// console.log("destination (true):");
	// console.log(matToStr(truePoseMat));
	// console.log("inverse:");
	// console.log(matToStr(inverseMat));
	// console.log("result:");
	// console.log(matToStr(offsetMat));

	return offsetMat;
};

TangoforiaUtils.decomposeMat = function(mat) {
	var retPosition = new THREE.Vector3();
	var retQuaternion = new THREE.Quaternion();
	var retScale = new THREE.Vector3();

	mat.decompose(retPosition, retQuaternion, retScale);

	return {pos: retPosition, rot: retQuaternion, scale: retScale};
};

TangoforiaUtils.worldFromMarker = function(markerRelPoseMat, markerTruePoseMat) {
	if(!markerTruePoseMat) {
		markerTruePoseMat = new THREE.Matrix4();
		markerTruePoseMat.identity();
	}

	return TangoforiaUtils.computeOffset(markerRelPoseMat, markerTruePoseMat);
};

function Tangoforia(root, camera) {
	this._referenceMarkers = {};
	this._offsetNode = new THREE.Object3D();
	this._tangoNode = new THREE.Object3D();
	root.add(this._offsetNode);
	this._offsetNode.add(this._tangoNode);

	this._cameraMat = new THREE.Matrix4();
	this._cameraMat.identity();
	this._camInverse = new THREE.Matrix4();
	this._camInverse.getInverse(this._cameraMat);

	var cam = camera;
	this._camera = cam;
	this._tangoNode.add(cam);

	this.tangoPosStr = "0,0,0";
	this.tangoRotStr = "0,0,0,0";

	this._lastTangoMat = new THREE.Matrix4();
	this._lastTangoMat.identity();
	this._lastMarker = null;

	// set the camera offset
	var offsetpos = new THREE.Vector3();
	offsetpos.set(0,0,0);
	var offsetquat = new THREE.Quaternion();
	offsetquat.setFromEuler( new THREE.Euler(10.0 * Math.PI / 180.0, 0, 0, 'XYZ') );

	this.setCameraOffset(offsetpos, offsetquat);

	this.verbose = false;
}

Tangoforia.prototype.addReferenceMarker = function(markername, transform) {
	this._referenceMarkers[markername] = {mat: transform};
};

function matToStr(m) {
	var ret = "[";
	var e = m.elements;
	ret += ([e[0], e[4], e[8], e[12]].join(",") + "\n");
	ret += ([e[1], e[5], e[9], e[13]].join(",") + "\n");
	ret += ([e[2], e[6], e[10], e[14]].join(",") + "\n");
	ret += ([e[3], e[7], e[11], e[15]].join(",") + "\n");
	ret += "]";
	return ret;
}

function quatToStr(q) {
	var ret = "[" + q.x + ", " + q.y + ", " + q.z + ", " + q.w + "]";
	return ret;	
}

Tangoforia.prototype.updateOffset = function() {
	console.log("Tango:");
	console.log(matToStr(this._lastTangoMat));
	console.log(quatToStr(this._lastTangoQuat));
	console.log(this._rawTangoPose);

	if(!this._lastMarker){
		console.log("Couldn't update offset: no visible marker!");
		return;
	}
	console.log("Updating offset!");

	var markername = this._lastMarker.name;
	var posemat = this._lastMarker.posemat;

	var truePoseMat = this._referenceMarkers[markername].mat;
	var reportedPoseMat = posemat;

	var vuforiaWorldMat = TangoforiaUtils.worldFromMarker(reportedPoseMat,
												   truePoseMat);
	var tangoWorldMat = new THREE.Matrix4();
	tangoWorldMat.multiplyMatrices(vuforiaWorldMat, this._camInverse);

	var offsetmat = TangoforiaUtils.computeOffset(this._lastTangoMat,
												  tangoWorldMat);

	console.log("Marker:");
	console.log(matToStr(this._lastMarker.posemat));
	console.log("World:");
	console.log(matToStr(tangoWorldMat));
	console.log("Tango-->World:");
	console.log(matToStr(offsetmat));

	var pos = new THREE.Vector3();
	var rot = new THREE.Quaternion();
	var scale = new THREE.Vector3();

	offsetmat.decompose(pos, rot, scale);

	if(this.verbose) {
		console.log("----OFFSET UPDATE----")
		console.log(pos);
		console.log(rot);
		console.log(scale);
		console.log("---------------------");
	}

	this._offsetNode.position.copy(pos);
	this._offsetNode.quaternion.copy(rot);
};

Tangoforia.prototype.setCameraOffset = function(position, rotation) {
	this._camera.position.copy(position);
	this._camera.quaternion.copy(rotation);

	this._cameraMat.makeRotationFromQuaternion(this._camera.quaternion);
	this._cameraMat.setPosition(this._camera.position);
	this._camInverse.getInverse(this._cameraMat);
};

Tangoforia.prototype.getTangoWorldPos = function() {
	var vector = new THREE.Vector3();
	vector.setFromMatrixPosition( this._tangoNode.matrixWorld );
	return vector;
};

Tangoforia.prototype.updateTango = function(tango_pose) {

	var p = tango_pose;
	this.tangoPosStr = p.slice(0,3).join(",");
	this.tangoRotStr = p.slice(3,7).join(",");

	this._lastTangoQuat = new THREE.Quaternion();
	this._lastTangoQuat.set(p[3], p[4], p[5], p[6]);
	this._rawTangoPose = p.slice();

	this._tangoNode.position.set(p[0], p[1], p[2]);
	this._tangoNode.quaternion.set(p[3], p[4], p[5], p[6]);

	this._lastTangoMat.makeRotationFromQuaternion(this._tangoNode.quaternion);
	this._lastTangoMat.setPosition(this._tangoNode.position);

	if(this.verbose) {
		console.log("-----TANGO UPDATE----")
		console.log(this.tangoPosStr);
		console.log(this.tangoRotStr);
		console.log("---------------------");
	}
};

Tangoforia.prototype.update = function(vuforiaData) {
	var ret = {};
	var marker;

	this._lastMarker = null;
	for(var i = 0; i < vuforiaData.length; ++i) {
		marker = vuforiaData[i];
		if(marker.name == "TANGO") {
			this.updateTango(marker.tango_pose);
		} else if(marker.name in this._referenceMarkers) {
			//console.log(marker);
			//this.updateOffset(marker.name, 
			//				  TangoforiaUtils.unpackVFTransform(marker.pose));
			this._lastMarker = {name: marker.name,
								posemat: TangoforiaUtils.unpackVFTransform(marker.pose),
								rawpose: marker.pose};
		} else {
			ret[marker.name] = TangoforiaUtils.unpackVFTransform(marker.pose);
		}
	}

	return ret;
};

TangoforiaUtils.onVFLoad = function(bla) {
	console.log("Loaded!");
	// Activate the halloween dataset (note the two callbacks in strings, for success and failure).
	WebVuforia.activateDataset(vuforia_xml, TangoforiaUtils.onVFError, TangoforiaUtils.onVFError);
	console.log("Activated!");
};

TangoforiaUtils.onVFError = function(errstuff) {
	console.log(errstuff);	
};

TangoforiaUtils.init_vuforia = function(datasetxml, datasetdat, callback) {
	vuforia_xml = datasetxml;

	console.log("Loading datasset...");
	WebVuforia.loadDataset(datasetxml, datasetdat, TangoforiaUtils.onVFLoad, TangoforiaUtils.onVFError);
	console.log("Called load.");

	// Set the update callback to a function in a string.
	WebVuforia.onUpdate(callback);
};