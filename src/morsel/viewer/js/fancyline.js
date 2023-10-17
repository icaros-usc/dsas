function FancyLine(options, builder) {
	this._builder = builder;
	this._vecpts = [];
	this._shaderAttributes = {pivotPos: {type: "v3", value: []},
		pivotDir: {type: "v3", value: []}, 
		arcPos: {type: "f", value: []},
		linecolor: {type: "v3", value: []}};
	this._npts = options.npts;
	this.buildGeometry();

	this._shaderUniforms = {lineWidth: {type: "f", value: options.linewidth || 0.3}};
	this._vshader = shaderlibrary.getShader("vs_screenline");
	this._t = 0;

	this._shaderMaterial = builder.getStyleMaterial(this._vshader, 
		this._shaderAttributes, this._shaderUniforms, options.style);

	this._vnode = new THREE.Mesh(this._geometry, this._shaderMaterial);
	this.update(options);
};

FancyLine.prototype.clear = function() {
	console.log("Deleting fancy line...");
};

FancyLine.prototype.update = function(newdata) {
	this._builder.applyCommonProperties(this, newdata);
	//this.genTestUpdate(0.0);
	if('points' in newdata){
		this.updateGeometry(newdata.points);
	}
};

FancyLine.prototype.genTestUpdate = function(phase) {
	var pts = [];
	for(var i = 0; i < this._npts; ++i) {
		var t = i * 0.01 + phase;
		var pt = [Math.cos(t*5.0)*10.0, Math.cos(t*7.0)*1.0, Math.cos(t*11.0)*10.0];
		pts.push(pt);
	}
	this.updateGeometry(pts);
};

FancyLine.prototype.getNode = function() {
	return this.getTHREENode();
};

FancyLine.prototype.getTHREENode = function() {
	return this._vnode;
};

FancyLine.prototype.getTHREEMat = function() {
	return this._shaderMaterial;
};

FancyLine.prototype.updateGeometry = function(newpts) {
	if(newpts.length != this._npts) {
		console.log("Point number mismatch for fancy line: expected " + newpts.length + ", got: "
			+ this._npts);
		// TODO: make this do something more sensible like fill the remainder with zeros
		return;
	};

	// convert everything to three vectors
	var vecpts = this._vecpts;
	var ntoadd = (newpts.length - vecpts.length);
	for(var i = 0; i < ntoadd; ++i) {
		vecpts.push(new THREE.Vector3(0.0,0.0,0.0));
	}

	//console.log("newptslen: "+newpts.length);
	//console.log("vecptslen: "+vecpts.length);

	for(var i = 0; i < newpts.length; ++i) {
		//console.log("newptsi: " + newpts[i]);
		//console.log("vecptsi: " + vecpts[i]);
		vecpts[i].fromArray(newpts[i]);
	};

	var pivotDirs = this._shaderAttributes.pivotDir.value;
	var pivotPoses = this._shaderAttributes.pivotPos.value;
	var arcPoses = this._shaderAttributes.arcPos.value;

	// The slightly less pretty way of choosing the pivot directions:
	// just use the direction of the next link
	// TODO: use average of previous+next links	

	var arcpos = 0;
	for(var i = 1; i < this._npts; ++i) {
		var p0 = vecpts[i-1];
		var p1 = vecpts[i];
		pivotDirs[2*i+0].subVectors(p0, p1);
		pivotDirs[2*i+1].subVectors(p1, p0);
		pivotPoses[2*i+0].copy(p1);
		pivotPoses[2*i+1].copy(p1);
		arcpos += pivotDirs[2*i+0].length();
		arcPoses[2*i+0] = arcpos;
		arcPoses[2*i+1] = arcpos;
	};
	// deal specially with first point
	pivotDirs[0].copy(pivotDirs[2]);
	pivotDirs[1].copy(pivotDirs[3]);
	pivotPoses[0].copy(vecpts[0]);
	pivotPoses[1].copy(vecpts[0]);

	this._shaderAttributes.pivotDir.needsUpdate = true;
	this._shaderAttributes.pivotPos.needsUpdate = true;
	this._shaderAttributes.arcPos.needsUpdate = true;
	//this._shaderAttributes.linecolor.needsUpdate = true;
};

FancyLine.prototype.buildGeometry = function() {
	var npts = this._npts;
	console.log("Building a line with " + npts + "points.");
	var posList = this._shaderAttributes.pivotPos.value;
	var dirList = this._shaderAttributes.pivotDir.value;
	var arcList = this._shaderAttributes.arcPos.value;
	var colorList = this._shaderAttributes.linecolor.value;

	this._geometry = new THREE.Geometry();

	var uv0 = new THREE.Vector3(0.0,0.0);
	var uv1 = new THREE.Vector3(1.0,0.0);
	var face_uv0 = [uv0, uv1, uv1];
	var face_uv1 = [uv0, uv1, uv0];

	// generate 2n vertices
	for (var i = 0; i < npts; ++i ) {
		var vertex0 = new THREE.Vector3();
		// TODO: initialize these in some better way
		vertex0.x = i;
		vertex0.y = 0;
		vertex0.z = 0;
		var vertex1 = vertex0.clone();
		vertex1.y = 1.0;

		this._geometry.vertices.push( vertex0 );
		this._geometry.vertices.push( vertex1 );
		posList.push(new THREE.Vector3(0,1,0));
		posList.push(new THREE.Vector3(0,1,0));
		dirList.push(new THREE.Vector3(0,1,0));
		dirList.push(new THREE.Vector3(0,1,0));
		arcList.push(i);
		arcList.push(i);
		colorList.push(new THREE.Vector3(1.0 - i/npts, i/npts, 0));
		colorList.push(new THREE.Vector3(1.0 - i/npts, i/npts, 0));
	}

	// generate 2(n-1) faces
	// don't worry about normals
	for (var i = 0; i < npts-1; ++i) {
		var v0 = i*2;
		var v1 = v0 + 1;
		var v2 = v0 + 2;
		var v3 = v0 + 3;

		var face0 = new THREE.Face3( v0, v1, v3 );
		var face1 = new THREE.Face3( v0, v3, v2 );
		this._geometry.faces.push(face0);
		this._geometry.faces.push(face1);
		this._geometry.faceVertexUvs[0].push(face_uv0);
		this._geometry.faceVertexUvs[0].push(face_uv1);
	}
};

FancyLine.prototype.addToScene = function(scene) {
	this._scene = scene;
	scene.add(this._vnode);
};

FancyLine.prototype.removeFromScene = function() {
	this._scene.remove(this._vnode);
};

FancyLine.prototype.setVisible = function(visible) {
	this._vnode.visible = visible;	
	if(this._labelobj) {
		this._labelobj.setVisible(visible);
	}
};

FancyLine.prototype.setPosition = function(x, y, z) {
	this._vnode.position.set(x, y, z);
};

FancyLine.prototype.addTextLabel = function(text) {
	this._tnode = spriteFromText(text);
	this._vnode.add(this._tnode);
	this._tnode.position.set(0, 0.5, 0);
};

FancyLine.prototype.updateAnimation = function(dt) {
	this._t += dt * 0.001;
	this.genTestUpdate(this._t);
};