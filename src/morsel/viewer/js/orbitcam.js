function OrbitCamera(basecam) {
	this._cam = basecam;
	this._viewpos = new THREE.Vector3(0.0,0.0,0.0);
	this._theta = 0;
	this._phi = 0;
	this._rate = 0.05;
	this._minradius = 0.001;
	this._maxradius = 20.0;
	this._radius = 5.0;
	this._tarphi = 0.0;
	this._tartheta = 0.0;
	this._reftheta = 0.0;
	this._tracked = false;
	this._trackslope = 0.6;
	this._tracktheta = 0.0;
}

function sphericalToCartesian(phi, theta, rad, offset) {
	var y = Math.sin(phi) * rad;
	var r2 = Math.cos(phi) * rad;
	var x = Math.cos(theta) * r2;
	var z = Math.sin(theta) * r2;
	var ret = new THREE.Vector3(x, y, z);
	if(offset) {
		ret.add(offset);
	}
	return ret;
}

OrbitCamera.prototype.setTracking = function(trackstate) {
	if(trackstate == true) {
		this._tracked = true;
		this._tracktheta = this._reftheta;
	} else {
		this._tracked = false;
		this._viewpos.set(0,0,0);
	}
};

OrbitCamera.prototype.updateTrack = function() {
	var trackdist = -this._trackslope * this._radius;

	this._tracktheta += ( this._reftheta - this._tracktheta ) * this._rate;

	var viewx = Math.cos(this._tracktheta) * trackdist;
	var viewz = Math.sin(this._tracktheta) * trackdist;

	this._viewpos.set(viewx, 0.0, viewz);
};

OrbitCamera.prototype.updateCam = function() {
	if(this._tracked) {
		this.updateTrack();
	}
	this._theta += ( this._tartheta + this._reftheta - this._theta ) * this._rate;
	this._phi 	+= ( this._tarphi - this._phi ) * this._rate;
	this._cam.position.copy(sphericalToCartesian(this._phi, this._theta, this._radius, this._viewpos));
	this._cam.lookAt( this._viewpos );
};

OrbitCamera.prototype.updateSpherical = function(targettheta, targetphi) {
	this._tarphi = targetphi;
	this._tartheta = targettheta;
	this.updateCam();
};

OrbitCamera.prototype.updateScreen = function(sx, sy, sw, sh) {
	var phi = ((sy / sh) * 2.0 - 1.0) * Math.PI;
	var theta = ((sx / sw) * 2.0 - 1.0) * Math.PI * 2.0;
	this.updateSpherical(theta, phi);
};

OrbitCamera.prototype.updateZoomDelta = function(dzoom) {
	this._radius = Math.max(this._minradius, Math.min(this._maxradius, this._radius + dzoom));
	this.updateCam();
};

OrbitCamera.prototype.setRefTheta = function(reftheta) {
	this._reftheta = reftheta;
};

OrbitCamera.prototype.updateScreenDelta = function(dx, dy, sw, sh) {
	var dphi = ((dy / sh) * 1.0) * Math.PI;
	var dtheta = ((dx / sw) * 1.0) * Math.PI * 2.0;
	//console.log("sw: " + sw + ", sh: " + sh);
	//console.log("dphi: " + dphi + ", dtheta: " + dtheta);
	this._tartheta += dtheta;
	this._tarphi = Math.max(-Math.PI/2.0, Math.min(Math.PI/2.0, this._tarphi + dphi));
	//console.log("tphi: " + this._tarphi + ", ttheta: " + this._tartheta);
	this.updateCam();	
};

OrbitCamera.prototype.panScreenDelta = function(dx, dy, sw, sh) {
	var du = (dx / sw) * 2.0;
	var dv = (dy / sh) * 2.0;

	var ct = Math.cos(this._theta);
	var st = Math.sin(this._theta);

	var dz =   du * ct - dv * st;
	var dx =  -du * st - dv * ct;

	this._viewpos.x += dx;
	this._viewpos.z += dz;

	this.updateCam();
};

OrbitCamera.prototype.resetViewPos = function() {
	this._viewpos.set(0,0,0);
};

var ocam;
var mouseDownState = false;
var mouseX = 0, mouseY = 0;
var prevMouseX = 0, prevMouseY = 0;
var mouseDX = 0, mouseDY = 0;

var zoomButtonDown = false;
var zoomButtonKeycode = 16; // shift
var panButtonDown = false;
var panButtonKeycode = 17; // ctrl


function initOrbitCamera(rawcamera) {
	ocam = new OrbitCamera(rawcamera);

	document.addEventListener( 'mousedown', onCamMouseDown, false );
	document.addEventListener( 'mouseup', onCamMouseUp, false );
	document.addEventListener( 'mousemove', onCamDocumentMouseMove, false );
	document.addEventListener( 'keydown', onCamDocumentKeyDown, false );
	document.addEventListener( 'keyup', onCamDocumentKeyUp, false );
}

function updateCamera() {
	mouseDX = mouseX - prevMouseX;
	mouseDY = mouseY - prevMouseY;
	prevMouseX = mouseX;
	prevMouseY = mouseY;

	if(mouseDownState == true) {
		//console.log("Dx: " + mouseDX + ", Dy: " + mouseDY)
		ocam.updateScreenDelta(mouseDX, mouseDY, windowX, windowY);
	} else if(zoomButtonDown) {
		ocam.updateZoomDelta(mouseDY / 10.0);
	} else if(panButtonDown) {
		ocam.panScreenDelta(mouseDX, mouseDY, windowX, windowY);
	} else {
		ocam.updateCam();
	}
}

function onCamMouseDown() {
	mouseDownState = true;
}

function onCamMouseUp() {
	mouseDownState = false;
}

function onWindowResize() {
	getSize();
	camera.aspect = windowX / windowY;
	camera.updateProjectionMatrix();

	renderer.setSize( windowX, windowY );
}

function onCamDocumentKeyDown(event) {
	if(event.keyCode == zoomButtonKeycode) {
		zoomButtonDown = true;
	}
	if(event.keyCode == panButtonKeycode) {
		panButtonDown = true;
	}
}

function onCamDocumentKeyUp(event) {
	if(event.keyCode == zoomButtonKeycode) {
		zoomButtonDown = false;
	}
	if(event.keyCode == panButtonKeycode) {
		panButtonDown = false;
	}
}

function onCamDocumentMouseMove( event ) {
	mouseX = event.clientX - windowHalfX;
	mouseY = event.clientY - windowHalfY;
}