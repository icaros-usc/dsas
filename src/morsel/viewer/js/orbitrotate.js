function OrbitRotator(targetobj) {
	this._target = targetobj;
	this._theta = 0;
	this._phi = 0;
	this._rate = 0.05;
	this._tarphi = 0.0;
	this._tartheta = 0.0;
	this._viewpos = new THREE.Vector3(0.0, 0.0, 0.0);
}

OrbitRotator.prototype.updateCam = function() {
	this._theta += ( this._tartheta - this._theta ) * this._rate;
	this._phi 	+= ( this._tarphi - this._phi ) * this._rate;
	this._target.position.copy(this._viewpos);
	this._target.rotation.set(this._phi, this._theta, 0.0);
};

OrbitRotator.prototype.updateZoomDelta = function(dzoom) {
	this._viewpos.z += dzoom;
	this.updateCam();
};

OrbitRotator.prototype.updateScreenDelta = function(dx, dy, sw, sh) {
	var dphi = ((dy / sh) * 1.0) * Math.PI;
	var dtheta = ((dx / sw) * 1.0) * Math.PI * 2.0;
	//console.log("sw: " + sw + ", sh: " + sh);
	//console.log("dphi: " + dphi + ", dtheta: " + dtheta);
	this._tartheta += dtheta;
	this._tarphi = Math.max(-Math.PI/2.0, Math.min(Math.PI/2.0, this._tarphi + dphi));
	//console.log("tphi: " + this._tarphi + ", ttheta: " + this._tartheta);
	this.updateCam();	
};

OrbitRotator.prototype.panScreenDelta = function(dx, dy, sw, sh) {
	var du = (dx / sw) * 2.0;
	var dv = (dy / sh) * 2.0;

	this._viewpos.x += du;
	this._viewpos.y -= dv;

	this.updateCam();
};

OrbitRotator.prototype.resetViewPos = function() {
	this._viewpos.set(0,0,0);
};

var ocam;
var mouseDownState = false;
var mouseX = 0, mouseY = 0;
var prevMouseX = 0, prevMouseY = 0;
var mouseDX = 0, mouseDY = 0;
var sMX = 0.0, sMY = 0.0;

var zoomButtonDown = false;
var zoomButtonKeycode = 16; // shift
var panButtonDown = false;
var panButtonKeycode = 17; // ctrl

var userMouseClick = null;


function initOrbitRotator(target) {
	ocam = new OrbitRotator(target);

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
	if(userMouseClick) {
		userMouseClick(sMX, sMY);
	}
}

function onCamMouseUp() {
	mouseDownState = false;
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
	sMX = event.clientX / windowX;
	sMY = event.clientY / windowY;
}