function VirtualButton(parentobj, options) {
	this.node = new THREE.Object3D();
	parentobj.add(this.node);

	this.options = {
		position: new THREE.Vector3(0.0, 0.0, 0.0),
		radius: 1.0
	};
	$.extend(this.options, options);

	this.createCollisionObject();

	this.node.position.copy(this.options.position);

	this.active = true;
	this.onClick = null;
	this.onView = null;
	this.onUpdate = null;
}


VirtualButton.prototype.createCollisionObject = function() {
	var geometry = new THREE.IcosahedronGeometry(this.options.radius, 2);
	var mat = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
	this.collisionObj = new THREE.Mesh( geometry, mat );
	if(!this.options.debug) {
		this.collisionObj.visible = false;
	}

	this.node.add(this.collisionObj);
};

VirtualButton.prototype.addDisplayObject = function(obj) {
	this.dobj = obj;
	this.node.add(obj);
};

VirtualButton.prototype.getWorldPos = function() {
	var wp = new THREE.Vector3();
	wp.setFromMatrixPosition( this.node.matrixWorld );
	return wp;
};

VirtualButton.prototype.getCollider = function() {
	return this.collisionObj;
};

function makeRing(options, shaderlib) {
	var opts = {npts: 20, 
			  linewidth: 0.01, 
			  linetex: null,
			  vmult: 10.0,
			  umult: 0.1};
	$.extend(opts, options);

	var npts = opts.npts;
	var rad = opts.radius;

	var dt = 2.0 * Math.PI / (npts - 1);
	var pts = [];
	for(var i = 0; i < npts; ++i) {
		var theta = dt * i;
		pts.push([Math.cos(theta) * rad, 0.0, Math.sin(theta) * rad]);
	}

	var pathLine = new Orbitline(opts, shaderlib);
	pathLine.updateGeometry(pts);
	return pathLine;
}

function RingButton(options, shaderlib, buttonmanager) {
	this._options = {
		radius: 1.0,
		linewidth: 0.01,
		npts: 20,
		highlight_color: new THREE.Vector4(1.0, 1.0, 0.0, 0.0),
		base_color: new THREE.Vector4(0.5, 0.5, 0.5, 0.0),
		highlight_tex: null,
		base_tex: null,
		debug: false,
		parent: null,
		rotaterate: 1.0,
		selviewtime: 10.0,
		id: "0"
	};
	$.extend(this._options, options);

	this._lastviewed = false;
	this._viewed = false;

	this.startView = null;
	this.endView = null;

	this._selected = false;
	this._theta = 0.0;

	this._viewtime = 0.0;

	console.log("Creating line circle...");
	var pts = [];
	var npts = this._options.npts;
	var rad = this._options.radius;
	var dt = 2.0 * Math.PI / (npts - 1);
	for(var i = 0; i < npts; ++i) {
		var theta = dt * i;
		pts.push([Math.cos(theta) * rad, 0.0, Math.sin(theta) * rad]);
	}
	this._pathLine = new Orbitline({npts: pts.length, 
								  linewidth: this._options.linewidth, 
								  linetex: this._options.linetex || this._options.base_tex,
								  vmult: 10.0,
								  umult: 0.1}, 
								  shaderlib);
	this._pathLine.updateGeometry(pts);

	this._node = new THREE.Object3D();
	if(this._options.parent) {
		this._options.parent.add(this._node);
	}

	this._node.add(this._pathLine.getNode());

	var tempthis = this;
	this._button = new VirtualButton(this._node, {debug: this._options.debug, 
												  radius: this._options.radius});
	this._button.onView = function() {
		tempthis._onView();
	}
	this._button.onClick = function() {
		tempthis._onClick();
	}
	this._button.onUpdate = function() {
		tempthis._onUpdate();
	}

	if(buttonmanager) {
		buttonmanager.addButton(this._button);
	}
}

RingButton.prototype.getNode = function() {
	return this._node;
};

RingButton.prototype._onView = function() {
	// if we weren't viewed last time, then viewing has
	// started
	if(!this._lastviewed) {
		this._startView();
	}

	this._viewed = true;
};

RingButton.prototype._onClick = function() {
	console.log("Clicked " + this._options.id);
	//this._options.linewidth *= 1.2;
	//this._pathLine.setLineWidth(this._options.linewidth);

	if(this.onClick) {
		this.onClick();
	}
};

RingButton.prototype._onUpdate = function() {
	// if we haven't been viewed since the last update,
	// and we were viewed the last time, then viewing
	// has ended
	if(!this._viewed && this._lastviewed) {
		this._endView();
	} else if(this._viewed) {
		this._viewtime += 1.0 / 60.0;
		if(this._viewtime > this._options.selviewtime && !this._selected) {
			this._onClick();
		} else if(!this._selected) {
			this._updateViewStyle();
		}
	}

	this._lastviewed = this._viewed;
	this._viewed = false;
};

RingButton.prototype._updateViewStyle = function() {
	var color0 = new THREE.Vector4();
	color0.copy(this._options.base_color);
	var color1 = new THREE.Vector4();
	color1.copy(this._options.highlight_color);

	var alpha = this._viewtime / this._options.selviewtime;
	if(alpha > 1.0) {
		alpha = 1.0;
	}

	this._pathLine.setLineColorV(color0.lerp(color1, alpha))
};

RingButton.prototype._startView = function() {
	console.log("Start view " + this._options.id);

	this._pathLine.setLineWidth(this._options.linewidth * 2.0);

	if(this.startView) {
		this.startView();
	}
};

RingButton.prototype._endView = function() {
	console.log("End view " + this._options.id);

	this._pathLine.setLineWidth(this._options.linewidth);
	this._viewtime = 0.0;

	if(this.endView) {
		this.endView();
	}
};

RingButton.prototype.getNode = function() {
	return this._node;
};

RingButton.prototype.select = function() {
	console.log("Selected " + this._options.id);

	this._pathLine.setLineColorV(this._options.highlight_color);
	this._pathLine.setLineTexture(this._options.highlight_tex);
	this._selected = true;
};

RingButton.prototype.deselect = function() {
	console.log("Deselected " + this._options.id);

	this._pathLine.setLineColorV(this._options.base_color);
	this._pathLine.setLineTexture(this._options.base_tex);
	this._selected = false;
};

function VirtualButtonManager(manageropts) {
	this._options = {
		allowviewselect: true,
	};
	$.extend(this._options, manageropts);
	this._buttons = [];

	this._raycaster = new THREE.Raycaster();
}

VirtualButtonManager.prototype.addButton = function(button) {
	this._buttons.push(button);
};


// apply a function f to objects along a ray specified by screen position
// (in normalized coordinates) [x,y]
// if closestOnly is set to true, only apply to the closest intersection,
// otherwise apply to every object
VirtualButtonManager.prototype._applyRay = function(x, y, f, closestOnly) {
	var nearest = 10000.0;
	var nearobj = null;

	//console.log("Ray " + x + " " + y);
	this._raycaster.setFromCamera( new THREE.Vector2(x,y), this._cam );

	for(var i = 0; i < this._buttons.length; ++i) {
		var intersects = this._raycaster.intersectObject( this._buttons[i].getCollider(), false );
		if(intersects.length > 0) {
			if(!closestOnly) {
				f(this._buttons[i]);
			} else {
				if(intersects[0].distance < nearest) {
					nearest = intersects[0].distance;
					nearobj = this._buttons[i];
				}
			}
		}
	}

	if(closestOnly && (nearobj != null)) {
		f(nearobj);
	}
};

VirtualButtonManager.prototype.updateClick = function(relX, relY) {
	// scale from [0,1] to [-1,1]
	relX = (relX * 2.0) - 1.0; 
	relY = -(relY * 2.0) + 1.0;

	console.log("Click: " + relX + ", " + relY);

	this._applyRay(relX, relY, function(button) {
		if(button.onClick) {
			button.onClick();
		}
	}, true);
};

VirtualButtonManager.prototype.updateAll = function() {
	for(var i = 0; i < this._buttons.length; ++i) {
		if(this._buttons[i].onUpdate) {
			this._buttons[i].onUpdate();
		}
	}
};

VirtualButtonManager.prototype.updateView = function(cam) {
	this._cam = cam;

	if(this._options.allowviewselect) {
		this._applyRay(0.0, 0.0, function(button) {
			if(button.onView) {
				button.onView();
			}
		}, true);
	}
};


function RingSelectionSet(options, shaderlib) {
	this._options = {
		radius: 0.1,
		npts: 30,
		linewidth: 0.01,
		linetex: null,
		debug: false,
		parent: null,
		selmovethresh: 0.2,
		allowviewselect: true,
		cursor: null,
		cursoralpha: 0.1
	};
	$.extend(this._options, options);

	this._cursor = this._options.cursor;
	this._shaderlib = shaderlib;
	this._root = new THREE.Object3D();
	this._options.parent.add(this._root);
	this._bmanager = new VirtualButtonManager(this._options);
	this._selected = null;

	this._buttons = [];
}

RingSelectionSet.prototype.addCursor = function(cursor) {
	this._cursor = cursor;
};

RingSelectionSet.prototype._updateCursor = function() {
	if(this._selected && this._cursor) {
		var cpos = this._cursor.getNode().position;
		var dpos = this._selected.getNode().position;

		this._cursor.getNode().position.copy(cpos.lerp(dpos, this._options.cursoralpha));
		this._cursor.getNode().visible = true;
	} else if(this._cursor) {
		this._cursor.getNode().visible = false;
	}
};

RingSelectionSet.prototype.update = function(dt) {
	this._updateCursor();
};

RingSelectionSet.prototype.updateButtonLocations = function(newLocations) {
	var prevSelPos = null;
	if(this._selected) {
		prevSelPos = new THREE.Vector3();
		prevSelPos.copy(this._selected.getNode().position);
	}

	// clear selection
	this._selectButton(null);

	// create buttons as necessary
	while(this._buttons.length < newLocations.length) {
		this._createButton();
	}

	// update positions
	var closetdist = 100000.0;
	var closestbutton = null;

	var tempvec = new THREE.Vector3();
	var tempdist = 0.0;


	for(var i = 0; i < this._buttons.length; ++i) {

		if(prevSelPos) {
			tempvec.subVectors(prevSelPos, this._buttons[i].getNode().position);
			tempdist = tempvec.length();
			if(tempdist < closetdist) {
				closetdist = tempdist;
				closestbutton = this._buttons[i];
			}
		}

		if(i < newLocations.length) {
			this._buttons[i].getNode().visible = true;
			this._buttons[i].getNode().position.copy(newLocations[i]);
		} else {
			this._buttons[i].getNode().visible = false;
		}
	}

	// reselect closest button to old selected button's position
	if(closestbutton && closetdist < this._options.selmovethresh) {
		this._selectButton(closestbutton);
	}
};

RingSelectionSet.prototype._selectButton = function(button) {
	for(var i = 0; i < this._buttons.length; ++i) {
		if(this._buttons[i] === button) {
			this._buttons[i].select();
		} else {
			this._buttons[i].deselect();
		}
	}
	this._selected = button;
};

RingSelectionSet.prototype._createButton = function() {
	console.log("Creating line circle...");
	var opts = {};
	$.extend(opts, this._options);
	$.extend(opts, {
		parent: this._root,
		base_tex: this._options.base_tex || this._options.linetex,
		highlight_tex: this._options.highlight_tex || this._options.linetex,
		id: this._buttons.length + 1
	});

	var b = new RingButton(opts, this._shaderlib, this._bmanager);

	var tgt = this;
	b.onClick = function() {
		if(!b._selected){
			tgt._selectButton(b);
		} else { // b._selected
			// deselect the button
			tgt._selectButton(null);
		}
	};

	this._buttons.push(b);
};