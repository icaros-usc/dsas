function FrameManager(options) {
	this._frames = {};

	this._options = {
		root: null,
		autocreate: true,
		autosubscribe: false
	};

	$.extend(this._options, options || {});
}

FrameManager.prototype.createNode = function(frameid) {
	var newnode = new THREE.Object3D();
	// Warning: Will fail if no root given in options
	this._options.root.add(newnode);

	return newnode;
};

FrameManager.prototype.setNodeTransform = function(node, pos, qrot, scale) {
	if(pos) {
		if(pos.length && pos.length == 3) {
			node.position.set(pos[0], pos[1], pos[2]);
		} else {
			node.position.set(pos.x, pos.y, pos.z);
		}
	}

	if(qrot) {
		if(qrot.length && qrot.length == 4) {
			node.quaternion.set(qrot[0], qrot[1], qrot[2], qrot[3]);
		} else {
			node.quaternion.set(qrot.x, qrot.y, qrot.z, qrot.w);
		}
		node.useQuaternion = true;
	}

	if(scale) {
		if(typeof(scale) === 'number') {
			node.scale.set(scale, scale, scale);
		} else if(scale.length && scale.length == 3) {
			node.scale.set(scale[0], scale[1], scale[2]);
		} else {
			node.scale.set(scale.x, scale.y, scale.z);
		}
	}
};

FrameManager.prototype.setFrameTransform = function(frameid, pos, qrot, scale) {
	var frame = this.getFrame(frameid);
	if(!frame) {
		console.log("FrameID [" + frameid + "]" + 
					"didn't exist and wasn't created-- can't set transform.");
		return;
	}
	this.setNodeTransform(frame.node, pos, qrot, scale);
};

FrameManager.prototype.subscribeFrame = function(frameid) {
	var frame = this.getFrame(frameid);
	if(!frame) {
		console.log("Frame " + frameid + " doesn't exist, can't subscribe.");
		return;
	}

	if(!this._options.tfClient) {
		console.log("No tfClient specified in options--" + 
					"can't subscribe frame " + frameid);
		return;	
	}

	if(frame.subscribed) {
		console.log("Frame " + frameid + " is already subscribed!");
		return;
	}

	// listen for TF updates
	var closureFrameID = frameid;
	var closureTarget = this;

	this._options.tfClient.subscribe(closureFrameID, function(msg) {
		// ROS message unpacking junk
		var tf = new ROSLIB.Transform(msg);

		// update the frame (leaving scale as null to not change it)
		closureTarget.setFrameTransform(closureFrameID, 
										tf.translation, 
										tf.rotation, 
										null);
	});

	frame.subscribed = true;
};

FrameManager.prototype.hasFrame = function(frameid) {
	return (frameid in this._frames);
};

FrameManager.prototype.addFrame = function(frameid) {
	var newnode = this.createNode(frameid);
	var newframe = {frameid: frameid, node: newnode};
	this._frames[frameid] = newframe;
	if(this._options.autosubscribe) {
		this.subscribeFrame(frameid, newframe);
	}
	return newframe;
};

FrameManager.prototype.getFrame = function(frameid) {
	if( this.hasFrame(frameid) ) {
		return this._frames[frameid];
	} else if ( this._options.autocreate ) {
		return this.addFrame(frameid);
	}
};