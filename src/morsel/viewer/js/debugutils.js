function MarkerSimulator(mnames) {
	this.markers = {};
	var curname;
	var curmarker;
	for(var i = 0; i < mnames.length; ++i) {
		curname = mnames[i];
		curmarker = {rotation: new THREE.Matrix4(), position: new THREE.Vector3(),
					 freqx: Math.random() * 3.0 - 1.5, freqy: Math.random() * 3.0 - 1.5,
					 rotvelx: Math.random() * 0.5, rotvely: Math.random() * 0.5};
		this.markers[curname] = curmarker;
	}
	this.t = 0.0;
}

MarkerSimulator.prototype.update = function(dt) {
	this.t += dt;
	var curmarker;
	var rotx, roty, rotz;
	var x, z;
	for(var name in this.markers) {
		curmarker = this.markers[name];
		x = Math.cos(curmarker.freqx * this.t) * 30.0;
		z = Math.sin(0.08 * this.t) * 30.0;
		// rotx = curmarker.rotvelx * this.t;
		// roty = curmarker.rotvely * this.t;
		rotx = Math.PI * 0.0;
		roty = Math.PI * 0.5;
		rotz = Math.PI * 0.0;
		//curmarker.position.set(x, z, z*0.1 - 100.0);
		curmarker.position.set(0, 0, z*1.5 - 100.0);
		var euler = new THREE.Euler(rotx, roty, rotz);
		curmarker.rotation.makeRotationFromEuler(euler);
	}

	return this.markers;
};