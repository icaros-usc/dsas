function MarkerManager(options) {
	this._options = {
		root: null,
	};

	this._markers = {};

	$.extend(this._options, options || {});

	this._root = this._options.root || new THREE.Object3D();
}

MarkerManager.prototype.addMeshMarker = function(id, meshurl) {
	// body...
};

MarkerManager.prototype.removeMarker = function(id) {
	// body...
};