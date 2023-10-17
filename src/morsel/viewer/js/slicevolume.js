// Slice volume renderer
// Author: Pyry Matikainen (@PyryM, pkmatikainen@gmail.com)
// 

function SliceVolume(options, shaderlib) {
    this._options = {
        thickness: 0.1,
        nslices: 20,
        nearDepth: 0.1,
        farDepth: 1.0,
        nearWidth: 0.15,
        farWidth: 1.5,
        nearHeight: 0.1,
        farHeight: 1.0,
        aspectRatio: 1.0,
        depthencoding: [0.33, 0.33, 0.33, 0.0],
    };
    $.extend(this._options, options);
    this._encoding = new THREE.Vector4();
    this._encoding.fromArray(this._options.depthencoding);
    this._root = this._options.root;

    this._shaderlib = shaderlib;

    this._slices = [];
    this._buildVolume();
}

SliceVolume.prototype.setThickness = function(newthickness) {
    this._thickness = newthickness;
    // reverse for loop is slightly faster
    // (might matter if you have a large number of slices)
    for (var i = this._slices.length - 1; i >= 0; i--) {
        this._slices[i].mat.uniforms["thickness"].value = newthickness;
    };
};

SliceVolume.prototype._computeCameraParams = function() {
    if(!(this._options.fov)) {
        return;
    }
    var aspect = this._options.aspectRatio;

    var slope = Math.tan(this._options.fov / 2.0);
    this._options.nearWidth     = slope * this._options.nearDepth;
    this._options.farWidth      = slope * this._options.farDepth;

    this._options.nearHeight    = this._options.nearWidth / aspect;
    this._options.farHeight     = this._options.farWidth  / aspect;

    this._slope = slope;
};

SliceVolume.prototype.sliceSizeAt = function(depth) {
    return [this._slope * depth, this._slope * depth / this._options.aspectRatio];
};

SliceVolume.prototype._buildVolume = function() {
    this._computeCameraParams();

    var d0 = this._options.nearDepth;
    var dd = (this._options.farDepth - this._options.nearDepth) / this._options.nslices;

    var w0 = this._options.nearWidth;
    var dw = (this._options.farWidth - this._options.nearWidth) / this._options.nslices;

    var h0 = this._options.nearHeight;
    var dh = (this._options.farHeight - this._options.nearHeight) / this._options.nslices;


    var width, height, depth;

    for(var i = 0; i < this._options.nslices; ++i) {
        width   = w0 + i*dw;
        height  = h0 + i*dh;
        depth   = d0 + i*dd;

        var curslice = this._buildSlice(width, height, depth);
        this._slices.push(curslice);
        
        this._root.add(curslice.node);
    }
};

SliceVolume.prototype._buildSlice = function(width, height, depth) {
    var shaderUniforms = {
                           thickness: {type: "f", value: this._options.thickness},
                           sliceDepth: {type: "f", value: depth}, 
                           depthDecodeParams: {type: "v4", value: this._encoding},
                         };

    var vshader, fshader;
    if(this._options.combinedmap) {
        vshader = this._shaderlib.getShader("vs_depthslice_uni");
        fshader = this._shaderlib.getShader("fs_depthslice_uni");
        shaderUniforms["depthUVParams"] = {type: "v4", value: new THREE.Vector4(0.5, 1.0, 0.0, 0.0)};
        shaderUniforms["colorUVParams"] = {type: "v4", value: new THREE.Vector4(0.5, 1.0, 0.5, 0.0)};
        shaderUniforms["combinedMap"] = {type: "t", value: this._options.combinedmap};
    } else {
        vshader = this._shaderlib.getShader("vs_depthslice");
        fshader = this._shaderlib.getShader("fs_depthslice");
        shaderUniforms["depthMap"] = {type: "t", value: this._options.depthmap};
        shaderUniforms["colorMap"] = {type: "t", value: this._options.colormap};
    }

    // build material
    var shaderopts = {
        uniforms:       shaderUniforms,
        vertexShader:   vshader,
        fragmentShader: fshader, 
        side: THREE.DoubleSide
    };

    var slicemat = new THREE.ShaderMaterial(shaderopts);
    var slicegeo = new THREE.PlaneGeometry(width, height);
    var slicenode = new THREE.Mesh(slicegeo, slicemat);
    slicenode.position.set(0, 0, depth);

    return {node: slicenode, geo: slicegeo, mat: slicemat};
};