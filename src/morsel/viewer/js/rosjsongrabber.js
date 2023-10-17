function ROSJsonGrabber(callback) {
	this.callback = callback;
}

ROSJsonGrabber.prototype.connect = function(srcurl, srcmessage) {
  var ros = new ROSLIB.Ros();
  this.ros = ros;

  // If there is an error on the backend, an 'error' emit will be emitted.
  ros.on('error', function(error) {
    console.log(error);
  });
  // Find out exactly when we made a connection.
  var targetThis = this;
  ros.on('connection', function() {
    console.log('Connection made!');
    targetThis.subscribeStuff();
  });

  ros.on('close', function() {
    console.log('Connection closed.');
  });

  // Create a connection to the rosbridge WebSocket server.
  this.srcmessage = srcmessage;
  ros.connect(srcurl);
};

ROSJsonGrabber.prototype.onMessage = function(message) {
  if(this.callback) {
    var jdata = JSON.parse(message.data);
    this.callback(jdata);
  }
};

ROSJsonGrabber.prototype.subscribeStuff = function() {
    console.log("Subscribing stuff!");
    var targetThis = this;
    listener = new ROSLIB.Topic({
      ros : this.ros,
      name : this.srcmessage,
      messageType : 'std_msgs/String'
    });
    // Then we add a callback to be called every time a message is published on this topic.
    listener.subscribe(function(message) {
      targetThis.onMessage(message);

      // If desired, we can unsubscribe from the topic as well.
      //listener.unsubscribe();
    });
};