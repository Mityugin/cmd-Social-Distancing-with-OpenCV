# cmd-Social-Distancing-with-OpenCV

In this example I have connected to street webcam in Yalta and detecting social distance violations in real time with CUDA acceleration.
~~~
// Only one class for this detection
vector<string> class_names{ “person” };
// Webcam live stream
string video = “http://91.223.48.19:557/cgueSCEI?container=mjpeg&stream=main";
~~~
Calculate distance between pairs of objects (find center of rectangle)

Then mark violators with Red label and count in one frame:
~~~
vio_ss << “Violators: “ << violators.size();
~~~
https://youtu.be/zm_EQdNIasw
