# cmd-Social-Distancing-with-OpenCV

In this example I have connected to street webcam in Yalta and detecting social distance violations in real time with CUDA acceleration.
~~~
// Only one class for this detection
vector<string> class_names{ “person” };
// Webcam live stream
string video = “http://91.223.48.19:557/cgueSCEI?container=mjpeg&stream=main";
~~~
Calculate distance between pairs of objects (find center of rectangle)
~~~
                        // Distance between pairs of rectangles
                        auto maxheight = max(rect1.height, rect2.height);
                        auto distx = abs((rect1.x + rect1.width / 2) - (rect2.x + rect2.width / 2));
                        auto disty = abs((rect1.y + rect1.height / 2) - (rect2.y + rect2.height / 2));
                        auto dist = sqrt(distx * distx + disty * disty);
                        
                        // If distance < height of rectangle add value to vector
                        if (dist < maxheight)
                        {
                            violators.push_back(idx);
                            violators.push_back(jdx);
                        }
~~~
Then mark violators with Red label and count in one frame:
~~~
vio_ss << “Violators: “ << violators.size();
~~~
https://youtu.be/zm_EQdNIasw
