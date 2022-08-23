# nfl_projects
Series of computer vision applications to American Football using NFL footage

The first part of this series consisted of using OpenCV to crop a video around a specific player's movement on the pitch. 
For instance, the video below is the output of the script for the player with the label H27:


https://user-images.githubusercontent.com/40761922/186257258-cd979b7d-98c8-4263-8c8a-68ad16d08c71.mov


The second part consisted of using the MediaPipe API to create a real-time 3D coordinate plot of the player:
![filename](https://user-images.githubusercontent.com/40761922/186255050-272db0fe-3c3b-4248-a2f9-560fee965c03.gif)

Evidently, there needs to be some more work done on it. Firstly, the coordinates are a bit innacurate. Secondly, it could only be exported as a GIF for some reason
