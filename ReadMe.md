About the project:-
This project aims to extract semantic (walls, doors/windows, room-type) information from architectural floor plan images using a multi-task deep learning network. The network has two branches:- the Boundaries branch which is used to identify physical boundaries (walls, doors/windows) in floor plans and the Room branch which is used to identify the room type. Connections exist from the boundaries branch to the room branch so as to help the model learn contextual information using direction aware kernels. Please have a look at the project report for more details. I also extracted quantitative information like room contours, room-wise carpet-area, room perimeter etc from the predictions of the predictions of the multi-task network. I worked on this project during my internship at CCTech, Pune. 

Directories:-
1. Contours: Contains scripts and notebooks that enable extraction of quatitative information (room contours, room-wise carpet area) from the predictions of the Boundaries branch of the multi-task deep learning network.
2. MasksGeneration: Contains scripts and notebooks that generate target masks (boundaries masks as well as room-type masks) from annotations corresponding to raw images.
3. Model:
   a. BoundariesBranch: Contains scripts & notebooks for implementing & training only the Boundaries branch of the network. 
   b. CombinedBranches: Contain scripts & notebooks for implementing & training both branches together.
4. ProjectReport: Contains the project report in both .tex and .pdf formats.
