# VoxelMorph++ with ThoraxCBCT

Example submission for the OncoReg Type 3 Challenge using VoxelMorph++ (Voxelmorph with keypoint supervision and multi-channel instance optimisation) with data from the ThoraxCBCT Type 2 Challenge Task.  
Participants may use this as a template for the containerisation of their submissions. 
To submit to the challenge every solution must be packaged as a Docker container including training and inference in case of a DL solution or any required scripts to output displacementfields with a classical solution.

Make sure to include a requirements.txt file and preferably base your solution on torch 2.1.2+cu121 as this is tested on our setup.
For easy containerisation you may use the Dockerfile provided as this is guarenteed to work with our infrastructure. Make sure to include all files requiring copying in the Dockerfile.

Please include logging in you submission, as can be seen in this example. 

It is advised to copy our train.sh / test.sh structure and you may also look at our data loading process as it is easy adaptable from ThoraxCBCT to OncoReg.


Build the docker:

```
docker build -t vxmpp /examples/VoxelMorph++/
```

Run docker and start training:

```
docker run ...
```

Run inference:

```
docker run ...
```


