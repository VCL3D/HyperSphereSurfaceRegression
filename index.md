![](./assets/img/qual_samples.jpg "Qualitative results on a sample of the 360IndoorsDataset test-set")

# Abstract
Omnidirectional vision is becoming increasingly relevant as more efficient 360o image acquisition is now possible.
However, the lack of annotated 360<sup>o</sup> datasets has hindered the application of deep learning techniques on spherical content. 
This is further exaggerated on tasks where ground truth acquisition is difficult, such as monocular surface estimation. 
While recent research approaches on the 2D domain overcome this challenge by relying on generating normals from depth cues 
using RGB-D sensors, this is very difficult to apply on the spherical domain. In this work, we address the unavailability 
of sufficient 360<sup>o</sup> ground truth normal data, by leveraging existing 3D datasets and remodelling them via rendering. 
We present a dataset of 360<sup>o</sup> images of indoor spaces with their corresponding ground truth surface normal, 
and train a deep convolutional neural network (CNN) on the task of monocular 360<sup>o</sup> surface estimation. 
We achieve this by minimizing a novel angular loss function defined on the hyper-sphere using simple quaternion algebra. 
We put an effort to appropriately compare with other state of the art methods trained on planar datasets and finally, 
present the practical applicability of our trained model on a spherical image re-lighting task using completely unseen data by 
qualitatively showing the promising generalization ability of our dataset and model.

# Paper
[![](./assets/img/paper_thumb.png "paper link")](https://arxiv.org/)
