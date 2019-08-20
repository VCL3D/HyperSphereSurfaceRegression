
<p align="center">
  <img src = "./assets/img/360HyperSphereBanner.png" alt="Qualitative Results" width="800"/>
</p>

<p align="center">
<table>
  <tr>
    <th>Loss Function</th>
    <th>Mean</th>
    <th>Median</th>
    <th>RMSE</th>
    <th>5<sup>o</sup></th>
    <th>11.25<sup>o</sup></th>
    <th>22.5<sup>o</sup></th>
    <th>30<sup>o</sup></th>
  </tr>
  <tr>
    <th>L<sub>2</sub></th>
    <th>7.72</th>
    <th>7.23</th>
    <th>8.39</th>
    <th>73.55</th>
    <th>79.88</th>
    <th>87.72</th>
    <th>90.43</th>
  </tr>
  <tr>
    <th>Cosine</th>
    <th>7.63</th>
    <th>7.14</th>
    <th>8.31</th>
    <th>73.89</th>
    <th>80.04</th>
    <th>87.29</th>
    <th>90.48</th>
  </tr>
  <tr>
    <th>Quaternion</th>
    <th>7.24</th>
    <th>6.72</th>
    <th>7.98</th>
    <th>75.8</th>
    <th>80.59</th>
    <th>87.3</th>
    <th>90.37</th>
  </tr>
  <tr>
    <th>Quaternion + Smooth</th>
    <th>7.14</th>
    <th>6.66</th>
    <th>7.88</th>
    <th>76.16</th>
    <th>80.82</th>
    <th>87.45</th>
    <th>90.47</th>
  </tr>
</table>
</p>

<h1 align="center"> Abstract </h1>
<p style="text-align: justify;">
Omnidirectional vision is becoming increasingly relevant as more efficient 360<sup>o</sup> image acquisition is now possible.
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
</p>

<h1 align="center"> Paper </h1>

<p align="center">
  <a href="https://arxiv.org/">
    <img src="./assets/img/paper_thumb_small.png" alt="Paper on Arxiv">
  </a>
</p>

<h1 align="center"> Data </h1>
<p align="center">
  Coming Soon...
</p>
<h1 align="center"> Model </h1>
<p align="center">
  Coming Soon...
</p>
<h1 align="center"> Citation </h1>
<p style="
    width: auto;
    background-color: #f2f2f2;
    font-size: small;
">
  <pre>
    <code>
      Coming Soon...
    </code>
  </pre>
</p>
