# FuzzyCMeans

* Fuzzy C-Means clustering implementation using OpenCV.

## Usage

```cpp
    cv::Mat dataset = (cv::Mat_<float> (4, 2)
      << 
      0, 0,
      5, 4,
      100, 150,
      200, 102);

    static unsigned int number_clusters = 2;
    float fuzziness = 1.5;
    float epsilon = 0.01;                       // Threshold
 
    SoftCDistType dist_type = kSoftCDistL2;
    SoftCInitType init_type = kSoftCInitKmeansPP;
    SoftC::Fuzzy f (dataset, number_clusters, fuzziness, epsilon, dist_type, init_type);
    unsigned int num_iterations = 100;
    f.clustering (num_iterations);

    cv::Mat centroids = f.get_centroids_ ();
    std::cout << centroids << std::endl;

    cv::Mat memberships = f.get_membership_ ();
    std::cout << memberships << std::endl;
}
```

#### Copyright (c) 2014 Shinichi Goto All rights reserved.
