# FuzzyClustering

- Fuzzy C-Means法の実装．
- 入力はOpenCVのcv::Matを想定．

## 機能．

- Fuzzy C-Meansクラスタリング．
- k-means++法の初期化方法を選択可能．
- 距離指標を選択可能．

## Usage

    cv::Mat dataset = (cv::Mat_<float> (4, 2)
      << 
      0, 0,
      5, 4,
      100, 150,
      200, 102);

    static unsigned int number_clusters = 2;    // クラスタ数
    float fuzziness = 1.5;                      // 乱雑さ
    float epsilon = 0.01;                       // 終了閾値
 
    SoftCDistType dist_type = kSoftCDistL2;         // 距離指標
    SoftCInitType init_type = kSoftCInitKmeansPP;   // 初期化方法
    SoftC::Fuzzy f (dataset, number_clusters, fuzziness, epsilon, dist_type, init_type);
    unsigned int num_iterations = 100;          // 繰り返しの回数
    f.clustering (num_iterations);

    cv::Mat centroids = f.get_centroids_ ();
    std::cout << centroids << std::endl;

    cv::Mat memberships = f.get_membership_ ();
    std::cout << memberships << std::endl;
}
#### Copyright (c) 2014 Shinichi Goto All rights reserved.
