#include "fuzzy_clustering.hpp"
#include <boost/foreach.hpp>
#include <iostream>

namespace SoftC {

  void Fuzzy::initRandom () {
    //
    // メンバーシップをランダム初期化
    //
    std::cout << "C-means Random Initialization" << std::endl;

    srand ((unsigned int) time (NULL));
    float normalization_factor;

    for (int j = 0 ; j < number_points_; j++){
      normalization_factor = 0.0;
      for (int i = 0; i < number_clusters_; i++)	
        normalization_factor += 
          membership_.at<float> (j, i) = (rand () / (RAND_MAX + 0.0));
      // 正規化
      for (int i = 0; i < number_clusters_; i++)
        membership_.at<float> (j, i) /= normalization_factor;
    }

    // centroids算出
    computeCentroids();
  }

  void Fuzzy::initKmeansPP () {
    //
    // k-means++の手法で初期化する場合
    //
    srand ((unsigned int) time (NULL));

    std::vector<int> center_indexes (0);
    std::vector<bool> already_selected_indexes (number_points_, false);

    // ランダムに最初の重心を選択
    int first_index = rand () % number_points_;

//    std::cout << "first index: " << first_index << std::endl;

    center_indexes.push_back (first_index);
    already_selected_indexes[first_index] = true;

    while (center_indexes.size () < number_clusters_) {

      // すべてのデータ点に対して
      // 最近傍の重心を見つけその距離を算出
      std::vector<float> nearest_distances (number_points_, 0.0);

      for (int p = 0; p < number_points_; ++p) {

        // 既にcentroidsとして選択済みの場合は考えない
        if (already_selected_indexes[p])
          continue;

        // pは各点のindexに対応
        cv::Mat point = rows_.row (p);
        std::vector<float> distances_from_centers (0);

        // 全ての重心との距離を計算
        for (int c = 0; c < center_indexes.size (); ++c) {
          int center_index = center_indexes[c];
          cv::Mat center = rows_.row (center_index);
          float dist = calc_dist (point, center, kSoftCDistL2);
          distances_from_centers.push_back (dist);
        }

        // 最近傍の重心を見つける
        int nearest_center_index = center_indexes[0];
        float min = distances_from_centers[0];
        for (int c = 1; c < distances_from_centers.size (); ++c) {
          float dist = distances_from_centers[c];
          if (dist < min) {
            min = dist;
            nearest_center_index = center_indexes[c];
          }
        }

        nearest_distances[p] = min;
      }

      assert (nearest_distances.size () == number_points_);

      // 上記のうち距離が最長のものを重心に加える
      float max = nearest_distances[0];
      float max_index = 0;
      for (int p = 1; p < nearest_distances.size (); ++p) {
        float dist = nearest_distances[p];
        if (dist > max) {
          max = dist;
          max_index = p;
        }
      }

      // 重心として選択
      center_indexes.push_back (max_index);
      already_selected_indexes[max_index] = true;
    }

    // centroidsを上記の点にセットする
    for (int j = 0; j < center_indexes.size (); ++j) {
      // FIXME
      // めっちゃworkaroundだが，完全一致だとupdate_membershipで
      // 分母が0になってしまったときに問題が．．．
      for (int d = 0; d < dimension_; ++d) {
        centroids_.at<float> (j, d) = rows_.at<float> (center_indexes[j], d) + 0.001;
      }
    }

    // membershipをアップデート
    updateMembership ();
    // 初期化からははみ出るが，ふたたびcenroids算出
    computeCentroids2();
  }

  void Fuzzy::initEverything () {
    switch (init_type_) {
      case kSoftCInitRandom:
        initRandom ();
        break;
      case kSoftCInitKmeansPP:
        initKmeansPP ();
        break;
      default:
        break;
    }
  }

  //
  // centroidsの初期化
  //
  void Fuzzy::computeCentroids(){
    
    // centroidの更新
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)	
        for (int f = 0; f < dimension_; f++)
          centroids_.at<float> (j, f) += membership_.at<float> (i, j) * rows_.at<float> (i, f);
    //    *p_centroids_ = prod (*p_membership_,        rows_);
    //   n_clusters   =      n_clusters          rows.size1()
    // X [rows.size2()=      X [rows.size1()=    X [rows.size2=
    //    =size_of_a_point_]    =number_points_]    size_of_a_point]  
    std::vector<float> sum_uk (number_clusters_, 0);
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)
        sum_uk[j] += membership_.at<float> (j, i);
    // 正規化
    for (int j = 0; j < number_clusters_; j++)
      for (int f = 0 ; f < dimension_; f++)
        centroids_.at<float> (j, f) /= sum_uk[j];
  }

  // 初期化以外
  void Fuzzy::computeCentroids2 (){
    cv::Mat u_ji_m  = cv::Mat::zeros (number_points_, number_clusters_, CV_32FC1);
    float normalization;

    //　初期化
    for (int j = 0; j < number_clusters_; j++)
      for (int f = 0; f < dimension_; f++)
        centroids_.at<float> (j, f) = 0.0;
    // 重みをfuzziness乗したものを計算する
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)
        u_ji_m.at<float> (i, j) = pow ( membership_.at<float> (i, j), fuzziness_);
    // centroidの更新．この後に正規化（分母）が必要
    for (int j = 0; j < number_clusters_; j++)
      for (int i = 0 ; i < number_points_; i++)	
        for (int f = 0; f < dimension_; f++)
          centroids_.at<float> (j, f) += u_ji_m.at<float> (i, j) * rows_.at<float> (i, f);

    // 点の正規化
    for (int j = 0; j < number_clusters_; j++){
      normalization = 0.0;
      for (int i = 0 ; i < number_points_; i++) 
        normalization += u_ji_m.at<float> (i, j);
      for (int f = 0; f < dimension_; f++)
        centroids_.at<float> (j, f) /= normalization;
    }
  }

  float Fuzzy::calc_dist (
      const cv::Mat &point,   // 行ベクトル
      const cv::Mat &center,  // 行ベクトル
      const SoftCDistType dist_type
      )
  {
    float f_dist = 0.f;
    int dimension = point.cols;

    switch (dist_type) {
      case kSoftCDistL1:
        {
          // L1, マンハッタン
          for (int d = 0; d < dimension; d++) {
            f_dist += fabs (point.at<float> (0,d) - center.at<float> (0,d));
          }
        }
        break;
      case kSoftCDistL2:
        {
          // L2, ユークリッド
          for (int d = 0; d < dimension; d++) {
            float t = point.at<float> (0,d) - center.at<float> (0,d);
            f_dist += t * t;
          }
        }
        break;
      case kSoftCDistHistInter:   // 未実装
        {
          // HIstogram intersection
          // computer vision最先端ガイド3より
          float sum_p = 0.f;
          for (int d = 0; d < dimension; d++) {
            float p = point.at<float> (0,d);
            float c = center.at<float> (0,d);
            float min = p < c ? p : c;
            f_dist += min;
//            f_dist += ((p + c - fabs (p - c)) / 2);
            sum_p += p;
          }
          f_dist /= sum_p;
        }
        break;
      default:
        std::cout << "Error while calculating distance for clustering"
          << std::endl;
        break;
    }
    return f_dist;
  }

  //
  // メンバーシップを更新  
  //     
  bool Fuzzy::updateMembership () {

    // i番目の点とj番目のクラスタの中心点の距離(norm)を格納
    cv::Mat matrix_norm_one_xi_minus_cj 
      = cv::Mat::zeros (number_clusters_, number_points_, CV_32FC1);

    //
    // 距離の初期化
    //
    for (unsigned int i = 0 ; i < number_points_; i++)
      for (unsigned int j = 0; j < number_clusters_; j++)
        matrix_norm_one_xi_minus_cj.at<float> (j, i) = 0.0;

    for (unsigned int i = 0 ; i < number_points_; i++) {
      // 各クラスタからの距離を計算
      cv::Mat point = rows_.row (i);
      for (unsigned int j = 0; j < number_clusters_; j++) {
        cv::Mat center = centroids_.row (j);
        matrix_norm_one_xi_minus_cj.at<float> (j, i) 
          = calc_dist (point, center, dist_type_);
      }
    }

    float coeff;
    for (unsigned int i = 0 ; i < number_points_; i++)
      for (unsigned int j = 0; j < number_clusters_; j++){
        coeff = 0.0;
        for (unsigned int k = 0; k < number_clusters_; k++) {

//        if (matrix_norm_one_xi_minus_cj.at<float> (j, i) == 0) {
//          coeff += pow (0, 2.0 / (fuzziness_ - 1.0));
//        } else if (matrix_norm_one_xi_minus_cj.at<float> (k, i) == 0) {
//          coeff += pow (1000000.0, 2.0 / (fuzziness_ - 1.0));
//        } else {
          coeff += 
            pow ( (matrix_norm_one_xi_minus_cj.at<float> (j, i) /
                  matrix_norm_one_xi_minus_cj.at<float> (k, i)) , 
                2.0 / (fuzziness_ - 1.0) );
        }

//        if (coeff == 0) {
//          new_membership_.at<float> (i, j) = 1.0;
//        } else {
          new_membership_.at<float> (i, j) = 1.0 / coeff;
//        }
      }

    if (!can_stop() ){
      // 終了しない場合は更新
      membership_ = new_membership_.clone ();
      return false; 
    }
    return true;
  }
};
