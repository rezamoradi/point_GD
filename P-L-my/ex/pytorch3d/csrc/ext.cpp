/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <torch/extension.h>

#include "ball_query/ball_query.h"
#include "gather_scatter/gather_scatter.h"
#include "knn/knn.h"
#include "sample_farthest_points/sample_farthest_points.h"
#include "mesh_normal_consistency/mesh_normal_consistency.h"
#include "point_mesh/point_mesh_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  m.def("knn_check_version", &KnnCheckVersion);
#endif
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
  m.def("ball_query", &BallQuery);
  m.def("sample_farthest_points", &FarthestPointSampling);
  m.def("gather_scatter", &GatherScatter);
  
  
  // PointEdge distance functions
  m.def("point_edge_dist_forward", &PointEdgeDistanceForward);
  m.def("point_edge_dist_backward", &PointEdgeDistanceBackward);
  m.def("edge_point_dist_forward", &EdgePointDistanceForward);
  m.def("edge_point_dist_backward", &EdgePointDistanceBackward);
  m.def("point_edge_array_dist_forward", &PointEdgeArrayDistanceForward);
  m.def("point_edge_array_dist_backward", &PointEdgeArrayDistanceBackward);

  // PointFace distance functions
  m.def("point_face_dist_forward", &PointFaceDistanceForward);
  m.def("point_face_dist_backward", &PointFaceDistanceBackward);
  m.def("face_point_dist_forward", &FacePointDistanceForward);
  m.def("face_point_dist_backward", &FacePointDistanceBackward);
  m.def("point_face_array_dist_forward", &PointFaceArrayDistanceForward);
  m.def("point_face_array_dist_backward", &PointFaceArrayDistanceBackward);
  
  
}
