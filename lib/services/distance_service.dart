// FILE: lib/services/distance_service.dart

import 'package:flutter/painting.dart';
import 'base_distance.dart';

/// Original heuristic estimator — now implements [BaseDistanceService].
/// Formula: distance ≈ referenceArea / normalisedBoxArea
/// Behaviour is identical to before.
class HeuristicDistanceService extends BaseDistanceService {
  // Tune this by placing a known object at a known distance.
  // A box filling ~25% of a 1080p frame at ~2 m ≈ 120 000.
  static const double _referenceArea = 120000.0;
  static const double _minDistance = 0.3;
  static const double _maxDistance = 15.0;

  @override
  String get name => 'Heuristic (area)';

  @override
  String get description => 'Fast · no calibration · rough estimate';

  @override
  String estimate(Rect boundingBox, Size imageSize) {
    final m = estimateMeters(boundingBox, imageSize);
    if (m <= _minDistance) return '< ${_minDistance.toStringAsFixed(1)} m';
    if (m >= _maxDistance) return '> ${_maxDistance.toStringAsFixed(0)} m';
    return '~${m.toStringAsFixed(1)} m';
  }

  @override
  double estimateMeters(Rect boundingBox, Size imageSize) {
    final boxArea = boundingBox.width * boundingBox.height;
    if (boxArea <= 0) return _maxDistance;
    final imageArea = imageSize.width * imageSize.height;
    final normalised = boxArea / imageArea;
    return ((_referenceArea / imageArea) / normalised)
        .clamp(_minDistance, _maxDistance);
  }
}
