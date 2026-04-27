// FILE: lib/services/distance_service.dart

import 'package:flutter/painting.dart';
import 'base_distance.dart';

/// Original heuristic estimator — now implements [BaseDistanceService].
/// Formula: distance ≈ referenceArea / normalisedBoxArea
/// Behaviour is identical to before.
class HeuristicDistanceService extends BaseDistanceService {
  // Calibrated: shelf at ~2 m → box 277×265 on 480×720 frame
  // _referenceArea = 2.0 × (277 × 265) = 146,810 → rounded to 150,000. At 2 meters, the box (for the reference object) takes up ~20% of the frame → 150,000 is ~20% of 480×720 (345,600).
  static const double _referenceArea = 150000.0;
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
    print('📦 BOX: w=${boundingBox.width.toInt()} h=${boundingBox.height.toInt()} | IMAGE: ${imageSize.width.toInt()}x${imageSize.height.toInt()}');
    final boxArea = boundingBox.width * boundingBox.height;
    if (boxArea <= 0) return _maxDistance;
    final imageArea = imageSize.width * imageSize.height;
    final normalised = boxArea / imageArea; // how much of the frame the box takes up
    return ((_referenceArea / imageArea) / normalised) // refArea/imageArea gives us the normalised area of the reference object, 
                                                       // dividing by the normalised box area gives us how many "reference objects" fit in the box, which relates to distance.
                                                       // so like 1 reference object -> distance = 2m (based on calibration), 4 reference objects → distance = 1m, etc.
        .clamp(_minDistance, _maxDistance);
  }
}
