// FILE: lib/services/base_distance.dart

import 'package:flutter/painting.dart';

/// The contract every distance estimator must fulfil.
///
/// To add a new estimator:
///   1. Create a class that extends [BaseDistanceService].
///   2. Implement [estimate] and [estimateMeters].
///   3. Register it in [DistanceType] and [AppProviders].
abstract class BaseDistanceService {
  /// Human-readable name shown in the Settings UI.
  String get name;

  /// Short description shown below the name.
  String get description;

  /// Returns a display string, e.g. "~1.4 m" or "< 0.3 m".
  ///
  /// [boundingBox] is in **image pixel** coordinates.
  /// [imageSize]   is the native resolution of the source frame / image.
  String estimate(Rect boundingBox, Size imageSize);

  /// Returns the raw estimated distance in metres.
  double estimateMeters(Rect boundingBox, Size imageSize);
}

/// Registry of available distance strategies.
enum DistanceType {
  heuristic,
  focalLength,
  midas,
}

extension DistanceTypeLabel on DistanceType {
  String get label {
    switch (this) {
      case DistanceType.heuristic:
        return 'Heuristic (area)';
      case DistanceType.focalLength:
        return 'Focal length';
      case DistanceType.midas:
        return 'Midas';
    }
  }
}
