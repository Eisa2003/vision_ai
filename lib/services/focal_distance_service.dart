import 'package:flutter/painting.dart';
import 'base_distance.dart';

class FocalLengthDistanceService extends BaseDistanceService {
  // Calibrated on SAM z flip 5 — friend/me at exactly 2 m
  // Friend height: 6ft = 1.829 m | boxHeight: 483px on 480×720 frame
  // focalLength = (483 × 2.0) / 1.829 = 527 px
  static const double _focalLength = 527.0;

  // My height: 5'7" = 1.702 m
  // Friend's height: 6'0" = 1.829 m
  // Using my height as default since app is built around me
  static const double _realHeight = 1.702;

  static const double _minDistance = 0.3;
  static const double _maxDistance = 15.0;

  @override
  String get name => 'Focal length';

  @override
  String get description => 'More accurate · calibrated to person height';

  @override
  String estimate(Rect boundingBox, Size imageSize) {
    final m = estimateMeters(boundingBox, imageSize);
    if (m <= _minDistance) return '< ${_minDistance.toStringAsFixed(1)} m';
    if (m >= _maxDistance) return '> ${_maxDistance.toStringAsFixed(0)} m';
    return '~${m.toStringAsFixed(1)} m';
  }

  @override
  double estimateMeters(Rect boundingBox, Size imageSize) {
    final pixelHeight = boundingBox.height;
    if (pixelHeight <= 0) return _maxDistance;

    final meters = (_realHeight * _focalLength) / pixelHeight;
    return meters.clamp(_minDistance, _maxDistance);
  }
}