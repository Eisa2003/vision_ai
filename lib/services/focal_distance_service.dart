import 'package:flutter/painting.dart';
import 'base_distance.dart';

class FocalLengthDistanceService extends BaseDistanceService {
  // Example values (MUST calibrate these)
  static const double _focalLength = 800; // pixels
  static const double _realWidth = 0.5;   // meters (example: person shoulder width)

  @override
  String get name => 'Focal length';

  @override
  String get description => 'More accurate · requires calibration';

  @override
  String estimate(Rect boundingBox, Size imageSize) {
    final m = estimateMeters(boundingBox, imageSize);
    print('Estimated distance using focal length: ${m.toStringAsFixed(2)} m');
    return '~${m.toStringAsFixed(2)} m';
  }

  @override
  double estimateMeters(Rect boundingBox, Size imageSize) {
    final pixelWidth = boundingBox.width;
    if (pixelWidth <= 0) return 10.0;

    return (_realWidth * _focalLength) / pixelWidth;
  }
}