// FILE: lib/services/base_detector.dart

import 'package:flutter/painting.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

/// A single normalised detection result.
/// All detectors — ML Kit, YOLO, anything else — must produce this type.
/// This decouples the screens from any specific model's output format.
class Detection {
  final Rect boundingBox;
  final String label;
  final double confidence;

  const Detection({
    required this.boundingBox,
    required this.label,
    required this.confidence,
  });
}

/// The contract every detector must fulfil.
///
/// To add a new model:
///   1. Create a class that extends [BaseDetector].
///   2. Implement [initialize], [detect], and [close].
///   3. Register it in [DetectorType] and [AppProviders].
abstract class BaseDetector {
  /// Human-readable name shown in the Settings UI.
  String get name;

  /// Short description shown below the name.
  String get description;

  /// One-time setup (loads model, allocates interpreter, etc.).
  /// Must be called before [detect].
  Future<void> initialize();

  /// Runs inference on one [InputImage] and returns zero or more [Detection]s.
  /// Must return an empty list (never throw) on transient errors.
  Future<List<Detection>> detect(InputImage image);

  /// Releases all native/platform resources.
  Future<void> close();
}

/// Registry of available detectors.
/// Add a new entry here whenever you add a [BaseDetector] implementation.
enum DetectorType {
  mlKit,
  yolo,
}

extension DetectorTypeLabel on DetectorType {
  String get label {
    switch (this) {
      case DetectorType.mlKit:
        return 'ML Kit';
      case DetectorType.yolo:
        return 'YOLOv8n';
    }
  }
}
