// FILE: lib/services/detection_service.dart

import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

/// Wraps ML Kit's ObjectDetector for both stream frames and static images.
class DetectionService {
  ObjectDetector? _detector;

  /// Call once before using [processImage] or [processInputImage].
  void initialize() {
    final options = ObjectDetectorOptions(
      mode: DetectionMode.stream,          // works for both stream & single
      classifyObjects: true,
      multipleObjects: true,
    );
    _detector = ObjectDetector(options: options);
  }

  bool get isInitialized => _detector != null;

  /// Run detection on an [InputImage] (camera frame or file-based image).
  /// Returns an empty list on error.
  Future<List<DetectedObject>> processInputImage(InputImage image) async {
    if (_detector == null) return [];
    try {
      return await _detector!.processImage(image);
    } catch (e) {
      // Silently ignore transient errors (e.g., frame drop during stream)
      return [];
    }
  }

  /// Convenience: detect from a file path (for gallery images).
  Future<List<DetectedObject>> processFile(String filePath) async {
    final inputImage = InputImage.fromFilePath(filePath);
    return processInputImage(inputImage);
  }

  /// Release native resources. Call in [dispose()].
  Future<void> close() async {
    await _detector?.close();
    _detector = null;
  }
}

/// Extracts a human-readable label from a [DetectedObject].
/// Falls back to "Unknown" when no classification is available.
String labelFor(DetectedObject object) {
  if (object.labels.isEmpty) return 'Unknown';
  // Labels are sorted by confidence descending
  return object.labels.first.text;
}

/// Extracts the top confidence score (0.0 – 1.0) from a [DetectedObject].
double confidenceFor(DetectedObject object) {
  if (object.labels.isEmpty) return 0.0;
  return object.labels.first.confidence;
}
